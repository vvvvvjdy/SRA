import argparse
import copy
from copy import deepcopy
import logging
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from pathlib import Path
from collections import OrderedDict
import json

import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.optimization import get_scheduler

from models.sit import SiT_models
from loss import SRALoss

from dataset import CustomDataset
from diffusers.models import AutoencoderKL

import math
from torchvision.utils import make_grid
from PIL import Image

logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}


    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    requires_grad(ema, False)

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SRALoss(
        prediction=args.prediction,
        path_type=args.path_type,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        block_out_s=args.block_out_s,
        block_out_t=args.block_out_t,
        t_max=args.t_max,
        loss_type=args.loss_type,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    # Setup dataset:
    train_dataset = CustomDataset(args.data_dir_train)


    num_images = len(train_dataset)
    local_batch_size = int(args.batch_size)


    # Create data loaders:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {num_images:,} images ({args.data_dir_train})")
        logger.info(
            f"Total batch size: {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    epoch_start = -1
    if args.resume_ckpt is not None:
        ckpt = torch.load(
            args.resume_ckpt,
            map_location='cpu',
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        epoch_start = ckpt['epoch'] - 1
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        logger.info(f"Starting training experiment: {args.exp_name}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_xs, _ = next(iter(train_dataloader))
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
    )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)


    for epoch in range(epoch_start+1, args.epochs):

        model.train()
        for images_l, y in train_dataloader:
            # save checkpoint (feel free to adjust the frequency)
            if (global_step % args.checkpoint_steps == 0) and global_step > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/step-{global_step}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")


            # sample and save images (feel free to adjust the frequency)
            if (global_step % args.sample_steps == 0) and global_step > 0:
                from samplers import euler_sampler
                with torch.no_grad():
                    model.eval()
                    samples = euler_sampler(
                        model,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale= 4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)

                    samples = vae.decode((samples - latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.

                # Save images locally
                accelerator.wait_for_everyone()
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))

                # Save as grid images
                out_samples = Image.fromarray(array2grid(out_samples))
                gt_samples = Image.fromarray(array2grid(gt_samples))

                if accelerator.is_main_process:
                    base_dir = os.path.join(args.output_dir, args.exp_name)
                    sample_dir = os.path.join(base_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    out_samples.save(f"{sample_dir}/samples_step_{global_step}.png")
                    gt_samples.save(f"{sample_dir}/gt_samples_step_{global_step}.png")
                    logger.info(f"Saved samples at step {global_step}")
                model.train()

            x = images_l.squeeze(dim=1).to(device)
            y = y.to(device)
            labels = y

            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            with accelerator.accumulate(model):
                gen_loss, align_loss = loss_fn(model, x, ema, labels)
                gen_loss_mean = gen_loss.mean()

                # we dynamically adjust the weight of align loss to make two losses at the same scale.
                if args.resolution == 256:
                    align_loss_mean = align_loss.mean() * 2 * (0.1 ** (((epoch - 149) / 1000 + 1) if epoch > 149 else 1))
                else:
                    align_loss_mean = align_loss.mean() * 2 * (0.1 ** (((epoch - 99) / 800 + 1) if epoch > 99 else 1))

                # total loss
                loss = gen_loss_mean + align_loss_mean

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)  # change ema function

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "gen_loss": accelerator.gather(gen_loss_mean).mean().detach().item(),
                "align_loss": accelerator.gather(align_loss_mean).mean().detach().item(),
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        # save checkpoint (feel free to adjust the frequency)
        if (epoch+1) % args.checkpoint_epochs == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "steps": global_step,
                }
                checkpoint_path = f"{checkpoint_dir}/epoch-{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")







        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--sample-steps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=801)
    parser.add_argument("--checkpoint-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-epochs", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=4100000)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir-train", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    # precision
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=8)

    # loss
    parser.add_argument("--loss-type", type=str, default="sml1", choices=["sml1", "l2", "l1"])
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="use class-free guidance if > 0")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])  # currently we only support v-prediction
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--block-out-s", type=int, default=4)
    parser.add_argument("--block-out-t", type=int, default=8)
    parser.add_argument("--t-max", type=float, default=0.2
                        , help="The max time-distance for teacher-student matching.")




    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)

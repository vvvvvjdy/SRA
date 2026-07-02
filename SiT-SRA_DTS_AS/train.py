import json
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from dataset import CustomDataset
from loss import SRALoss
from model import SiT_models


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1.0, latents_bias=0.0):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return z * latents_scale + latents_bias


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    seq_len = latent_size * latent_size // (2 ** 2)

    block_kwargs = {
        "fused_attn": args.fused_attn,
        "qk_norm": args.qk_norm,
        "attention_separation": args.attention_separation,
    }

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        use_alignment_head=args.use_alignment_loss,
        **block_kwargs,
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    requires_grad(ema, False)

    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1).to(device)

    loss_fn = SRALoss(
        prediction=args.prediction,
        path_type=args.path_type,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        block_out_s=args.block_out_s,
        block_out_t=args.block_out_t,
        mask_ratio=args.mask_ratio,
        full_sample_prob=args.full_sample_prob,
        loss_type=args.loss_type,
        use_align_loss=args.use_alignment_loss,
        teacher_t=args.teacher_t,
        teacher_mask=args.teacher_mask,
        dual_time_scheduling=args.dual_time_scheduling,
    )

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = CustomDataset(args.data_dir)
    num_images = len(train_dataset)
    local_batch_size = int(args.batch_size // accelerator.num_processes // args.gradient_accumulation_steps)
    if local_batch_size < 1:
        raise ValueError(
            "batch-size must be at least num_processes * gradient_accumulation_steps"
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {num_images:,} images ({args.data_dir})")
        logger.info(
            f"Total batch size: {args.batch_size} "
            f"(batch size per device: {local_batch_size} x {accelerator.num_processes} devices x "
            f"{args.gradient_accumulation_steps} gradient accumulation steps)"
        )
        log_gen = os.path.join(save_dir, "loss_log", "loss_gen_log.jsonl")
        os.makedirs(os.path.dirname(log_gen), exist_ok=True)
        with open(log_gen, "w") as f:
            f.write("loss for generator\n")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    global_step = 0
    epoch_start = -1
    gt_rand = 0
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        if "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])
        epoch_start = ckpt["epoch"] - 1
        global_step = ckpt["steps"]

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    if accelerator.is_main_process:
        logger.info(f"Starting training experiment: {args.exp_name}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    sample_batch_size = max(1, 32 // accelerator.num_processes)
    gt_xs = next(iter(train_dataloader))[0]
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
    )
    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    xT = torch.randn((ys.size(0), 4, latent_size, latent_size), device=device)

    for epoch in range(epoch_start + 1, args.epochs):
        if global_step > args.max_train_steps // 10:
            args.sample_steps = args.sample_steps * 10

        model.train()
        for images_l, y in train_dataloader:
            if (global_step % args.checkpoint_steps == 0) and global_step > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        "model": unwrapped_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/step-{global_step}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step % args.sample_steps == 0) and global_step > 0:
                from samplers import euler_sampler

                with torch.no_grad():
                    model.eval()
                    samples = euler_sampler(
                        model,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale=4.0,
                        guidance_low=0.0,
                        guidance_high=1.0,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)

                    samples = vae.decode((samples - latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.0
                    gt_samples = (gt_samples + 1) / 2.0

                accelerator.wait_for_everyone()
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))

                out_samples = Image.fromarray(array2grid(out_samples))
                gt_samples = Image.fromarray(array2grid(gt_samples))

                if accelerator.is_main_process:
                    sample_dir = os.path.join(save_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    out_samples.save(f"{sample_dir}/samples_step_{global_step}.png")
                    if gt_rand <= 0:
                        gt_samples.save(f"{sample_dir}/gt_samples.png")
                        gt_rand += 1
                    logger.info(f"Saved samples at step {global_step}")
                model.train()

            x = images_l.squeeze(dim=1).to(device)
            labels = y.to(device)

            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    gen_loss, align_loss = loss_fn(model, x, ema, labels, seq_len=seq_len)
                gen_loss_mean = gen_loss.mean()

                if args.use_alignment_loss:
                    align_loss_mean = align_loss.mean()
                    loss = gen_loss_mean + args.align_weight * align_loss_mean
                else:
                    align_loss_mean = torch.tensor(0.0, device=device)
                    loss = gen_loss_mean

                accelerator.backward(loss)
                grad_norm = torch.tensor(0.0, device=device)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "gen_loss": accelerator.gather(gen_loss_mean).mean().detach().item(),
                "align_loss": accelerator.gather(align_loss_mean).mean().detach().item(),
                "total_loss": accelerator.gather(loss).mean().detach().item(),
                "grad_n": accelerator.gather(torch.as_tensor(grad_norm, device=device)).mean().item(),
                "glo_s": global_step,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.sync_gradients and accelerator.is_main_process:
                with open(log_gen, "a") as f_log_gen:
                    f_log_gen.write(f"{json.dumps(logs)}\n")

            if global_step >= args.max_train_steps:
                break

        if (epoch + 1) % args.checkpoint_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    "model": unwrapped_model.state_dict(),
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

    model.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


if __name__ == "__main__":
    from arguments import parse_args

    args = parse_args()
    main(args)

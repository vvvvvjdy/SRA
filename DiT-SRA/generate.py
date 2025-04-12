# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta
from models.dit import DiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from PIL import Image
import numpy as np
import math
import argparse
from diffusion import create_diffusion





def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)
    # Setup DDP:cd
    dist.init_process_group("nccl",timeout=timedelta(seconds=6000))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    # Load model:
    latent_size = args.resolution // 8
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **block_kwargs
    ).to(device)

    diffusion = create_diffusion(str(args.num_steps))

    # ckpt_path is required!
    ckpt_path = args.ckpt
    if ckpt_path is None:
        raise ValueError("ckpt_path is required!")
    else:
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}')['ema']

    model.load_state_dict(state_dict)
    print(f"{rank} loaded model from {ckpt_path}")
    model.eval()  # important for classifer-free guidance.
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0， if cfg_scale =1.0, it means no cfg"
    using_cfg = args.cfg_scale > 1.0
    if rank == 0:
        print("Using cfg:", using_cfg)

    # Create folder to save samples:

    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    exp_name  = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    folder_name = f"{exp_name}-{ckpt_string_name}-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"



    if rank == 0:
        print(f"Saving samples to {args.sample_dir}/{folder_name}")
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward_without_cfg

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        latents_scale = torch.tensor(
            [0.18215, 0.18215, 0.18215, 0.18215, ]
        ).view(1, 4, 1, 1).to(device)
        latents_bias = -torch.tensor(
            [0., 0., 0., 0., ]
        ).view(1, 4, 1, 1).to(device)
        samples = vae.decode((samples - latents_bias) / latents_scale).sample
        samples = (samples + 1) / 2.
        samples = torch.clamp(
            255. * samples, 0, 255
        ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total += global_batch_size
    dist.barrier()
    print(f"Rank={rank} finished")
    dist.barrier()
    dist.destroy_process_group()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a DiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    # notice that the latent feature decoded by ema is sharper than mse ( more semantic shape)

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode

    # distributed
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')



    args = parser.parse_args()
    main(args)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample images from a pre-trained SiT model using DDP.

The generated png files can be converted to an ADM-compatible .npz file with
npz_convert.py for FID and other evaluation metrics.
"""

import argparse
import math
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from model import SiT_models
from samplers import euler_maruyama_sampler, euler_sampler


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "DDP sampling requires at least one GPU."
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl", timeout=timedelta(seconds=6000))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    block_kwargs = {
        "fused_attn": args.fused_attn,
        "qk_norm": args.qk_norm,
        "attention_separation": args.attention_separation,
    }
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        use_alignment_head=args.use_alignment_loss,
        **block_kwargs,
    )

    if args.ckpt is None:
        raise ValueError("--ckpt is required")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    print(f"{rank} loaded model from {args.ckpt}")

    model = model.to(device).eval()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    assert args.cfg_scale >= 1.0, "cfg_scale should be >= 1.0; cfg_scale=1.0 disables CFG"
    if rank == 0:
        print("Using cfg:", args.cfg_scale > 1.0)

    sample_folder_dir = args.sample_dir
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving png samples to {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)

    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1).to(device)

    total = 0
    for _ in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            num_steps=args.num_steps,
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            mask_ratio=args.mask_ratio,
            mask_type=args.mask_type,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            samples = vae.decode((samples - latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.0
            samples = torch.clamp(255.0 * samples, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

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
    parser.add_argument("--global-seed", type=int, default=0)

    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TF32 matmuls on Ampere GPUs.",
    )

    parser.add_argument("--ckpt", type=str, default=None, help="Path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-alignment-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attention-separation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-ratio", type=float, default=0.0, help="The ratio of mixed tokens for sampling.")
    parser.add_argument("--mask-type", type=str, default="random", choices=["random", "fix"])

    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    parser.add_argument("--mode", type=str, default="sde", choices=["sde", "ode"])
    parser.add_argument("--cfg-scale", type=float, default=1.8)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)

    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    args = parser.parse_args()
    main(args)

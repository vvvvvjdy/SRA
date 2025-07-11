#!/usr/bin/env python3
"""
show_pca.py
Visualize PCA feature maps of SiTs.
"""

import argparse
import os



import matplotlib.pyplot as plt
import torch
from PIL import Image

from diffusers import AutoencoderKL
from getfeature import get_f
from pcav import pca_v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA visualization for SiTs")
    # I/O
    parser.add_argument( "--img_path", type=str, default="macaw.JPEG",
                        help="Path to input image")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Optional: save result to this file; otherwise show with plt.show()")

    # Model
    parser.add_argument("--ckpt", type=str,
                       required=True,
                        help="Path to model checkpoint")

    # Hyper-parameters
    parser.add_argument("--label_id", type=int, default=88,
                        help="ImageNet class id")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 13, 18, 23, 28],
                        help="Layer indices to visualize")
    parser.add_argument("--times", type=float, nargs="+", default=[0.7, 0.5, 0.3, 0.1],
                        help="Sampling timesteps")
    parser.add_argument("--thr", type=float, default=0.00,
                        help="PCA threshold")
    parser.add_argument("--baseline", type=bool, default=False,
                        help="Use SiT baseline model (default: False)")
    return parser.parse_args()


def showpca(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    ).to(device)

    # Load model
    if args.baseline:
        from sito import SiT_models
        model = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        from sit import SiT_models
        model = SiT_models["SiT-XL/2"](
            input_size=32,
            num_classes=1000,
            use_cfg=True,
            fused_attn=False,
            qk_norm=False,
        )
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt["ema"] if "ema" in ckpt else ckpt
        model.load_state_dict(state)

    model.to(device).eval()
    print(f"Loaded model from {args.ckpt}")

    # Load image
    image = Image.open(args.img_path).convert("RGB")

    # Extract features and compute PCA for every timestep
    rgbs = []
    for t in args.times:
        features = get_f(
            model, vae, image, 256, device, args.label_id,
            args.layers, t, baseline=args.baseline
        )
        pcas = pca_v(features, 3, args.thr)
        rgbs.extend(pcas)

    # Plot
    n_layer = len(args.layers)
    n_time = len(args.times)
    fig, axes = plt.subplots(n_time, n_layer, figsize=(n_layer * 2, n_time * 2))
    axes = axes.flatten() if n_time > 1 or n_layer > 1 else [axes]
    for ax, img in zip(axes, rgbs):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path, dpi=150)
        print(f"Saved to {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    showpca(parse_args())
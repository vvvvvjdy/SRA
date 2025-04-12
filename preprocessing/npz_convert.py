import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import argparse
import os

def create_npz_from_sample_folder(sample_dir, num=50_000,save_path=None):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = save_path if save_path else f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, help="Path to the directory containing .png samples.")
    parser.add_argument("--num-fid-samples", type=int, default=50_000, help="Number of samples to convert.")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="sde")
    parser.add_argument("--baseline", type=bool, default=False)
    args = parser.parse_args()
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    if args.baseline:
        folder_name = f"{model_string_name}-baseline-{args.resolution}-vae-{args.vae}-" \
                        f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}"
    else:
        folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                      f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    save_path = f"{args.sample_dir}/sp/{folder_name}.npz"
    create_npz_from_sample_folder(sample_folder_dir, num=args.num_fid_samples)
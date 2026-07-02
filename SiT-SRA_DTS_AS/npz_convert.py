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
    parser.add_argument("--sample-dir-images", type=str, help="Path to the directory containing .png samples.")
    parser.add_argument("--num-fid-samples", type=int, default=50_000, help="Number of samples to convert.")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save the converted samples.")
    args = parser.parse_args()
    sample_folder_dir = args.sample_dir_images
    save_path = args.save_path if args.save_path else sample_folder_dir
    create_npz_from_sample_folder(sample_folder_dir, num=args.num_fid_samples,save_path=save_path)
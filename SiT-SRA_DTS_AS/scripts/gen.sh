#!/bin/bash

NUM_GPUS=8
MASTER_PORT=29501

MODEL="SiT-B/2"
CKPT="[YOUR_CKPT_PATH]"
SAMPLE_ROOT="[YOUR_SAMPLE_DIR]"
IMAGE_DIR="${SAMPLE_ROOT}/images"
NPZ_PATH="${SAMPLE_ROOT}/samples.npz"

PER_PROC_BATCH_SIZE=64
NUM_FID_SAMPLES=50000
PATH_TYPE="linear"
MODE="sde"
NUM_STEPS=250
CFG_SCALE=1.0
GUIDANCE_HIGH=0.7
RESOLUTION=256
VAE="ema"
GLOBAL_SEED=0

mkdir -p "${IMAGE_DIR}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    generate.py \
    --num-fid-samples "${NUM_FID_SAMPLES}" \
    --path-type "${PATH_TYPE}" \
    --per-proc-batch-size "${PER_PROC_BATCH_SIZE}" \
    --mode "${MODE}" \
    --num-steps "${NUM_STEPS}" \
    --cfg-scale "${CFG_SCALE}" \
    --guidance-high "${GUIDANCE_HIGH}" \
    --sample-dir "${IMAGE_DIR}" \
    --model "${MODEL}" \
    --ckpt "${CKPT}" \
    --vae "${VAE}" \
    --resolution "${RESOLUTION}" \
    --global-seed "${GLOBAL_SEED}" \
    --attention-separation \
    --use-alignment-loss

python npz_convert.py \
    --sample-dir-images "${IMAGE_DIR}" \
    --num-fid-samples "${NUM_FID_SAMPLES}" \
    --save-path "${NPZ_PATH}"

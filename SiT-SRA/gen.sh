#!/bin/bash

MODEL="SiT-XL/2"
PER_PROC_BATCH_SIZE=64
NUM_FID_SAMPLES=50000
PATH_TYPE="linear"
MODE="sde"
NUM_STEPS=250
# cfg_scale=1.0 (without CFG),by default we use cfg_scale=1.8 with guidance interval.
CFG_SCALE=1.8
GUIDANCE_HIGH=0.7
RESOLUTION=256
VAE="ema"
GLOBAL_SEED=0
SAMPLE_DIR=[Base directory to save images]
CKPT=[Your checkpoint path]




python -m torch.distributed.launch \
    --nproc_per_node=8 \
    generate.py \
    --num-fid-samples $NUM_FID_SAMPLES \
    --path-type $PATH_TYPE \
    --per-proc-batch-size $PER_PROC_BATCH_SIZE \
    --mode $MODE \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG_SCALE \
    --guidance-high $GUIDANCE_HIGH \
    --sample-dir $SAMPLE_DIR \
    --model $MODEL \
    --ckpt $CKPT \
    --vae $VAE \
    --resolution $RESOLUTION \
    --global-seed $GLOBAL_SEED \


python npz_convert.py \
     --model $MODEL \
    --ckpt $CKPT \
    --sample-dir $SAMPLE_DIR \
    --num-fid-samples $NUM_FID_SAMPLES \
    --resolution $RESOLUTION \
    --vae $VAE \
    --cfg-scale $CFG_SCALE \
    --global-seed $GLOBAL_SEED \
    --mode $MODE \






#!/bin/bash
#!/bin/bash
export MASTER_PORT=19998
export CUDA_VISIBLE_DEVICES=0,1,2,3,

MODEL="DiT-XL/2"
SAMPLE_DIR=[Base directory to save images]
PER_PROC_BATCH_SIZE=64
NUM_FID_SAMPLES=50000
NUM_STEPS=250
# cfg_scale=1.0 (without CFG),by default it set to 1.5
CFG_SCALE=1.0
RESOLUTION=256
VAE="ema"
GLOBAL_SEED=0
SAMPLE_DIR=[Base directory to save images]
CKPT=[Your checkpoint path]



python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    generate.py \
    --num-fid-samples $NUM_FID_SAMPLES \
    --per-proc-batch-size $PER_PROC_BATCH_SIZE \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG_SCALE \
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






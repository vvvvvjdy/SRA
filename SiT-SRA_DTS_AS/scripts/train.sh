#!/bin/bash

CONFIG_FILE="configs/default.yaml"
NUM_PROCS=8
MAIN_PORT=29500

DATA_DIR="[YOUR_DATA_PATH]"
OUTPUT_DIR="exps"
EXP_NAME="sitb-maskr0.25-as-dts-res256"

accelerate launch \
    --config_file "${CONFIG_FILE}" \
    --main_process_port "${MAIN_PORT}" \
    --num_processes "${NUM_PROCS}" \
    train.py \
    --mixed-precision "fp16" \
    --seed 0 \
    --path-type "linear" \
    --prediction "v" \
    --weighting "uniform" \
    --resolution 256 \
    --batch-size 256 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-4 \
    --model "SiT-B/2" \
    --block-out-s 3 \
    --block-out-t 8 \
    --mask-ratio 0.25 \
    --dual-time-scheduling \
    --attention-separation \
    --teacher-mask \
    --teacher-t "sra" \
    --use-alignment-loss \
    --align-weight 0.5 \
    --loss-type "cos" \
    --sample-steps 50000 \
    --checkpoint-steps 100000 \
    --checkpoint-epochs 200 \
    --epochs 801 \
    --max-train-steps 4100000 \
    --output-dir "${OUTPUT_DIR}" \
    --exp-name "${EXP_NAME}" \
    --data-dir "${DATA_DIR}"

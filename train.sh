#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Training settings
TRAIN_SHARDS="${SCRIPT_DIR}/datasets/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets/val/*.tar"
OUTPUT_DIR="${SCRIPT_DIR}/output"
RUN_NUMBER=1

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards "${VAL_SHARDS}" \
    --output_dir "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    --batch_size 6 \
    --num_frozen 0 \
    --lr_g 1e-5 \
    --lr_d 1e-5 \
    --max_train_steps 500000 \
    --gradient_accumulation_steps 8 \
    --max_audio_length 3.0 \
    --lambda_adv 1.0 \
    --lambda_fm 1.0 \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms 1.0 \
    --lambda_d_msd 0.1 \
    --save_every 625 \
    --eval_every 125 \
    --log_every 2 \
    --wandb_project Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}" \
    --mixed_precision bf16

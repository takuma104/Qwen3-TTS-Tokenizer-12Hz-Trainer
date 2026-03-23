#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Training settings
TRAIN_SHARDS="${SCRIPT_DIR}/datasets/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets/val/*.tar"
OUTPUT_DIR="${SCRIPT_DIR}/output"
RUN_NUMBER=25

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards "${VAL_SHARDS}" \
    --output_dir "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    --ref_discriminator_checkpoint "${OUTPUT_DIR}/run10/checkpoint-best-disc" \
    --resume_from "${OUTPUT_DIR}/run24/checkpoint-step-22500" \
    --batch_size 8 \
    --num_decoder_block_frozen 0 \
    --d_reg_every 16 \
    --lr_g 1e-5 \
    --lr_d 2e-5 \
    --beta2_d 0.9 \
    --max_train_steps 1000000 \
    --gradient_accumulation_steps 4 \
    --max_audio_length 3.0 \
    --lambda_adv 0.3 \
    --lambda_fm 3.0 \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms 1.0 \
    --save_every 1250 \
    --eval_every 250 \
    --log_every 3 \
    --wandb_project Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}" \
    --mixed_precision bf16

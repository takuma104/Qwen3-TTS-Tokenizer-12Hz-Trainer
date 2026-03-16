#!/bin/bash

uv run src/evaluate_checkpoints.py \
    --checkpoints \
        output/run_gan12/checkpoint-best \
        output/run_gan13/checkpoint-best \
        output/run9/checkpoint-step-97500 \
        output/run10/checkpoint-best \
        output/run10/checkpoint-step-186250 \
        output/run13/checkpoint-best \
        output/run14/checkpoint-step-246250 \
    --hf_dataset simon3000/genshin-voice \
    --fixed_discriminator output/run9/checkpoint-step-97500 \
    --num_samples 100 \
    --output_dir benchmark_results/


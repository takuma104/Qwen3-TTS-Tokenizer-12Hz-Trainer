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
    --shard_pattern "datasets/val/*.tar" \
    --fixed_discriminator output/run10/checkpoint-best \
    --num_samples 100 \
    --output_dir benchmark_results/


# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Merge trained DecoderBlock weights with base 24kHz model to create a 48kHz model.

The resulting model uses model_type="qwen3_tts_tokenizer_12hz" and can be loaded
with standard AutoModel.from_pretrained() without any custom code.

Usage:
    python finetuning/decoder_block_48k/merge.py \
        --base_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
        --checkpoint output/decoder_block_48k/checkpoint-best \
        --output_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz-v2
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge DecoderBlock weights into 48kHz model"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Base 24kHz model path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Trained checkpoint path (containing decoder_block.safetensors and config.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output 48kHz model path",
    )
    return parser.parse_args()


def resolve_model_path(model_path: str) -> Path:
    """Resolve model path, downloading from HuggingFace Hub if needed."""
    if os.path.exists(model_path):
        return Path(model_path)
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(repo_id=model_path)
    return Path(local_path)


def main():
    args = parse_args()

    print(f"Base model: {args.base_model_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output path: {args.output_path}")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint config
    checkpoint_path = Path(args.checkpoint)
    with open(checkpoint_path / "config.json") as f:
        checkpoint_config = json.load(f)

    new_upsample_rates = checkpoint_config["new_upsample_rates"]
    add_48k_decoder_block = checkpoint_config.get("add_48k_decoder_block", True)
    extra_upsample_rate = (
        checkpoint_config["extra_upsample_rate"] if add_48k_decoder_block else 1
    )
    num_frozen = checkpoint_config["num_frozen_decoder_modules"]

    print(f"New upsample_rates: {new_upsample_rates}")
    print(f"Extra upsample rate: {extra_upsample_rate}")

    # Resolve base model path
    base_path = resolve_model_path(args.base_model_path)

    # Load base model config
    with open(base_path / "config.json") as f:
        config_dict = json.load(f)

    # Update config for 48kHz
    decoder_config = config_dict.get("decoder_config", {})
    decoder_config["upsample_rates"] = new_upsample_rates
    config_dict["decoder_config"] = decoder_config

    # Update sample rates
    # model_type stays as "qwen3_tts_tokenizer_12hz" (upstream compatible)
    config_dict["output_sample_rate"] = (
        config_dict.get("output_sample_rate", 24000) * extra_upsample_rate
    )
    config_dict["decode_upsample_rate"] = (
        config_dict.get("decode_upsample_rate", 1920) * extra_upsample_rate
    )

    # Save updated config
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved config (model_type={config_dict['model_type']})")
    print(f"  output_sample_rate: {config_dict['output_sample_rate']}")
    print(f"  decode_upsample_rate: {config_dict['decode_upsample_rate']}")

    # Load base model weights
    print("\nLoading base model weights...")
    base_model_files = sorted(base_path.glob("*.safetensors"))
    if not base_model_files:
        base_model_files = sorted(base_path.glob("*.bin"))

    base_state_dict = {}
    for model_file in base_model_files:
        if str(model_file).endswith(".safetensors"):
            base_state_dict.update(load_file(str(model_file)))
        else:
            base_state_dict.update(torch.load(str(model_file), map_location="cpu"))
    print(f"Loaded {len(base_state_dict)} keys from base model")

    # Remove old base decoder layers that will be replaced by checkpoint weights
    # (all decoder.decoder.{i} where i >= num_frozen)
    keys_to_remove = []
    for k in base_state_dict:
        if k.startswith("decoder.decoder."):
            parts = k.split(".")
            try:
                idx = int(parts[2])
                if idx >= num_frozen:
                    keys_to_remove.append(k)
            except (ValueError, IndexError):
                pass
    for k in keys_to_remove:
        del base_state_dict[k]
    print(
        f"Removed {len(keys_to_remove)} old base decoder layer keys (decoder.decoder.{num_frozen}+)"
    )

    # Load trained decoder block weights
    print("Loading trained decoder block weights...")
    checkpoint_weights = load_file(str(checkpoint_path / "decoder_block.safetensors"))
    print(f"Loaded {len(checkpoint_weights)} keys from checkpoint")

    # Merge: add trained weights with "decoder." prefix
    for k, v in checkpoint_weights.items():
        base_state_dict[f"decoder.{k}"] = v

    print(f"Merged state dict: {len(base_state_dict)} keys")

    # Save merged model
    output_model_path = output_path / "model.safetensors"
    save_file(base_state_dict, str(output_model_path))
    print(f"Saved merged model to {output_model_path}")

    # Copy other files from base model
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
    ]

    for filename in files_to_copy:
        src_path = base_path / filename
        if src_path.exists():
            shutil.copy(src_path, output_path / filename)
            print(f"Copied {filename}")

    # Verify by loading the model
    print("\nVerifying merged model...")
    try:
        from qwen_tts import Qwen3TTSTokenizer

        model = Qwen3TTSTokenizer.from_pretrained(
            str(output_path),
            attn_implementation="eager",
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        print(f"Model loaded successfully!")
        print(f"  model_type: {model.config.model_type}")
        print(f"  output_sample_rate: {model.config.output_sample_rate}")
        print(f"  decode_upsample_rate: {model.config.decode_upsample_rate}")
        print(f"  upsample_rates: {list(model.model.decoder.config.upsample_rates)}")
        print(f"  total_upsample: {model.model.decoder.total_upsample}")
        print(f"  decoder modules: {len(model.model.decoder.decoder)}")

        total_params = sum(p.numel() for p in model.model.decoder.parameters())
        print(f"  Total decoder parameters: {total_params:,}")

        # Verify total_upsample matches expected
        expected_total = int(
            np.prod(
                new_upsample_rates
                + list(config_dict["decoder_config"]["upsampling_ratios"])
            )
        )
        actual_total = model.model.decoder.total_upsample
        assert (
            actual_total == expected_total
        ), f"total_upsample mismatch: {actual_total} != {expected_total}"
        print(f"  total_upsample verified: {actual_total}")

    except Exception as e:
        print(f"Warning: Model verification failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n48kHz model saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()

# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
DecoderBlock Addition Method - 48kHz Inference Script

Supports two modes:
  1. Merged model: Load a merged 48kHz model directly (no custom code needed)
  2. Checkpoint + base: Restore 48kHz decoder from base model + trained checkpoint

Usage:
    # Method 1: Merged model (recommended)
    python finetuning/decoder_block_48k/inference.py \
        --model_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz-v2 \
        --input_audio input.wav \
        --output_audio output_48k.wav

    # Method 2: Checkpoint + base model
    python finetuning/decoder_block_48k/inference.py \
        --checkpoint output/decoder_block_48k/checkpoint-best \
        --input_audio input.wav \
        --output_audio output_48k.wav

    # Decode from audio_codes (.npy)
    python finetuning/decoder_block_48k/inference.py \
        --model_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz-v2 \
        --input_codes codes.npy \
        --output_audio output_48k.wav
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="48kHz Inference (DecoderBlock method)"
    )

    # Model (choose one)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to merged 48kHz model",
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
        default=None,
        help="Trained checkpoint path",
    )

    # Input (choose one)
    parser.add_argument(
        "--input_audio",
        type=str,
        default=None,
        help="Input audio file (encode → decode to 48kHz)",
    )
    parser.add_argument(
        "--input_codes",
        type=str,
        default=None,
        help="Input audio_codes (.npy, shape: [seq_len, 16])",
    )

    # Output
    parser.add_argument(
        "--output_audio",
        type=str,
        default="output_48k.wav",
        help="Output audio file path",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    return parser.parse_args()


class Qwen3TTSTokenizer48kHz:
    """48kHz tokenizer using the DecoderBlock addition method.

    No custom model code is needed - this uses the standard upstream
    Qwen3TTSTokenizerV2Decoder with extended upsample_rates config.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        checkpoint: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        if model_path:
            self._load_merged_model(model_path)
        elif checkpoint:
            self._load_from_checkpoint(base_model_path, checkpoint)
        else:
            raise ValueError("Either 'model_path' or 'checkpoint' must be specified")

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

    def _load_merged_model(self, model_path: str):
        """Load merged 48kHz model (standard upstream loading)."""
        print(f"Loading merged 48kHz model from {model_path}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            model_path,
            dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )
        self.output_sample_rate = self.tokenizer.get_output_sample_rate()
        print(f"Model loaded. Output sample rate: {self.output_sample_rate} Hz")
        print(
            f"  upsample_rates: "
            f"{list(self.tokenizer.model.decoder.config.upsample_rates)}"
        )

    def _load_from_checkpoint(self, base_model_path: str, checkpoint: str):
        """Load base model and apply trained checkpoint weights."""
        print(f"Loading base model from {base_model_path}...")

        # Load base model
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            base_model_path,
            dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )

        # Load checkpoint config
        checkpoint_path = Path(checkpoint)
        with open(checkpoint_path / "config.json") as f:
            ckpt_config = json.load(f)

        new_upsample_rates = ckpt_config["new_upsample_rates"]
        add_48k_decoder_block = ckpt_config.get("add_48k_decoder_block", True)
        extra_upsample_rate = (
            ckpt_config["extra_upsample_rate"] if add_48k_decoder_block else 1
        )

        print(f"Checkpoint config: upsample_rates={new_upsample_rates}")

        # Create new decoder config
        base_decoder_config = self.tokenizer.model.decoder.config
        config_dict = base_decoder_config.to_dict()
        config_dict["upsample_rates"] = new_upsample_rates
        for key in ("model_type", "transformers_version"):
            config_dict.pop(key, None)

        new_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)

        # Create new decoder and load base weights
        new_decoder = Qwen3TTSTokenizerV2Decoder(new_config)
        new_decoder.load_state_dict(
            self.tokenizer.model.decoder.state_dict(), strict=False
        )

        # Load trained checkpoint weights
        print(f"Loading checkpoint weights from {checkpoint}...")
        checkpoint_weights = load_file(
            str(checkpoint_path / "decoder_block.safetensors")
        )
        missing, unexpected = new_decoder.load_state_dict(
            checkpoint_weights, strict=False
        )
        print(f"  Loaded {len(checkpoint_weights)} checkpoint keys")

        # Replace decoder
        new_decoder = new_decoder.to(self.device).to(self.dtype)
        new_decoder.eval()
        self.tokenizer.model.decoder = new_decoder

        # Update sample rates
        new_output_rate = self.tokenizer.config.output_sample_rate * extra_upsample_rate
        new_decode_upsample_rate = (
            self.tokenizer.config.decode_upsample_rate * extra_upsample_rate
        )
        self.tokenizer.config.output_sample_rate = new_output_rate
        self.tokenizer.config.decode_upsample_rate = new_decode_upsample_rate
        self.tokenizer.model.output_sample_rate = new_output_rate
        self.tokenizer.model.decode_upsample_rate = new_decode_upsample_rate

        self.output_sample_rate = new_output_rate
        print(f"48kHz decoder ready. Output sample rate: {self.output_sample_rate} Hz")

    def encode(self, audio_path: str):
        """Encode audio file to audio_codes."""
        return self.tokenizer.encode(audio_path, return_dict=True)

    def decode(self, encoded) -> Tuple[List[np.ndarray], int]:
        """Decode audio_codes to 48kHz waveform."""
        return self.tokenizer.decode(encoded)

    def decode_from_codes(
        self, audio_codes: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[List[np.ndarray], int]:
        """Decode from raw audio_codes (shape: [seq_len, 16] or [batch, seq_len, 16])."""
        if isinstance(audio_codes, np.ndarray):
            audio_codes = torch.from_numpy(audio_codes).long()
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(0)
        return self.tokenizer.decode({"audio_codes": audio_codes})

    def encode_decode(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Round-trip: encode audio → decode to 48kHz."""
        encoded = self.encode(audio_path)
        wavs, sr = self.decode(encoded)
        return wavs[0], sr

    def get_output_sample_rate(self) -> int:
        return self.output_sample_rate


def main():
    args = parse_args()

    if args.input_audio is None and args.input_codes is None:
        print("Error: Either --input_audio or --input_codes must be specified")
        sys.exit(1)

    if args.model_path is None and args.checkpoint is None:
        print("Error: Either --model_path or --checkpoint must be specified")
        sys.exit(1)

    # Load model
    print("=" * 50)
    print("Initializing 48kHz Tokenizer (DecoderBlock method)")
    print("=" * 50)

    tokenizer = Qwen3TTSTokenizer48kHz(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )

    # Inference
    print("\n" + "=" * 50)
    print("Running Inference")
    print("=" * 50)

    if args.input_audio:
        print(f"Input audio: {args.input_audio}")
        wav, sr = tokenizer.encode_decode(args.input_audio)
    else:
        print(f"Input codes: {args.input_codes}")
        audio_codes = np.load(args.input_codes)
        print(f"Audio codes shape: {audio_codes.shape}")
        wavs, sr = tokenizer.decode_from_codes(audio_codes)
        wav = wavs[0]

    # Save
    print(f"\nOutput sample rate: {sr} Hz")
    print(f"Output duration: {len(wav) / sr:.2f} seconds")
    print(f"Saving to: {args.output_audio}")

    sf.write(args.output_audio, wav, sr)

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()

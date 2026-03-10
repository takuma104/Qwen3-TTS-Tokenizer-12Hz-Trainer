# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3TTSTokenizerV2Decoder Fine-tuning Script

Supports GAN training (MPD + MSD) and/or reconstruction-only training.
Optionally adds a 48kHz decoder block on top of the base 24kHz decoder.
Can train only the new decoder blocks or the entire decoder.

Usage:
    # Single GPU (GAN + 48kHz decoder block)
    python trainer.py \
        --train_shards "data/train-{000000..000010}.tar" \
        --output_dir output/run1

    # Reconstruction only (no GAN)
    python trainer.py \
        --train_shards "data/train-*.tar" \
        --no-use_gan \
        --output_dir output/run1

    # Train full decoder (24kHz, no GAN)
    python trainer.py \
        --train_shards "data/train-*.tar" \
        --no-use_gan \
        --no-add_48k_decoder_block \
        --train_full_decoder \
        --output_dir output/run1
"""

import argparse
import gc
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress MPS-backend STFT resize deprecation warning (PyTorch internal bug, harmless)
warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized",
    module="torch.functional",
)

# Add project root to path (for `import xcodec2`)
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add xcodec2 dir to path (for xcodec2's internal bare imports)
sys.path.insert(0, str(Path(__file__).parent.parent / "xcodec2"))

from xcodec2.criterions import (
    MultiResolutionMelSpectrogramLoss,
)
from xcodec2.module import (
    HiFiGANMultiPeriodDiscriminator,
    SpecDiscriminator,
)
from dataset import create_webdataset_loader
from losses import (
    global_rms_loss,
    generator_adversarial_loss,
    discriminator_loss,
    feature_matching_loss,
)
from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

BASE_SAMPLE_RATE = 24_000  # Hz, base Qwen3-TTS-Tokenizer output rate


def expand_shards(path: str, print_fn=print) -> "str | list[str]":
    """Expand glob wildcards to a sorted file list, or return the path unchanged."""
    if "*" in path and "{" not in path:
        expanded = sorted(glob.glob(path))
        if not expanded:
            print_fn(f"Error: No files found matching pattern: {path}")
            sys.exit(1)
        print_fn(f"Found {len(expanded)} tar files")
        return expanded
    return path


def align_audio(
    pred_audio: torch.Tensor, target_audio: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Squeeze channel dim if present and truncate both tensors to the same length."""
    pred = pred_audio.squeeze(1) if pred_audio.dim() == 3 else pred_audio
    target = target_audio.squeeze(1) if target_audio.dim() == 3 else target_audio
    min_len = min(pred.shape[-1], target.shape[-1])
    return pred[..., :min_len], target[..., :min_len], min_len


def apply_length_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    min_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-out padding regions beyond each sample's valid length."""
    mask = torch.arange(min_len, device=pred.device)[None, :] < lengths[:, None]
    return pred * mask, target * mask


def compute_grad_norm(module: nn.Module) -> float:
    """Compute L2 gradient norm across all parameters with gradients."""
    return (
        sum(
            p.grad.norm().item() ** 2 for p in module.parameters() if p.grad is not None
        )
        ** 0.5
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3TTSTokenizerV2Decoder fine-tuning script"
    )

    # Data
    parser.add_argument(
        "--train_shards",
        type=str,
        required=True,
        help="WebDataset shard pattern for training data",
    )
    parser.add_argument(
        "--val_shards",
        type=str,
        default=None,
        help="WebDataset shard pattern for validation data",
    )

    # Model
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Base 24kHz decoder model path",
    )
    parser.add_argument(
        "--extra_upsample_rate",
        type=int,
        default=2,
        help="Additional upsample rate to append when --add_48k_decoder_block is set (default: 2 for 48kHz)",
    )
    parser.add_argument(
        "--num_decoder_block_frozen",
        type=int,
        default=None,
        help="Number of decoder blocks to freeze (default: base_num_decoder_modules - 2). Ignored when --train_full_decoder is set.",
    )
    parser.add_argument(
        "--use_gan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GAN training (MPD + MSD discriminators). Use --no-use_gan to disable.",
    )
    parser.add_argument(
        "--add_48k_decoder_block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append extra_upsample_rate to upsample_rates to target 48kHz. Use --no-add_48k_decoder_block to fine-tune the base 24kHz decoder.",
    )
    parser.add_argument(
        "--train_full_decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the entire Qwen3TTSTokenizerV2Decoder (including pre_conv, pre_transformer, upsample). When False, only trains the new/unfrozen decoder blocks.",
    )

    # Checkpoint resume
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Resume training from a checkpoint directory. "
            "Always loads decoder_block.safetensors (generator weights). "
            "Also restores discriminator and optimizer/scheduler states if present."
        ),
    )

    parser.add_argument(
        "--no_resume_optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not resume optimizer state when resuming from a checkpoint.",
    )

    # Training settings
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--lr_g", type=float, default=1e-4, help="Generator learning rate"
    )
    parser.add_argument(
        "--lr_d", type=float, default=2e-4, help="Discriminator learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of warmup steps (lr goes from 0 to target lr)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # GAN loss weights
    parser.add_argument(
        "--lambda_adv", type=float, default=1.0, help="Adversarial loss weight"
    )
    parser.add_argument(
        "--lambda_fm", type=float, default=1.0, help="Feature matching loss weight"
    )
    parser.add_argument(
        "--lambda_d_mpd", type=float, default=1.0, help="MPD discriminator loss weight"
    )
    parser.add_argument(
        "--lambda_d_msd", type=float, default=1.0, help="MSD discriminator loss weight"
    )
    parser.add_argument(
        "--lambda_multi_res_mel",
        type=float,
        default=15.0,
        help="Multi-resolution mel loss weight (inworld-ai style, 7 scales). 0=disabled",
    )
    parser.add_argument(
        "--lambda_global_rms",
        type=float,
        default=1.0,
        help="Global dB RMS loss weight (inworld-ai style). 0=disabled",
    )

    # Data settings
    parser.add_argument(
        "--max_audio_length",
        type=float,
        default=5.0,
        help="Maximum audio length (seconds)",
    )
    parser.add_argument(
        "--min_audio_length",
        type=float,
        default=1.0,
        help="Minimum audio length (seconds)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of DataLoader workers"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/decoder_block_48k_gan",
        help="Output directory",
    )
    parser.add_argument(
        "--save_every", type=int, default=5000, help="Checkpoint save interval (steps)"
    )
    parser.add_argument(
        "--eval_every", type=int, default=1000, help="Evaluation interval (steps)"
    )
    parser.add_argument(
        "--log_every", type=int, default=10, help="Log output interval (steps)"
    )

    # Logging
    parser.add_argument("--log_with", type=str, default="wandb", help="Logging method")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen3-tts-decoder-block-48k",
        help="WandB project",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="WandB run name"
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None, help="Maximum training steps"
    )

    return parser.parse_args()




class DecoderTrainingWrapper(nn.Module):
    """Wraps Qwen3TTSTokenizerV2Decoder for efficient training.

    When train_full_decoder=False: runs frozen layers under torch.no_grad() to save VRAM,
    and only computes gradients for the unfrozen decoder blocks.
    When train_full_decoder=True: runs the entire decoder with gradients.
    """

    def __init__(
        self,
        decoder: Qwen3TTSTokenizerV2Decoder,
        num_frozen_decoder_modules: int,
        train_full_decoder: bool = False,
    ):
        super().__init__()
        self.decoder = decoder
        self.num_frozen = num_frozen_decoder_modules
        self.train_full_decoder = train_full_decoder

    def forward(self, codes):
        if codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError(
                f"Expected {self.decoder.config.num_quantizers} layers of codes, "
                f"got {codes.shape[1]}"
            )

        if self.train_full_decoder:
            # Full decoder with gradients
            hidden = self.decoder.quantizer.decode(codes)
            hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
            hidden = self.decoder.pre_transformer(
                inputs_embeds=hidden
            ).last_hidden_state
            hidden = hidden.permute(0, 2, 1)
            for blocks in self.decoder.upsample:
                for block in blocks:
                    hidden = block(hidden)
            wav = hidden
            for block in self.decoder.decoder:
                wav = block(wav)
        else:
            # Frozen part: no_grad for VRAM savings
            with torch.no_grad():
                hidden = self.decoder.quantizer.decode(codes)
                hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
                hidden = self.decoder.pre_transformer(
                    inputs_embeds=hidden
                ).last_hidden_state
                hidden = hidden.permute(0, 2, 1)
                for blocks in self.decoder.upsample:
                    for block in blocks:
                        hidden = block(hidden)
                wav = hidden
                for block in self.decoder.decoder[: self.num_frozen]:
                    wav = block(wav)
            wav = wav.detach()

            # Trainable part: gradients enabled
            for block in self.decoder.decoder[self.num_frozen :]:
                wav = block(wav)

        return wav.clamp(min=-1, max=1)


def create_model(args, accelerator):
    """Create decoder model, optionally adding 48kHz decoder block."""
    accelerator.print(f"Loading base decoder from {args.decoder_model_path}...")

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.decoder_model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_decoder = tokenizer.model.decoder
    base_state_dict = base_decoder.state_dict()
    base_num_decoder_modules = len(base_decoder.decoder)

    accelerator.print(
        f"Base decoder: upsample_rates={list(base_decoder.config.upsample_rates)}, "
        f"decoder modules={base_num_decoder_modules}"
    )

    # Build config (optionally adding 48kHz decoder block)
    config_dict = base_decoder.config.to_dict()
    base_upsample_rates = list(config_dict["upsample_rates"])
    if args.add_48k_decoder_block:
        new_upsample_rates = base_upsample_rates + [args.extra_upsample_rate]
        config_dict["upsample_rates"] = new_upsample_rates
        accelerator.print(f"New upsample_rates (48kHz): {new_upsample_rates}")
    else:
        new_upsample_rates = base_upsample_rates
        accelerator.print(
            f"Using base upsample_rates (no 48k block): {base_upsample_rates}"
        )
    for key in ("model_type", "transformers_version"):
        config_dict.pop(key, None)

    decoder_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
    if accelerator.device.type == "cuda":
        decoder_config._attn_implementation = "flash_attention_2"

    decoder = Qwen3TTSTokenizerV2Decoder(decoder_config).to(torch.bfloat16)
    missing_keys, unexpected_keys = decoder.load_state_dict(
        base_state_dict, strict=False
    )
    accelerator.print(
        f"Weight loading: {len(missing_keys)} missing keys (new blocks), "
        f"{len(unexpected_keys)} unexpected keys (old final layers)"
    )

    del tokenizer, base_decoder, base_state_dict
    gc.collect()

    # Freeze/unfreeze parameters
    if args.train_full_decoder:
        if args.num_decoder_block_frozen is not None:
            accelerator.print(
                "WARNING: --num_decoder_block_frozen is ignored when --train_full_decoder is set."
            )
        for param in decoder.parameters():
            param.requires_grad = True
        num_frozen = 0
    else:
        if args.num_decoder_block_frozen is not None:
            num_frozen = args.num_decoder_block_frozen
            if num_frozen < 0 or num_frozen > len(decoder.decoder):
                raise ValueError(
                    f"--num_decoder_block_frozen must be in [0, {len(decoder.decoder)}], got {num_frozen}"
                )
        else:
            num_frozen = base_num_decoder_modules - 2
        for param in decoder.parameters():
            param.requires_grad = False
        for i in range(num_frozen, len(decoder.decoder)):
            for param in decoder.decoder[i].parameters():
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in decoder.parameters())
    accelerator.print(
        f"Generator trainable: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params * 100:.4f}%)"
    )

    wrapper = DecoderTrainingWrapper(
        decoder, num_frozen, train_full_decoder=args.train_full_decoder
    )

    # Load generator weights from checkpoint
    if args.resume_from:
        checkpoint_path = Path(args.resume_from) / "decoder_block.safetensors"
        if checkpoint_path.exists():
            accelerator.print(f"Loading generator weights from {checkpoint_path}...")
            trained_weights = load_file(str(checkpoint_path))
            missing, unexpected = decoder.load_state_dict(trained_weights, strict=False)
            accelerator.print(
                f"Generator weights loaded: {len(missing)} missing, "
                f"{len(unexpected)} unexpected"
            )
        else:
            accelerator.print(
                f"WARNING: decoder_block.safetensors not found at {checkpoint_path}"
            )

    return wrapper, num_frozen, base_upsample_rates, new_upsample_rates


def create_discriminators(accelerator):
    """
    Create MPD and SpecDiscriminator discriminators.
    Parameters ported from inworld-ai/tts (48kHz training)
    """

    mpd = HiFiGANMultiPeriodDiscriminator(
        periods=[2, 3, 5, 7, 11],
        max_downsample_channels=512,
        channels=16,
        channel_increasing_factor=4,
    )

    msd = SpecDiscriminator(
        stft_params={
            "fft_sizes": [78, 126, 206, 334, 542, 876, 1418, 2296],
            "hop_sizes": [39, 63, 103, 167, 271, 438, 709, 1148],
            "win_lengths": [78, 126, 206, 334, 542, 876, 1418, 2296],
            "window": "hann_window",
        },
        in_channels=1,
        out_channels=1,
        kernel_sizes=[5, 3],
        channels=32,
        max_downsample_channels=512,
        downsample_scales=[2, 2, 2],
        use_weight_norm=True,
    )

    mpd_params = sum(p.numel() for p in mpd.parameters())
    msd_params = sum(p.numel() for p in msd.parameters())
    accelerator.print(
        f"Discriminator params: MPD={mpd_params:,}, SpecDisc={msd_params:,}, "
        f"Total={mpd_params + msd_params:,}"
    )

    return mpd, msd


@torch.no_grad()
def eval_step(
    model: nn.Module,
    mel_loss_fn: MultiResolutionMelSpectrogramLoss,
    dataloader: DataLoader,
    accelerator: Accelerator,
    max_batches: int = 50,
) -> dict:
    """Evaluation (mel loss only)."""
    model.eval()

    total_mel_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if num_batches >= max_batches:
            break

        audio_codes = batch["audio_codes"].to(accelerator.device).transpose(1, 2)
        target_audio = batch["audio"].to(accelerator.device)
        audio_lengths = batch["audio_lengths"].to(accelerator.device)

        pred_48k = model(audio_codes)

        # Align shapes and mask padding
        pred, target, min_len = align_audio(pred_48k, target_audio)
        pred, target = apply_length_mask(pred, target, audio_lengths, min_len)

        mel_loss = mel_loss_fn(pred, target)
        total_mel_loss += mel_loss.item()
        num_batches += 1

    model.train()
    return {"val/loss_multi_res_mel": total_mel_loss / max(num_batches, 1)}


def save_checkpoint(
    model: nn.Module,
    mpd: "nn.Module | None",
    msd: "nn.Module | None",
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: "torch.optim.Optimizer | None",
    scheduler_g,
    scheduler_d,
    step: int,
    epoch: int,
    args,
    accelerator: Accelerator,
    num_frozen: int,
    base_upsample_rates: list,
    new_upsample_rates: list,
    is_best: bool = False,
):
    """Save checkpoint (generator weights + optional discriminator + training state)."""
    if not accelerator.is_main_process:
        return

    output_dir = Path(args.output_dir)
    checkpoint_name = "checkpoint-best" if is_best else f"checkpoint-step-{step}"
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generator trainable weights (merge.py compatible)
    unwrapped_model = accelerator.unwrap_model(model)
    decoder = unwrapped_model.decoder
    if args.train_full_decoder:
        # Save entire decoder state
        trainable_state_dict = {k: v.cpu() for k, v in decoder.state_dict().items()}
    else:
        trainable_state_dict = {}
        for k, v in decoder.state_dict().items():
            parts = k.split(".", 2)
            if (
                parts[0] == "decoder"
                and len(parts) > 1
                and parts[1].isdigit()
                and int(parts[1]) >= num_frozen
            ):
                trainable_state_dict[k] = v.cpu()
    save_file(trainable_state_dict, str(checkpoint_dir / "decoder_block.safetensors"))

    # Discriminator weights (only when GAN is enabled)
    if args.use_gan and mpd is not None and msd is not None:
        unwrapped_mpd = accelerator.unwrap_model(mpd)
        unwrapped_msd = accelerator.unwrap_model(msd)
        torch.save(
            {
                "mpd": unwrapped_mpd.state_dict(),
                "msd": unwrapped_msd.state_dict(),
            },
            checkpoint_dir / "discriminator.pt",
        )

    # Training state
    torch.save(
        {
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict() if optimizer_d else None,
            "scheduler_g": scheduler_g.state_dict() if scheduler_g else None,
            "scheduler_d": scheduler_d.state_dict() if scheduler_d else None,
            "step": step,
            "epoch": epoch,
        },
        checkpoint_dir / "training_state.pt",
    )

    # Config
    config_dict = {
        "base_upsample_rates": base_upsample_rates,
        "new_upsample_rates": new_upsample_rates,
        "extra_upsample_rate": args.extra_upsample_rate,
        "num_frozen_decoder_modules": num_frozen,
        "use_gan": args.use_gan,
        "add_48k_decoder_block": args.add_48k_decoder_block,
        "train_full_decoder": args.train_full_decoder,
        "step": step,
        "epoch": epoch,
        "training_type": "gan" if args.use_gan else "reconstruction",
        "lambda_adv": args.lambda_adv,
        "lambda_fm": args.lambda_fm,
        "lambda_d_mpd": args.lambda_d_mpd,
        "lambda_d_msd": args.lambda_d_msd,
        "lambda_multi_res_mel": args.lambda_multi_res_mel,
        "lambda_global_rms": args.lambda_global_rms,
    }
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    accelerator.print(f"Saved checkpoint to {checkpoint_dir}")


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_dir=args.output_dir,
    )

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create generator
    model, num_frozen, base_upsample_rates, new_upsample_rates = create_model(
        args, accelerator
    )

    # Create discriminators (only when GAN is enabled)
    if args.use_gan:
        mpd, msd = create_discriminators(accelerator)
    else:
        mpd, msd = None, None
        accelerator.print("GAN disabled: skipping discriminator creation.")

    # Mel loss (reconstruction component)
    target_sample_rate = BASE_SAMPLE_RATE * (
        args.extra_upsample_rate if args.add_48k_decoder_block else 1
    )
    multi_res_mel_loss_fn = MultiResolutionMelSpectrogramLoss(
        sample_rate=target_sample_rate
    ).to(accelerator.device)

    # Training data
    accelerator.print(f"Loading training data: {args.train_shards}...")
    shard_pattern = expand_shards(args.train_shards, accelerator.print)

    train_dataloader = create_webdataset_loader(
        shard_pattern=shard_pattern,
        target_sample_rate=target_sample_rate,
        max_audio_length=args.max_audio_length,
        min_audio_length=args.min_audio_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=1000,
    )

    # Validation data (optional)
    val_dataloader = None
    if args.val_shards:
        shard_pattern = expand_shards(args.val_shards, accelerator.print)

        val_dataloader = create_webdataset_loader(
            shard_pattern=shard_pattern,
            target_sample_rate=target_sample_rate,
            max_audio_length=args.max_audio_length,
            min_audio_length=args.min_audio_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_buffer=0,
        )

    # Separate optimizers for G and D
    optimizer_g = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_g,
        betas=(0.8, 0.99),
        weight_decay=args.weight_decay,
    )
    if args.use_gan:
        optimizer_d = AdamW(
            list(mpd.parameters()) + list(msd.parameters()),
            lr=args.lr_d,
            betas=(0.8, 0.99),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer_d = None

    # Schedulers
    if args.max_train_steps:
        total_steps = args.max_train_steps
    else:
        try:
            total_steps = (
                len(train_dataloader)
                * args.num_epochs
                // args.gradient_accumulation_steps
            )
        except TypeError:
            accelerator.print(
                "WARNING: Cannot determine dataset length. "
                "Please specify --max_train_steps."
            )
            total_steps = 500000

    warmup_steps = args.warmup_steps
    cosine_steps_g = max(1, total_steps - warmup_steps)
    if warmup_steps > 0:
        scheduler_g = SequentialLR(
            optimizer_g,
            schedulers=[
                LinearLR(optimizer_g, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer_g, T_max=cosine_steps_g, eta_min=args.lr_g * 0.1),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler_g = CosineAnnealingLR(
            optimizer_g, T_max=total_steps, eta_min=args.lr_g * 0.1
        )
    if args.use_gan:
        cosine_steps_d = max(1, total_steps - warmup_steps)
        if warmup_steps > 0:
            scheduler_d = SequentialLR(
                optimizer_d,
                schedulers=[
                    LinearLR(optimizer_d, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                    CosineAnnealingLR(optimizer_d, T_max=cosine_steps_d, eta_min=args.lr_d * 0.1),
                ],
                milestones=[warmup_steps],
            )
        else:
            scheduler_d = CosineAnnealingLR(
                optimizer_d, T_max=total_steps, eta_min=args.lr_d * 0.1
            )
    else:
        scheduler_d = None
    accelerator.print(f"Total training steps: {total_steps}")

    # Prepare with Accelerate
    model, optimizer_g, train_dataloader, scheduler_g = accelerator.prepare(
        model, optimizer_g, train_dataloader, scheduler_g
    )
    if args.use_gan:
        mpd, optimizer_d, scheduler_d = accelerator.prepare(
            mpd, optimizer_d, scheduler_d
        )
        msd = accelerator.prepare(msd)
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Cache discriminator params and dtype (used repeatedly in training loop)
    if args.use_gan:
        disc_params = list(mpd.parameters()) + list(msd.parameters())
        disc_dtype = next(mpd.parameters()).dtype
    else:
        disc_params = []
        disc_dtype = next(model.parameters()).dtype

    # Initialize tracker
    if args.log_with and accelerator.is_main_process:
        tracker_config = {
            "batch_size": args.batch_size,
            "lr_g": args.lr_g,
            "lr_d": args.lr_d if args.use_gan else None,
            "use_gan": args.use_gan,
            "add_48k_decoder_block": args.add_48k_decoder_block,
            "train_full_decoder": args.train_full_decoder,
            "lambda_adv": args.lambda_adv,
            "lambda_fm": args.lambda_fm,
            "lambda_d_mpd": args.lambda_d_mpd,
            "lambda_d_msd": args.lambda_d_msd,
            "lambda_multi_res_mel": args.lambda_multi_res_mel,
            "lambda_global_rms": args.lambda_global_rms,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "extra_upsample_rate": args.extra_upsample_rate,
            "max_audio_length": args.max_audio_length,
            "training_type": "gan" if args.use_gan else "reconstruction",
        }
        if args.log_with == "wandb":
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=tracker_config,
                init_kwargs={
                    "wandb": {
                        "name": args.wandb_run_name,
                        "entity": args.wandb_entity,
                        "dir": args.output_dir,
                    }
                },
            )
        else:
            accelerator.init_trackers(
                project_name=args.wandb_project, config=tracker_config
            )
    elif args.log_with:
        accelerator.init_trackers(project_name=args.wandb_project)

    # Resume training state (generator weights already loaded in create_model())
    start_step = 0
    start_epoch = 0
    if args.resume_from:
        accelerator.print(f"Resuming training state from {args.resume_from}...")
        checkpoint_dir = Path(args.resume_from)

        # Load checkpoint config to check num_frozen compatibility
        prev_num_frozen = None
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                ckpt_config = json.load(f)
            prev_num_frozen = ckpt_config.get("num_frozen_decoder_modules")

        num_frozen_changed = (
            prev_num_frozen is not None and prev_num_frozen != num_frozen
        )
        if num_frozen_changed:
            accelerator.print(
                f"NOTE: num_frozen changed ({prev_num_frozen} -> {num_frozen}). "
                f"Generator optimizer/scheduler state will NOT be restored."
            )

        # Load discriminator weights (only when GAN is enabled)
        disc_path = checkpoint_dir / "discriminator.pt"
        if args.use_gan and disc_path.exists():
            disc_state = torch.load(disc_path, map_location="cpu")
            accelerator.unwrap_model(mpd).load_state_dict(disc_state["mpd"])
            accelerator.unwrap_model(msd).load_state_dict(disc_state["msd"])

        # Load training state
        training_state = torch.load(
            checkpoint_dir / "training_state.pt", map_location="cpu"
        )
        start_step = training_state["step"]
        start_epoch = training_state["epoch"]

        if not args.no_resume_optimizer:
            if not num_frozen_changed:
                optimizer_g.load_state_dict(training_state["optimizer_g"])
                if training_state["scheduler_g"] and scheduler_g:
                    scheduler_g.load_state_dict(training_state["scheduler_g"])

            # Discriminator optimizer/scheduler is always restored (unaffected by num_frozen)
            if (
                args.use_gan
                and optimizer_d is not None
                and training_state.get("optimizer_d")
            ):
                optimizer_d.load_state_dict(training_state["optimizer_d"])
            if args.use_gan and training_state.get("scheduler_d") and scheduler_d:
                scheduler_d.load_state_dict(training_state["scheduler_d"])

        accelerator.print(f"Resumed from step {start_step}, epoch {start_epoch}")

    # Training loop
    global_step = start_step
    best_val_loss = float("inf")

    model.train()
    if args.use_gan:
        mpd.train()
        msd.train()

    # Persistent across optimizer-step logs.
    mpd_grad_norm = 0.0
    msd_grad_norm = 0.0
    total_audio_sec = 0
    for epoch in range(start_epoch, args.num_epochs):
        accelerator.print(f"\n{'=' * 50}")
        accelerator.print(f"Epoch {epoch + 1}/{args.num_epochs}")
        accelerator.print(f"{'=' * 50}")

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar):
            audio_codes = batch["audio_codes"].to(accelerator.device).transpose(1, 2)
            target_audio = batch["audio"].to(accelerator.device)
            audio_lengths = batch["audio_lengths"].to(accelerator.device)

            # Generator forward
            pred_48k = model(audio_codes)

            # Align shapes for loss computation
            pred, target, min_len = align_audio(pred_48k, target_audio)

            # Mask padding region
            pred, target = apply_length_mask(pred, target, audio_lengths, min_len)

            # Initialize GAN loss placeholders
            loss_d = pred.new_zeros(())
            loss_d_mpd = pred.new_zeros(())
            loss_d_msd = pred.new_zeros(())
            loss_g_adv = pred.new_zeros(())
            loss_fm = pred.new_zeros(())
            dr_mpd = pred.new_zeros(())
            dg_mpd = pred.new_zeros(())
            dr_msd = pred.new_zeros(())
            dg_msd = pred.new_zeros(())

            if args.use_gan:
                # Reshape to (B, 1, T) for discriminators
                pred_wav = pred.unsqueeze(1)
                target_wav = target.unsqueeze(1)

                # Align dtype with discriminator params (handles generator checkpoint
                # loaded in bf16 when mixed_precision=no)
                pred_wav = pred_wav.to(dtype=disc_dtype)
                target_wav = target_wav.to(dtype=disc_dtype)

            # Update D and G under a single accumulation context so
            # `accelerator.sync_gradients` is aligned for both.
            accumulate_models = [model] + ([mpd, msd] if args.use_gan else [])
            with accelerator.accumulate(*accumulate_models):
                if args.use_gan:
                    # =====================
                    # Discriminator update
                    # =====================
                    # MPD
                    mpd_real_outputs = mpd(target_wav)
                    mpd_fake_outputs = mpd(pred_wav.detach())
                    loss_d_mpd, dr_mpd, dg_mpd = discriminator_loss(
                        mpd_real_outputs, mpd_fake_outputs
                    )

                    # MSD
                    msd_real_outputs = msd(target_wav)
                    msd_fake_outputs = msd(pred_wav.detach())
                    loss_d_msd, dr_msd, dg_msd = discriminator_loss(
                        msd_real_outputs, msd_fake_outputs
                    )

                    loss_d = (
                        args.lambda_d_mpd * loss_d_mpd + args.lambda_d_msd * loss_d_msd
                    )

                    optimizer_d.zero_grad()
                    accelerator.backward(loss_d)
                    # Capture per-model gradient norms immediately before D step.
                    if accelerator.sync_gradients:
                        mpd_grad_norm = compute_grad_norm(mpd)
                        msd_grad_norm = compute_grad_norm(msd)
                    accelerator.clip_grad_norm_(disc_params, args.max_grad_norm)
                    optimizer_d.step()
                    scheduler_d.step()

                # =====================
                # Generator update
                # =====================
                if args.use_gan:
                    # MPD (real outputs computed without grad for FM loss)
                    mpd_fake_outputs_g = mpd(pred_wav)
                    with torch.no_grad():
                        mpd_real_outputs_g = mpd(target_wav)
                    loss_g_adv_mpd = generator_adversarial_loss(mpd_fake_outputs_g)
                    loss_fm_mpd = feature_matching_loss(
                        mpd_real_outputs_g, mpd_fake_outputs_g
                    )

                    # MSD (real outputs computed without grad for FM loss)
                    msd_fake_outputs_g = msd(pred_wav)
                    with torch.no_grad():
                        msd_real_outputs_g = msd(target_wav)
                    loss_g_adv_msd = generator_adversarial_loss(msd_fake_outputs_g)
                    loss_fm_msd = feature_matching_loss(
                        msd_real_outputs_g, msd_fake_outputs_g
                    )

                    # Combined adversarial + feature matching
                    loss_g_adv = loss_g_adv_mpd + loss_g_adv_msd
                    loss_fm = loss_fm_mpd + loss_fm_msd

                # Multi-resolution mel loss (inworld-ai style, 7 scales)
                if args.lambda_multi_res_mel > 0:
                    loss_multi_res_mel = multi_res_mel_loss_fn(pred, target)
                else:
                    loss_multi_res_mel = pred.new_zeros(())

                # Global dB RMS loss (inworld-ai style)
                if args.lambda_global_rms > 0:
                    loss_global_rms = global_rms_loss(pred, target)
                else:
                    loss_global_rms = pred.new_zeros(())

                # Total generator loss
                loss_g = (
                    args.lambda_adv * loss_g_adv
                    + args.lambda_fm * loss_fm
                    + args.lambda_multi_res_mel * loss_multi_res_mel
                    + args.lambda_global_rms * loss_global_rms
                )

                optimizer_g.zero_grad()
                accelerator.backward(loss_g)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer_g.step()
                scheduler_g.step()

            # Count/log/eval/save only on real optimizer sync steps.
            if accelerator.sync_gradients:
                global_step += 1
                audio_sec = audio_lengths.sum().item() / target_sample_rate
                total_audio_sec += audio_sec

                # Logging
                if global_step % args.log_every == 0:
                    log_dict = {
                        "g/loss_total": loss_g.item(),
                        "g/loss_multi_res_mel": loss_multi_res_mel.item(),
                        "g/loss_global_rms": loss_global_rms.item(),
                        "train/lr/generator": scheduler_g.get_last_lr()[0],
                        "train/audio_sec": audio_sec,
                        "train/total_audio_hour": total_audio_sec / 3600.0,
                    }
                    if args.use_gan:
                        log_dict.update(
                            {
                                "d/loss_total": loss_d.item(),
                                "d/mpd_loss": loss_d_mpd.item(),
                                "d/msd_loss": loss_d_msd.item(),
                                "d_mpd/grad_norm": mpd_grad_norm,
                                "d_msd/grad_norm": msd_grad_norm,
                                "d_mpd/dr": dr_mpd.item(),
                                "d_mpd/dg": dg_mpd.item(),
                                "d_msd/dr": dr_msd.item(),
                                "d_msd/dg": dg_msd.item(),
                                "g/loss_adv": loss_g_adv.item(),
                                "g/loss_fm": loss_fm.item(),
                                "train/lr/discriminator": scheduler_d.get_last_lr()[0],
                            }
                        )
                    accelerator.log(log_dict, step=global_step)

                    progress_bar.set_postfix(
                        g=loss_g.item(),
                        mel=loss_multi_res_mel.item(),
                        **(
                            {"d": loss_d.item(), "adv": loss_g_adv.item()}
                            if args.use_gan
                            else {}
                        ),
                    )

                # Evaluation
                if (
                    val_dataloader
                    and global_step % args.eval_every == 0
                    and global_step > 0
                ):
                    val_losses = eval_step(
                        model, multi_res_mel_loss_fn, val_dataloader, accelerator
                    )
                    accelerator.print(
                        f"\nStep {global_step} - Validation: {val_losses}"
                    )
                    accelerator.log(val_losses, step=global_step)

                    if val_losses["val/loss_multi_res_mel"] < best_val_loss:
                        best_val_loss = val_losses["val/loss_multi_res_mel"]
                        save_checkpoint(
                            model,
                            mpd,
                            msd,
                            optimizer_g,
                            optimizer_d,
                            scheduler_g,
                            scheduler_d,
                            global_step,
                            epoch,
                            args,
                            accelerator,
                            num_frozen,
                            base_upsample_rates,
                            new_upsample_rates,
                            is_best=True,
                        )

                # Save checkpoint
                if global_step % args.save_every == 0 and global_step > 0:
                    save_checkpoint(
                        model,
                        mpd,
                        msd,
                        optimizer_g,
                        optimizer_d,
                        scheduler_g,
                        scheduler_d,
                        global_step,
                        epoch,
                        args,
                        accelerator,
                        num_frozen,
                        base_upsample_rates,
                        new_upsample_rates,
                    )

                if args.max_train_steps and global_step >= args.max_train_steps:
                    break

        # Save at end of epoch
        save_checkpoint(
            model,
            mpd,
            msd,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d,
            global_step,
            epoch,
            args,
            accelerator,
            num_frozen,
            base_upsample_rates,
            new_upsample_rates,
        )

        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    # Final checkpoint
    save_checkpoint(
        model,
        mpd,
        msd,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
        global_step,
        args.num_epochs,
        args,
        accelerator,
        num_frozen,
        base_upsample_rates,
        new_upsample_rates,
    )

    accelerator.end_training()
    accelerator.print("\nTraining completed!")


if __name__ == "__main__":
    main()

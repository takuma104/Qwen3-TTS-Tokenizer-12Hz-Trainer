# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Checkpoint Evaluation Script

Evaluates multiple training checkpoints on WebDataset audio,
comparing them across 4 metrics:
  - multi_res_mel: multi-resolution mel spectrogram loss (lower is better)
  - utmos: neural MOS prediction, 0-5 scale (higher is better)  [optional]
  - mcd: mel cepstral distortion (lower is better)
  - dg: discriminator score from a fixed reference discriminator (higher is better)

Usage:
    python evaluate_checkpoints.py \\
        --checkpoints ../output/run8/checkpoint-step-1000 ../output/run8/checkpoint-best \\
        --shard_pattern "../data/shards-{000000..000010}.tar" \\
        --num_samples 200 \\
        --output_dir ../eval_results/comparison
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from tqdm import tqdm
import pyworld
import pysptk

# Suppress MPS-backend STFT resize deprecation warning (PyTorch internal bug, harmless)
warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized",
    module="torch.functional",
)

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))
# Add xcodec2 dir to path (for xcodec2's internal bare imports)
sys.path.insert(0, str(Path(__file__).parent.parent / "xcodec2"))

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)
from xcodec2.criterions.mel_loss import MultiResolutionMelSpectrogramLoss
from xcodec2.module.mpd import HiFiGANMultiPeriodDiscriminator
from xcodec2.module.mstft import SpecDiscriminator

try:
    import utmosv2
    UTMOS_AVAILABLE = True
except Exception:
    UTMOS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _extract_mcep(wav: np.ndarray, sr: int, order: int = 24, frame_period: float = 5.0) -> np.ndarray:
    """Extract mel-cepstral coefficients using WORLD + pysptk."""
    wav64 = wav.astype(np.float64)
    f0, t = pyworld.dio(wav64, sr, frame_period=frame_period)
    f0 = pyworld.stonemask(wav64, f0, t, sr)
    sp = pyworld.cheaptrick(wav64, f0, t, sr)
    # pysptk.mc2b expects float64 log-power spectrum
    mcep = pysptk.sp2mc(sp, order=order, alpha=pysptk.util.mcepalpha(sr))
    return mcep  # shape: [n_frames, order+1]


def mcd_score(pred: np.ndarray, target: np.ndarray, sr: int, order: int = 24) -> float:
    """Mel Cepstral Distortion in dB (lower is better).

    Uses WORLD for spectral envelope extraction and pysptk for
    mel-cepstral analysis. Excludes c0 (energy). Typical values: 3-10 dB.
    """
    mcep_pred = _extract_mcep(pred, sr, order=order)
    mcep_tgt = _extract_mcep(target, sr, order=order)

    # Exclude c0, align frames
    min_frames = min(mcep_pred.shape[0], mcep_tgt.shape[0])
    if min_frames == 0:
        return float("nan")
    diff = mcep_pred[:min_frames, 1:] - mcep_tgt[:min_frames, 1:]
    return float((10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))


def create_discriminators() -> Tuple[HiFiGANMultiPeriodDiscriminator, "SpecDiscriminator"]:
    """Create MPD and MSD with the same params used during training."""
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
    return mpd, msd


def load_fixed_discriminator(
    checkpoint_path: str, device: torch.device
) -> Optional[Tuple[HiFiGANMultiPeriodDiscriminator, "SpecDiscriminator"]]:
    """Load discriminator weights from a checkpoint into fixed MPD+MSD."""
    disc_path = Path(checkpoint_path) / "discriminator.pt"
    if not disc_path.exists():
        print(f"  [WARN] discriminator.pt not found in {checkpoint_path}, skipping dg metric")
        return None

    mpd, msd = create_discriminators()
    disc_state = torch.load(disc_path, map_location="cpu", weights_only=True)
    mpd.load_state_dict(disc_state["mpd"])
    msd.load_state_dict(disc_state["msd"])
    mpd = mpd.to(device).eval()
    msd = msd.to(device).eval()
    print(f"  Fixed discriminator loaded from: {checkpoint_path}")
    return mpd, msd


@torch.no_grad()
def compute_dg(
    pred_wav: np.ndarray,
    mpd: HiFiGANMultiPeriodDiscriminator,
    msd: "SpecDiscriminator",
    device: torch.device,
) -> Tuple[float, float]:
    """Compute discriminator scores for a waveform (higher = more realistic).

    Returns (dg_mpd, dg_msd) separately.
    """
    wav_t = torch.from_numpy(pred_wav).float().unsqueeze(0).unsqueeze(0).to(device)

    def _mean_score(outputs) -> float:
        vals = []
        for out_list in outputs:
            dg = out_list[-1]  # final layer output
            vals.append(torch.mean(dg.float()).item())
        return float(np.mean(vals)) if vals else float("nan")

    return _mean_score(mpd(wav_t)), _mean_score(msd(wav_t))


def load_checkpoint_decoder(
    base_tokenizer: Qwen3TTSTokenizer,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3TTSTokenizerV2Decoder:
    """Build a decoder by patching the base model with checkpoint weights."""
    ckpt_dir = Path(checkpoint_path)
    with open(ckpt_dir / "config.json") as f:
        ckpt_config = json.load(f)

    new_upsample_rates = ckpt_config["new_upsample_rates"]

    base_decoder_config = base_tokenizer.model.decoder.config
    config_dict = base_decoder_config.to_dict()
    config_dict["upsample_rates"] = new_upsample_rates
    for key in ("model_type", "transformers_version"):
        config_dict.pop(key, None)

    new_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
    new_decoder = Qwen3TTSTokenizerV2Decoder(new_config)

    # Copy base weights (for shared layers)
    new_decoder.load_state_dict(base_tokenizer.model.decoder.state_dict(), strict=False)

    # Load checkpoint-specific weights
    ckpt_weights = load_file(str(ckpt_dir / "decoder_block.safetensors"))
    new_decoder.load_state_dict(ckpt_weights, strict=False)

    return new_decoder.to(device).to(dtype).eval()


def decode_with_decoder(
    decoder: Qwen3TTSTokenizerV2Decoder,
    audio_codes: np.ndarray,
    base_tokenizer: Qwen3TTSTokenizer,
    ckpt_config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, int]:
    """Decode audio_codes using a patched decoder, returns (waveform, sample_rate)."""
    extra_upsample_rate = ckpt_config.get("extra_upsample_rate", 2)
    output_sr = base_tokenizer.config.output_sample_rate * extra_upsample_rate

    codes_t = torch.from_numpy(audio_codes).long()
    if codes_t.dim() == 2:
        codes_t = codes_t.unsqueeze(0)  # [1, seq_len, 16]
    codes_t = codes_t.transpose(1, 2).to(device)  # [1, 16, seq_len]

    with torch.no_grad():
        wav = decoder(codes_t)  # [1, 1, T] or [1, T]

    wav = wav.squeeze().float().cpu().numpy()
    return wav, output_sr


def decode_with_base_tokenizer(
    audio_codes: np.ndarray,
    base_tokenizer: Qwen3TTSTokenizer,
    device: torch.device,
    target_sr: int = 48000,
) -> Tuple[np.ndarray, int]:
    """Decode audio_codes using the base tokenizer's original decoder, resampled to target_sr."""
    native_sr = base_tokenizer.config.output_sample_rate

    codes_t = torch.from_numpy(audio_codes).long()
    if codes_t.dim() == 2:
        codes_t = codes_t.unsqueeze(0)  # [1, seq_len, 16]
    codes_t = codes_t.transpose(1, 2).to(device)  # [1, 16, seq_len]

    with torch.no_grad():
        wav = base_tokenizer.model.decoder(codes_t)  # [1, 1, T] or [1, T]

    wav = wav.squeeze().float().cpu().numpy()

    # Resample to target_sr so metrics are comparable with 48kHz checkpoints
    if native_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=target_sr)

    return wav, target_sr


# ---------------------------------------------------------------------------
# WebDataset loading
# ---------------------------------------------------------------------------


def load_webdataset_samples(
    shard_pattern: str,
    target_sample_rate: int,
    num_samples: int,
    min_duration: float = 1.0,
    max_duration: float = 10.0,
    token_per_second: float = 12.5,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Load samples from WebDataset, returning (audio_codes, audio, sample_rate).

    Each sample contains pre-encoded audio_codes (.npy) and original audio
    (.wav/.mp3/.flac/.ogg/.opus). Audio is resampled to target_sample_rate.

    Returns list of (audio_codes [seq_len, 16], audio_array, target_sample_rate).
    """
    import glob as glob_mod
    import io

    import webdataset as wds

    # Expand glob pattern if applicable
    if "*" in shard_pattern and "{" not in shard_pattern:
        expanded_files = sorted(glob_mod.glob(shard_pattern))
        if not expanded_files:
            print(f"[ERROR] No files found matching pattern: {shard_pattern}")
            return []
        print(f"Found {len(expanded_files)} tar files")
        pattern = expanded_files
    else:
        pattern = shard_pattern

    print(f"\nLoading WebDataset: {shard_pattern}")

    dataset = (
        wds.WebDataset(pattern, shardshuffle=False)
        .decode("rgb")
    )

    samples = []
    skipped_dur = 0
    skipped_rms = 0
    skipped_audio = 0

    for sample in tqdm(dataset, desc="Loading samples", total=num_samples):
        if len(samples) >= num_samples:
            break

        # Load audio_codes
        audio_codes = sample.get("npy")
        if audio_codes is None:
            continue
        assert isinstance(audio_codes, np.ndarray), "audio_codes must be a numpy array"
        assert audio_codes.ndim == 1
        audio_codes = audio_codes.reshape(-1, 16)  # (seq_len, 16)

        # Duration filter via code length
        num_codes = audio_codes.shape[0]
        min_codes = int(token_per_second * min_duration)
        max_codes = int(token_per_second * max_duration)
        if num_codes < min_codes or num_codes > max_codes:
            skipped_dur += 1
            continue

        # Get audio data (support multiple formats)
        audio_data = None
        for ext in ["wav", "mp3", "flac", "ogg", "opus"]:
            if ext in sample:
                audio_data = sample[ext]
                break

        if audio_data is None:
            skipped_audio += 1
            continue

        # Load with librosa
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)

        # Resample to target sample rate
        if sr != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)

        # RMS filter
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-4:  # ~ -80 dB
            skipped_rms += 1
            continue

        samples.append((audio_codes, audio, target_sample_rate))

    print(
        f"  Collected {len(samples)} valid samples  "
        f"(skipped: duration={skipped_dur}, no_audio={skipped_audio}, rms={skipped_rms})"
    )
    return samples


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "multi_res_mel": "Multi-Res Mel Loss (lower=better)",
    "utmos": "UTMOSv2 Score (higher=better)",
    "mcd": "MCD [dB] - Mel Cepstral Distortion (lower=better)",
    "dg_mpd": "MPD Discriminator Score (higher=better)",
    "dg_msd": "MSD Discriminator Score (higher=better)",
}


def plot_histograms(
    results: Dict[str, Dict[str, List[float]]],
    checkpoint_names: List[str],
    output_dir: Path,
):
    """Save overlay histograms per metric."""
    metrics = list(next(iter(results.values())).keys())
    colors = plt.cm.tab10.colors

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))
        for idx, ckpt_name in enumerate(checkpoint_names):
            vals = [v for v in results[ckpt_name][metric] if not np.isnan(v)]
            if not vals:
                continue
            ax.hist(
                vals,
                bins=30,
                alpha=0.5,
                color=colors[idx % len(colors)],
                label=ckpt_name,
                density=True,
            )
        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution: {metric}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = output_dir / f"histogram_{metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_violin_box(
    results: Dict[str, Dict[str, List[float]]],
    checkpoint_names: List[str],
    output_dir: Path,
):
    """Save violin + box plot per metric."""
    metrics = list(next(iter(results.values())).keys())

    for metric in metrics:
        data_per_ckpt = []
        for ckpt_name in checkpoint_names:
            vals = [v for v in results[ckpt_name][metric] if not np.isnan(v)]
            data_per_ckpt.append(vals)

        if all(len(d) == 0 for d in data_per_ckpt):
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(checkpoint_names) * 2), 6))
        positions = list(range(1, len(checkpoint_names) + 1))

        # Violin
        vp = ax.violinplot(
            [d if d else [float("nan")] for d in data_per_ckpt],
            positions=positions,
            showmedians=False,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_alpha(0.4)

        # Box on top
        ax.boxplot(
            [d if d else [float("nan")] for d in data_per_ckpt],
            positions=positions,
            widths=0.3,
            patch_artist=False,
            medianprops=dict(color="red", linewidth=2),
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(checkpoint_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"Checkpoint Comparison: {metric}")
        fig.tight_layout()
        out_path = output_dir / f"violin_{metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints on WebDataset audio")
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="List of checkpoint directory paths"
    )
    parser.add_argument(
        "--shard_pattern",
        type=str,
        required=True,
        help='WebDataset shard pattern (e.g., "data/shards-{000000..000010}.tar" or "data/*.tar")',
    )
    parser.add_argument(
        "--num_samples", type=int, default=200, help="Number of audio samples to evaluate"
    )
    parser.add_argument(
        "--fixed_discriminator",
        type=str,
        default=None,
        help="Checkpoint to load discriminator from (default: last --checkpoints entry)",
    )
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Base 24kHz model path",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cuda, cpu, mps"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=48000,
        help="Target sample rate for output (default: 48000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  dtype: {dtype}")
    print(f"Output: {output_dir}")

    if not UTMOS_AVAILABLE:
        print(
            "\n[WARN] utmos not installed — UTMOS metric will be skipped.\n"
            "       To enable: pip install utmos\n"
        )

    # ---- Validate checkpoints ------------------------------------------------
    checkpoint_paths = [Path(p) for p in args.checkpoints]
    for p in checkpoint_paths:
        if not (p / "decoder_block.safetensors").exists():
            print(f"[ERROR] decoder_block.safetensors not found in: {p}")
            sys.exit(1)

    checkpoint_names = [str(p) for p in checkpoint_paths]
    print(f"\nCheckpoints to evaluate ({len(checkpoint_paths)}):")
    for name in checkpoint_names:
        print(f"  - {name}")

    # ---- Load base model (once) ----------------------------------------------
    print(f"\nLoading base model: {args.base_model_path}")
    base_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.base_model_path,
        dtype=dtype,
        device_map=str(device) if device.type != "cpu" else None,
    )
    base_tokenizer.model.eval()

    # ---- Load fixed discriminator (once) ------------------------------------
    disc_checkpoint = args.fixed_discriminator or str(checkpoint_paths[-1])
    disc_pair = load_fixed_discriminator(disc_checkpoint, device)
    if disc_pair is not None:
        # Keep discriminators in float32 — SpecDiscriminator runs STFT internally
        # which always produces float32, so the whole pipeline must stay float32.
        disc_pair[0].float()
        disc_pair[1].float()

    # ---- Load UTMOS (once) --------------------------------------------------
    utmos_predictor = None
    if UTMOS_AVAILABLE:
        print("\nLoading UTMOSv2 predictor...")
        utmos_predictor = utmosv2.create_model(pretrained=True, device=device)

    # ---- Mel loss function (once) -------------------------------------------
    mel_loss_fn = MultiResolutionMelSpectrogramLoss(
        sample_rate=args.target_sample_rate
    ).to(device)

    # ---- Load WebDataset samples (audio_codes + audio already available) ------
    wds_samples = load_webdataset_samples(
        shard_pattern=args.shard_pattern,
        target_sample_rate=args.target_sample_rate,
        num_samples=args.num_samples,
    )

    if len(wds_samples) == 0:
        print("[ERROR] No valid samples loaded from WebDataset.")
        sys.exit(1)

    # Build valid_pairs: list of (audio_codes_np, (audio_np, sample_rate))
    valid_pairs = [
        (codes, (audio, sr))
        for codes, audio, sr in wds_samples
    ]
    print(f"\nEvaluating on {len(valid_pairs)} samples")

    # ---- Helper: evaluate a decode function on valid_pairs --------------------
    def evaluate_decoder(
        name: str,
        decode_fn,
        valid_pairs: List,
    ) -> Dict[str, List[float]]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        metric_lists: Dict[str, List[float]] = {
            "multi_res_mel": [],
            "mcd": [],
            "dg_mpd": [],
            "dg_msd": [],
        }
        if utmos_predictor is not None:
            metric_lists["utmos"] = []

        for codes, (orig_array, orig_sr) in tqdm(valid_pairs, desc=name):
            try:
                pred_wav, pred_sr = decode_fn(codes)

                # Resample original to pred_sr for fair comparison
                if orig_sr != pred_sr:
                    target_ref = librosa.resample(orig_array, orig_sr=orig_sr, target_sr=pred_sr)
                else:
                    target_ref = orig_array.copy()

                # Align length
                min_len = min(len(pred_wav), len(target_ref))
                pred_trim = pred_wav[:min_len]
                ref_trim = target_ref[:min_len]

                # --- multi_res_mel ---
                pred_t = torch.from_numpy(pred_trim).float().unsqueeze(0).to(device)
                ref_t = torch.from_numpy(ref_trim).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    mel_val = mel_loss_fn(pred_t, ref_t).item()
                metric_lists["multi_res_mel"].append(mel_val)

                # --- MCD ---
                mcd_val = mcd_score(pred_trim, ref_trim, pred_sr)
                metric_lists["mcd"].append(mcd_val)

                # --- dg (fixed discriminator) ---
                if disc_pair is not None:
                    mpd_disc, msd_disc = disc_pair
                    dg_mpd_val, dg_msd_val = compute_dg(pred_trim, mpd_disc, msd_disc, device)
                    metric_lists["dg_mpd"].append(dg_mpd_val)
                    metric_lists["dg_msd"].append(dg_msd_val)
                else:
                    metric_lists["dg_mpd"].append(float("nan"))
                    metric_lists["dg_msd"].append(float("nan"))

                # --- UTMOS ---
                if utmos_predictor is not None:
                    mos = utmos_predictor.predict(data=pred_trim, sr=pred_sr, device=device, num_workers=0, verbose=False)
                    metric_lists["utmos"].append(float(mos.item() if hasattr(mos, 'item') else mos))

            except Exception as e:
                warnings.warn(f"Metric computation failed: {e}")
                for k in metric_lists:
                    metric_lists[k].append(float("nan"))

        # Per-entry summary
        print(f"\n  Summary for {name}:")
        for metric, vals in metric_lists.items():
            valid_vals = [v for v in vals if not np.isnan(v)]
            if valid_vals:
                print(
                    f"    {metric:>15s}:  mean={np.mean(valid_vals):.4f}  "
                    f"std={np.std(valid_vals):.4f}  "
                    f"median={np.median(valid_vals):.4f}"
                )

        return metric_lists

    # ---- Evaluate baseline (base tokenizer original decoder) ----------------
    results: Dict[str, Dict[str, List[float]]] = {}
    all_names: List[str] = []

    baseline_name = f"baseline (24kHz→{args.target_sample_rate // 1000}kHz resampled)"
    all_names.append(baseline_name)
    results[baseline_name] = evaluate_decoder(
        baseline_name,
        lambda codes: decode_with_base_tokenizer(codes, base_tokenizer, device, args.target_sample_rate),
        valid_pairs,
    )

    # ---- Evaluate each checkpoint -------------------------------------------
    for ckpt_path, ckpt_name in zip(checkpoint_paths, checkpoint_names):
        all_names.append(ckpt_name)

        with open(ckpt_path / "config.json") as f:
            ckpt_config = json.load(f)

        decoder = load_checkpoint_decoder(base_tokenizer, str(ckpt_path), device, dtype)

        results[ckpt_name] = evaluate_decoder(
            ckpt_name,
            lambda codes, _dec=decoder, _cfg=ckpt_config: decode_with_decoder(
                _dec, codes, base_tokenizer, _cfg, device, dtype
            ),
            valid_pairs,
        )

    # ---- Save summary CSV ---------------------------------------------------
    import csv

    csv_path = output_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        metrics_all = list(next(iter(results.values())).keys())
        fieldnames = ["checkpoint"] + [
            f"{m}_{stat}"
            for m in metrics_all
            for stat in ["mean", "std", "median", "p25", "p75", "n_valid"]
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ckpt_name in all_names:
            row = {"checkpoint": ckpt_name}
            for m in metrics_all:
                vals = [v for v in results[ckpt_name][m] if not np.isnan(v)]
                if vals:
                    row[f"{m}_mean"] = f"{np.mean(vals):.6f}"
                    row[f"{m}_std"] = f"{np.std(vals):.6f}"
                    row[f"{m}_median"] = f"{np.median(vals):.6f}"
                    row[f"{m}_p25"] = f"{np.percentile(vals, 25):.6f}"
                    row[f"{m}_p75"] = f"{np.percentile(vals, 75):.6f}"
                    row[f"{m}_n_valid"] = str(len(vals))
                else:
                    for stat in ["mean", "std", "median", "p25", "p75", "n_valid"]:
                        row[f"{m}_{stat}"] = "nan"
            writer.writerow(row)
    print(f"\nSaved: {csv_path}")

    # ---- Best checkpoint per metric -----------------------------------------
    print("\n" + "=" * 60)
    print("Best checkpoint per metric:")
    print("=" * 60)
    lower_is_better = {"multi_res_mel", "mcd"}
    for m in metrics_all:
        means = {}
        for ckpt_name in all_names:
            vals = [v for v in results[ckpt_name][m] if not np.isnan(v)]
            if vals:
                means[ckpt_name] = np.mean(vals)
        if not means:
            print(f"  {m:>15s}: no valid data")
            continue
        if m in lower_is_better:
            best = min(means, key=means.__getitem__)
        else:
            best = max(means, key=means.__getitem__)
        print(f"  {m:>15s}: {best}  ({means[best]:.4f})")

    # ---- Visualize ----------------------------------------------------------
    print("\nGenerating plots...")
    plot_histograms(results, all_names, output_dir)
    plot_violin_box(results, all_names, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

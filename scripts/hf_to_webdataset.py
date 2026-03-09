#!/usr/bin/env python3
"""
Convert HuggingFace dataset to WebDataset format for Qwen3TTS training.

Input:  HuggingFace dataset with audio field (voice['audio']['array'] / ['sampling_rate'])
Output: WebDataset tar shards under <output_dir>/train/ and <output_dir>/val/
        Each sample contains:
          - .flac  : 48kHz mono 16-bit FLAC audio, exactly --duration seconds
          - .npy   : flattened int32 codec codes (shape: seq_len * 16,)

Usage:
    python scripts/hf_to_webdataset.py <dataset> <output_dir> [options]

Example:
    python scripts/hf_to_webdataset.py \
        "my-org/my-voice-dataset" ./output \
        --duration 3.0 \
        --shard-size 1000 \
        --val-percent 5.0
"""

import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import webdataset as wds
from datasets import load_dataset
from loguru import logger
from numpy.typing import NDArray
from qwen_tts import Qwen3TTSTokenizer
from resampy import resample
from tqdm import tqdm

TOKENIZER_SR = 24000  # Qwen3TTS tokenizer input sample rate
OUTPUT_SR = 48000  # Output FLAC sample rate / assembly sample rate

VAD_SR = 16000  # SileroVAD internal sample rate
VAD_WINDOW = 512  # SileroVAD frame size (samples at VAD_SR)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_tokenizer(device: str) -> Qwen3TTSTokenizer:
    logger.info("Loading Qwen3TTS tokenizer (Qwen/Qwen3-TTS-Tokenizer-12Hz)...")
    return Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        dtype=torch.bfloat16,
        device_map=device,
    )


def load_vad_model():
    """Load SileroVAD model via torch.hub."""
    logger.info("Loading SileroVAD model...")
    model, _ = torch.hub.load(
        repo_or_dir="litagin02/silero-vad",
        model="silero_vad",
        onnx=True,
        trust_repo=True,
    )
    return model


# ---------------------------------------------------------------------------
# VAD utilities (logic adapted from FineSpeechJa/src/silerovad_split.py)
# ---------------------------------------------------------------------------


def _get_speech_probs(
    audio_tensor: torch.Tensor,
    vad_model,
    sampling_rate: int = VAD_SR,
    window_size: int = VAD_WINDOW,
) -> np.ndarray:
    vad_model.reset_states()
    probs = []
    for start in range(0, len(audio_tensor), window_size):
        chunk = audio_tensor[start : start + window_size]
        if len(chunk) == window_size:
            probs.append(vad_model(chunk, sampling_rate).item())
        else:
            probs.append(0.0)
    if probs:
        probs[-1] = 0.0  # ensure last frame is unvoiced
    return np.array(probs)


def _hyst(x: np.ndarray, th_lo: float, th_hi: float) -> np.ndarray:
    """Hysteresis thresholding (from silerovad_split.py)."""
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size:
        return np.zeros_like(x, dtype=bool)
    cnt = np.cumsum(lo_or_hi)
    return np.where(cnt, hi[ind[cnt - 1]], False)


def remove_silence_vad(
    audio: np.ndarray,
    sr: int,
    vad_model,
    silence_gap_s: float = 0.1,
    min_voiced_s: float = 1.0,
    pos_thresh: float = 0.5,
    neg_thresh: float = 0.35,
) -> np.ndarray:
    """Remove unvoiced regions using SileroVAD and rejoin voiced segments.

    Voiced segments are concatenated with `silence_gap_s` seconds of silence
    between them.  Segments shorter than `min_voiced_s` are discarded.
    If no voiced region is found, the original audio is returned.

    Args:
        audio:         Mono float32 audio at sample rate `sr`.
        sr:            Sample rate of `audio`.
        vad_model:     Loaded SileroVAD model.
        silence_gap_s:  Silence duration inserted between voiced segments.
        min_voiced_s:   Voiced segments shorter than this (seconds) are discarded.
        pos_thresh:     VAD activation threshold.
        neg_thresh:     VAD deactivation threshold.

    Returns:
        Processed mono float32 audio at the original `sr`.
    """
    # Pad 0.5 s on both sides so VAD can reliably detect edge activity
    pad_orig = int(sr * 0.5)
    audio_padded = np.pad(audio, (pad_orig, pad_orig), mode="constant")

    # Resample padded audio to VAD_SR
    audio_16k = (
        resample(audio_padded, sr, VAD_SR).astype(np.float32)
        if sr != VAD_SR
        else audio_padded.copy()
    )

    probs = _get_speech_probs(torch.from_numpy(audio_16k), vad_model)
    binary = _hyst(probs, neg_thresh, pos_thresh).astype(int)

    voiced_indices = np.where(binary == 1)[0]
    if len(voiced_indices) == 0:
        return audio  # nothing detected – return as-is

    # Find contiguous voiced segments (groups of adjacent frame indices)
    breaks = np.where(np.diff(voiced_indices) > 1)[0]
    seg_starts = np.concatenate([[voiced_indices[0]], voiced_indices[breaks + 1]])
    seg_ends = np.concatenate([voiced_indices[breaks], [voiced_indices[-1]]])

    # Convert frame indices → sample positions in the padded original-SR audio
    min_voiced_samples = int(min_voiced_s * sr)
    silence_pad = np.zeros(int(silence_gap_s * sr), dtype=np.float32)
    segments: list[np.ndarray] = []
    for f_start, f_end in zip(seg_starts, seg_ends):
        start_s = f_start * VAD_WINDOW / VAD_SR
        end_s = (f_end + 1) * VAD_WINDOW / VAD_SR
        s = max(0, int(start_s * sr))
        e = min(len(audio_padded), int(end_s * sr))
        if e - s >= min_voiced_samples:
            segments.append(audio_padded[s:e])

    if not segments:
        return audio

    # Concatenate segments with silence gaps
    result = segments[0]
    for seg in segments[1:]:
        result = np.concatenate([result, silence_pad, seg])

    return result


# ---------------------------------------------------------------------------
# Audio normalization
# ---------------------------------------------------------------------------


def normalize_audio(
    wav: NDArray[np.float32], sr: int
) -> tuple[NDArray[np.float32], float]:
    DEFAULT_BLOCK_SIZE: float = 0.400  # seconds, Style-Bert-VITS2 Setting
    NORMALIZED_LUFS: float = -23.0  # LUFS, Style-Bert-VITS2 Setting

    if len(wav) < sr * DEFAULT_BLOCK_SIZE:
        pad_length = int(sr * DEFAULT_BLOCK_SIZE) - len(wav)
        wav = np.pad(wav, (0, pad_length), mode="constant", constant_values=0.0)

    meter = pyln.Meter(sr, block_size=DEFAULT_BLOCK_SIZE)
    loudness = meter.integrated_loudness(wav)
    if loudness == float("-inf"):
        return wav, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normalized_wav = pyln.normalize.loudness(wav, loudness, NORMALIZED_LUFS).astype(
            np.float32
        )
        if np.max(np.abs(normalized_wav)) >= 1.0:
            normalized_wav = pyln.normalize.peak(normalized_wav, -1.0).astype(
                np.float32
            )
    return normalized_wav, loudness


# ---------------------------------------------------------------------------
# Audio preparation (VAD) — returns audio at OUTPUT_SR
# ---------------------------------------------------------------------------


def prepare_audio_vad(
    audio_array: np.ndarray,
    sr: int,
    vad_model,
    silence_gap_s: float,
    min_voiced_s: float,
) -> np.ndarray:
    """Convert to mono, normalize, remove silence via VAD, resample to OUTPUT_SR.

    Returns:
        audio_48k: float32 ndarray at OUTPUT_SR (48 kHz)
    Raises:
        ValueError: if no voiced region is detected after VAD.
    """
    # Mono
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=-1)
    audio = audio_array.astype(np.float32)

    # Normalize (loudness)
    audio, _ = normalize_audio(audio, sr)

    # Remove unvoiced regions at original SR
    audio = remove_silence_vad(
        audio, sr, vad_model, silence_gap_s=silence_gap_s, min_voiced_s=min_voiced_s
    )
    if len(audio) == 0:
        raise ValueError("No voiced region detected after VAD.")

    # Resample to OUTPUT_SR (48 kHz)
    audio_48k = resample(audio, sr, OUTPUT_SR) if sr != OUTPUT_SR else audio.copy()
    return audio_48k


# ---------------------------------------------------------------------------
# Duration assembler
# ---------------------------------------------------------------------------


class DurationAssembler:
    """Assembles a stream of variable-length audio pieces into fixed-duration chunks.

    Works entirely at OUTPUT_SR (48 kHz).  Leftovers from one piece are prepended
    (with a silence gap) to the next, so no audio is wasted.  The last incomplete
    chunk is zero-padded and returned by flush().
    """

    def __init__(self, duration_s: float, silence_gap_s: float):
        self.duration_samples = int(duration_s * OUTPUT_SR)
        self.gap = np.zeros(int(silence_gap_s * OUTPUT_SR), dtype=np.float32)
        self.tail: np.ndarray = np.array([], dtype=np.float32)

    def add(self, audio_48k: np.ndarray) -> list[np.ndarray]:
        """Append audio to the internal buffer and return any completed chunks."""
        if len(self.tail) > 0:
            combined = np.concatenate([self.tail, self.gap, audio_48k])
        else:
            combined = audio_48k

        chunks: list[np.ndarray] = []
        while len(combined) >= self.duration_samples:
            chunks.append(combined[: self.duration_samples].copy())
            combined = combined[self.duration_samples :]

        self.tail = combined
        return chunks

    def flush(self) -> list[np.ndarray]:
        """Zero-pad the remaining tail to duration_samples and return it."""
        if len(self.tail) == 0:
            return []
        padded = np.zeros(self.duration_samples, dtype=np.float32)
        padded[: len(self.tail)] = self.tail
        self.tail = np.array([], dtype=np.float32)
        return [padded]


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_flac(audio_48k: np.ndarray) -> bytes:
    """Encode float32 audio array to 16-bit FLAC bytes at OUTPUT_SR."""
    buf = io.BytesIO()
    audio_int16 = (np.clip(audio_48k, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(buf, audio_int16, OUTPUT_SR, format="flac", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def encode_npy(codes_np: np.ndarray) -> bytes:
    """Serialize numpy array to bytes."""
    buf = io.BytesIO()
    np.save(buf, codes_np)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Shard writer helper
# ---------------------------------------------------------------------------


class ShardWriter:
    """Writes WebDataset .tar shards to a directory, rolling over at shard_size."""

    def __init__(self, output_dir: Path, shard_size: int):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_idx = 0
        self.count = 0
        self._writer: wds.TarWriter | None = None
        self._open_shard()

    def _open_shard(self):
        if self._writer is not None:
            self._writer.close()
        path = str(self.output_dir / f"shard-{self.shard_idx:06d}.tar")
        self._writer = wds.TarWriter(path)
        logger.debug(f"Opened shard: {path}")

    def write(self, key: str, flac_bytes: bytes, npy_bytes: bytes):
        if self.count > 0 and self.count % self.shard_size == 0:
            self.shard_idx += 1
            self._open_shard()
        self._writer.write({"__key__": key, "flac": flac_bytes, "npy": npy_bytes})
        self.count += 1

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def num_shards(self) -> int:
        return self.shard_idx + 1 if self.count > 0 else 0


# ---------------------------------------------------------------------------
# Tokenizer batching
# ---------------------------------------------------------------------------


@torch.inference_mode()
def tokenize_batch(
    audio_list: list[np.ndarray],
    tokenizer: Qwen3TTSTokenizer,
) -> list[np.ndarray]:
    """Encode a batch of 24kHz float32 audio arrays.

    Returns a list of int32 numpy arrays, each flattened (seq_len * 16,).
    """
    encoded = tokenizer.encode(audios=audio_list, sr=TOKENIZER_SR)
    return [
        codes.cpu().numpy().flatten().astype(np.int32) for codes in encoded.audio_codes
    ]


# ---------------------------------------------------------------------------
# Write helper
# ---------------------------------------------------------------------------


def _tokenize_and_write(
    batch: list[tuple[np.ndarray, np.ndarray]],
    tokenizer: Qwen3TTSTokenizer,
    train_writer: ShardWriter,
    val_writer: ShardWriter,
    rng: np.random.Generator,
    val_fraction: float,
    global_key_counter: list[int],  # mutable counter passed as list
) -> None:
    """Tokenize a batch of (chunk_24k, chunk_48k) pairs and write to shards."""
    if not batch:
        return
    codes_list = tokenize_batch([c24 for c24, _ in batch], tokenizer)
    for (_, chunk_48k), codes_np in zip(batch, codes_list):
        flac_bytes = encode_flac(chunk_48k)
        npy_bytes = encode_npy(codes_np)
        key = f"{global_key_counter[0]:08d}"
        global_key_counter[0] += 1
        if rng.random() < val_fraction:
            val_writer.write(key, flac_bytes, npy_bytes)
        else:
            train_writer.write(key, flac_bytes, npy_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to WebDataset format for Qwen3TTS training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="HuggingFace dataset name (e.g. 'my-org/my-dataset') or local path",
    )
    parser.add_argument(
        "output_dir",
        help="Output root directory; train/ and val/ sub-dirs are created here",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        metavar="CONFIG",
        help="Dataset configuration / subset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of samples per WebDataset shard (tar file)",
    )
    parser.add_argument(
        "--val-percent",
        type=float,
        default=0.1,
        help="Percentage of samples assigned to the validation set",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration (seconds) of each output WebDataset sample",
    )
    parser.add_argument(
        "--stage-size",
        type=int,
        default=200,
        help=(
            "Number of chunks to accumulate before shuffling and writing half of them. "
            "Larger values spread same-source chunks further apart in the output shards"
        ),
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=100,
        help="Streaming shuffle buffer size passed to dataset.shuffle()",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and train/val splitting",
    )
    parser.add_argument(
        "--silence-gap",
        type=float,
        default=0.1,
        help="Silence duration (seconds) inserted between voiced segments after VAD, "
        "and between pieces when joining audio from different sources",
    )
    parser.add_argument(
        "--min-voiced",
        type=float,
        default=1.0,
        help="Voiced segments shorter than this (seconds) are discarded",
    )
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    tokenizer = load_tokenizer(device)
    vad_model = load_vad_model()

    # Prepare output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} (split={args.split})")
    load_kwargs: dict = {"split": args.split, "streaming": True}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    dataset = load_dataset(args.dataset, **load_kwargs)
    shuffled = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    train_writer = ShardWriter(train_dir, args.shard_size)
    val_writer = ShardWriter(val_dir, args.shard_size)

    rng = np.random.default_rng(args.seed)
    val_fraction = args.val_percent / 100.0

    assembler = DurationAssembler(args.duration, args.silence_gap)

    # staging: list of (chunk_24k, chunk_48k); shuffled periodically to spread
    # chunks from the same long audio source across output shards
    staging: list[tuple[np.ndarray, np.ndarray]] = []
    global_key = [0]  # mutable counter
    skipped = 0

    def flush_staging(force: bool = False) -> None:
        """Shuffle staging and write the first half (or all if force=True)."""
        if not staging:
            return
        rng.shuffle(staging)
        if force:
            cut = len(staging)
        else:
            cut = len(staging) // 2
        batch, staging[:] = staging[:cut], staging[cut:]
        _tokenize_and_write(
            batch, tokenizer, train_writer, val_writer, rng, val_fraction, global_key
        )
        logger.info(
            f"Wrote {cut} chunks | train={train_writer.count:,} val={val_writer.count:,}"
        )

    try:
        with tqdm(desc="Processing", unit="chunks") as pbar:
            for voice in shuffled:
                audio_array = voice["audio"]["array"]
                sr = voice["audio"]["sampling_rate"]

                try:
                    audio_48k = prepare_audio_vad(
                        audio_array, sr, vad_model, args.silence_gap, args.min_voiced
                    )
                except Exception as e:
                    logger.warning(f"Audio processing failed: {e}")
                    skipped += 1
                    continue

                # Assemble into fixed-duration chunks
                chunks = assembler.add(audio_48k)
                for chunk_48k in chunks:
                    chunk_24k = resample(chunk_48k, OUTPUT_SR, TOKENIZER_SR)
                    staging.append((chunk_24k, chunk_48k))
                    pbar.update(1)

                # Periodic shuffle + partial write to spread same-source chunks
                if len(staging) >= args.stage_size:
                    flush_staging(force=False)

            # Flush remaining audio in assembler tail
            for chunk_48k in assembler.flush():
                chunk_24k = resample(chunk_48k, OUTPUT_SR, TOKENIZER_SR)
                staging.append((chunk_24k, chunk_48k))
                pbar.update(1)

            # Final flush: write everything remaining
            flush_staging(force=True)

    finally:
        train_writer.close()
        val_writer.close()

    logger.info(
        f"Done!\n"
        f"  Train  : {train_writer.count:,} samples in {train_writer.num_shards} shard(s)\n"
        f"  Val    : {val_writer.count:,} samples in {val_writer.num_shards} shard(s)\n"
        f"  Skipped: {skipped:,} source items"
    )


if __name__ == "__main__":
    main()

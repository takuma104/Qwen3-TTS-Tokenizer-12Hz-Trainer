"""
Dataset for 48kHz Upsampler Training

Data format:
- audio_codes: Encoded audio codes (12Hz, 16 quantizers)
- audio: Original audio file path (48kHz or to be resampled)
"""

from typing import List

import librosa
import numpy as np
import torch


def collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for batching

    Pads audio of different lengths to align them
    """
    # Get maximum lengths
    max_codes = max(b["audio_codes"].shape[0] for b in batch)
    max_samples_48k = max(b["audio_48k"].shape[0] for b in batch)

    batch_size = len(batch)

    # Initialize batch tensors
    audio_codes = torch.zeros(batch_size, max_codes, 16, dtype=torch.long)
    audio_48k = torch.zeros(batch_size, max_samples_48k)
    code_lengths = torch.zeros(batch_size, dtype=torch.long)
    audio_48k_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, b in enumerate(batch):
        codes_len = b["audio_codes"].shape[0]
        samples_48k = b["audio_48k"].shape[0]

        audio_codes[i, :codes_len] = b["audio_codes"]
        audio_48k[i, :samples_48k] = b["audio_48k"]
        code_lengths[i] = codes_len
        audio_48k_lengths[i] = samples_48k

    return {
        "audio_codes": audio_codes,  # (batch, max_codes, 16)
        "audio_48k": audio_48k,  # (batch, max_samples_48k)
        "code_lengths": code_lengths,  # (batch,)
        "audio_48k_lengths": audio_48k_lengths,  # (batch,)
    }


def create_webdataset_loader(
    shard_pattern: str,
    target_sample_rate: int = 48000,
    max_audio_length: float = 10.0,
    min_audio_length: float = 1.0,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
    token_per_second: float = 12.5,
):
    """
    Create WebDataset format data loader

    Args:
        shard_pattern: tar file pattern (e.g., "output/shards-{000000..000010}.tar")
        target_sample_rate: Target sample rate
        max_audio_length: Maximum audio length (seconds)
        min_audio_length: Minimum audio length (seconds)
        batch_size: Batch size
        num_workers: Number of workers
        shuffle_buffer: Shuffle buffer size
        token_per_second: Tokens per second for length filtering
    Returns:
        DataLoader
    """
    import io
    import webdataset as wds
    from torch.utils.data import DataLoader

    def _process_sample(sample):
        """Process WebDataset sample"""
        # Convert audio_codes from numpy to tensor
        audio_codes = sample["npy"]  # numpy array of shape (seq_len, 16)
        assert isinstance(audio_codes, np.ndarray), "audio_codes must be a numpy array"
        assert audio_codes.ndim == 1
        audio_codes = audio_codes.reshape(-1, 16)  # (seq_len, 16)
        audio_codes = torch.from_numpy(audio_codes).long()
        num_codes = audio_codes.shape[0]

        # Minimum length check
        min_codes = int(token_per_second * min_audio_length)
        if num_codes < min_codes:
            return None

        # Get audio data (support multiple formats)
        audio_data = None
        for ext in ["wav", "mp3", "flac", "ogg", "opus"]:
            if ext in sample:
                audio_data = sample[ext]
                break

        if audio_data is None:
            return None

        # Load with librosa
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)

        # Crop to maximum length (if necessary)
        max_codes = int(token_per_second * max_audio_length)
        if num_codes > max_codes:
            # Select random start position
            start_code = torch.randint(0, num_codes - max_codes, (1,)).item()
            end_code = start_code + max_codes

            audio_codes = audio_codes[start_code:end_code]
            num_codes = max_codes

            # Crop audio to corresponding range
            samples_per_code = sr / token_per_second
            start_sample = int(start_code * samples_per_code)
            end_sample = int(end_code * samples_per_code)
            audio = audio[start_sample:end_sample]

        # Resample to target (48kHz)
        if sr != target_sample_rate:
            audio_48k = librosa.resample(
                audio, orig_sr=sr, target_sr=target_sample_rate
            )
        else:
            audio_48k = audio

        # Convert to tensor
        audio_48k = torch.from_numpy(audio_48k).float()

        # Skip very quiet audio
        rms = torch.sqrt(torch.mean(audio_48k**2, dim=-1) + 1e-8)
        rms_db = (20 * torch.log10(rms)).item()
        if rms_db < -40.0:
            return None

        return {
            "audio_codes": audio_codes,  # (seq_len, 16)
            "audio_48k": audio_48k,  # (samples_48k,)
        }

    # Build WebDataset
    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=1000)
        .shuffle(shuffle_buffer if shuffle_buffer > 0 else 0)
        .decode("rgb")  # Non-image data as-is
        .map(_process_sample)
        .select(lambda x: x is not None)  # Filter None
    )

    # Batching with WebLoader
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers).batched(
        batch_size, collation_fn=collate_fn
    )

    return loader


if __name__ == "__main__":
    # For testing
    import glob
    import sys

    if len(sys.argv) < 2:
        print("Usage: python upsampler_dataset.py <webdataset_pattern>")
        sys.exit(1)

    path = sys.argv[1]

    print("Testing WebDataset loader...")

    # Expand glob pattern if applicable
    if "*" in path and "{" not in path:
        expanded_files = sorted(glob.glob(path))
        if not expanded_files:
            print(f"Error: No files found matching pattern: {path}")
            sys.exit(1)
        print(f"Found {len(expanded_files)} tar files")
        shard_pattern = expanded_files
    else:
        shard_pattern = path

    loader = create_webdataset_loader(
        shard_pattern=shard_pattern,
        batch_size=8,
        num_workers=0,
    )
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  audio_codes: {batch['audio_codes'].shape}")
        print(f"  audio_48k: {batch['audio_48k'].shape}")
        if i >= 2:
            break

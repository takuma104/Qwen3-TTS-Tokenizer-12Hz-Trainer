# Qwen3-TTS-Tokenizer-12Hz Trainer

[🤗 Model Weight / Audio Sample on Hugging Face](https://huggingface.co/takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz)

A GAN-based fine-tuning framework for [Qwen/Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) decoder. The primary use case is extending the decoder with an extra upsample block to produce **48 kHz** output — but the same script can also be used to fine-tune the base 24 kHz decoder on custom speech data.

For a concrete example of what this trainer produces, see the model card for [takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz](https://huggingface.co/takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz), which was trained with this codebase.

## Features

- **48 kHz decoder extension** — appends one extra `DecoderBlock` (upsample rate ×2) to the existing decoder stack, bumping output from 24 kHz to 48 kHz (`--add_48k_decoder_block`, on by default)
- **GAN training** — MPD (Multi-Period Discriminator) + MSD (Multi-Scale/Spec Discriminator) from xcodec2 (`--use_gan`, on by default)
- **Composite loss** — adversarial + feature matching + multi-resolution Mel spectrogram (7 scales) + global RMS loss
- **Flexible freeze control** — freeze only the base decoder blocks and train just the new ones, or fine-tune the entire decoder end-to-end (`--num_decoder_block_frozen` / `--train_full_decoder`)
- **WebDataset input** — streams pre-encoded `(audio_codes, audio)` pairs from `.tar` shards, supporting large-scale datasets efficiently
- **`accelerate` + bf16 mixed precision** — multi-GPU ready via HuggingFace Accelerate
- **W&B logging** — training/validation metrics and audio samples logged to Weights & Biases
- **Checkpoint resume** — resume from any saved checkpoint with `--resume_from`

## How It Works 

In the HuggingFace ecosystem, model weights are always distributed with a `config.json` that defines the architecture; `transformers` dynamically instantiates the model graph from these parameters before loading the weights — and Qwen3-TTS fully adheres to this convention. Its 12 Hz codec decoder performs 1920× upsampling (12.5 Hz × 1920 = 24 kHz) governed by the `upsample_rates` list in `config.json`. This 48 kHz variant achieves double the output sample rate purely by appending `2` to that list (`[8, 5, 4, 3] → [8, 5, 4, 3, 2]`), letting `transformers` instantiate one additional `DecoderBlock` at load time — no code changes required.


## Repository Structure

```
.
├── src/
│   ├── trainer.py       # Main training script
│   ├── dataset.py       # WebDataset loader
│   ├── losses.py        # global_rms_loss, GAN loss helpers
│   ├── merge.py         # Merge trained weights into a deployable HF model
│   └── inference.py     # Quick decode/evaluation helper
├── datasets/
│   ├── train/           # Training shards (*.tar)
│   └── val/             # Validation shards (*.tar)
├── train.sh             # Training launch script
└── pyproject.toml
```

## Dataset Preparation

Use `scripts/hf_to_webdataset.py` to convert a HuggingFace dataset into the WebDataset format required for training.

```bash
uv run python scripts/hf_to_webdataset.py \
    "my-org/my-voice-dataset" datasets/ \
    --shard-size 1000 \
    --val-percent 5.0
```

This will create `datasets/train/*.tar` and `datasets/val/*.tar` shards.
Each audio clip is resampled to **48 kHz mono 16-bit FLAC**, and codec codes are extracted with the Qwen3-TTS-Tokenizer-12Hz model.

Key options:

| Option | Default | Description |
|---|---|---|
| `--shard-size` | `1000` | Number of samples per `.tar` shard |
| `--val-percent` | `0.1` | Percentage of samples for the validation set |
| `--max-duration` | `40.0` | Skip clips longer than this many seconds |
| `--batch-duration` | `160.0` | Seconds of audio per tokenizer batch (tune for GPU memory) |
| `--shuffle-buffer` | `100` | Streaming shuffle buffer size |
| `--dataset-config` | — | HuggingFace dataset subset/config name |
| `--split` | `train` | Dataset split to load |

Run `uv run python scripts/hf_to_webdataset.py --help` for the full list.

## Dataset Format

Training data must be in **WebDataset** (`.tar`) format. Each sample in a shard must contain:

| Key | Type | Description |
|---|---|---|
| `audio_codes.npy` | `int64 (T, 16)` | Pre-encoded audio codes at 12 Hz, 16 quantizers |
| `audio.wav` / `audio.flac` | float32 PCM | Original audio at the target sample rate (48 kHz recommended) |

Place shards under `datasets/train/` and `datasets/val/`, or adjust the paths in `train.sh`.

## Installation

**Requirements:** Python 3.12+, CUDA-capable GPU, [uv](https://docs.astral.sh/uv/)

### 1. Clone (with submodules)

```bash
git clone --recursive https://github.com/takuma104/Qwen3-TTS-Tokenizer-12Hz-Trainer.git
cd Qwen3-TTS-Tokenizer-12Hz-Trainer
```

> The `--recursive` flag is required to pull in `xcodec2`, which provides the GAN discriminators used during training.

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Log in to Weights & Biases

```bash
uv run wandb login
```

### 5. Configure Accelerate

```bash
uv run accelerate config
```

Select your GPU setup (single GPU, multi-GPU, bf16 mixed precision, etc.) when prompted.

## Training

Edit the hyperparameters in `train.sh` if needed, then run:

```bash
bash train.sh
```

Key parameters in `train.sh`:

| Parameter | Default | Description |
|---|---|---|
| `--batch_size` | `4` | Per-device batch size |
| `--gradient_accumulation_steps` | `8` | Effective batch = batch_size × accum steps × num_GPUs |
| `--lr_g` | `1e-5` | Generator (decoder) learning rate |
| `--lr_d` | `1e-5` | Discriminator learning rate |
| `--max_train_steps` | `500000` | Total training steps |
| `--num_decoder_block_frozen` | `0` | Freeze this many base decoder blocks (0 = train all) |
| `--lambda_adv` | `1.0` | Adversarial loss weight |
| `--lambda_fm` | `1.0` | Feature matching loss weight |
| `--lambda_multi_res_mel` | `15.0` | Multi-resolution Mel loss weight |
| `--lambda_global_rms` | `1.0` | Global RMS loss weight |
| `--mixed_precision` | `bf16` | Mixed precision mode |

To fine-tune only the base 24 kHz decoder (no extra upsample block, no GAN):

```bash
uv run accelerate launch src/trainer.py \
    --train_shards "datasets/train/*.tar" \
    --no-add_48k_decoder_block \
    --no-use_gan \
    --train_full_decoder \
    --output_dir output/run1
```

### Resuming from a checkpoint

```bash
uv run accelerate launch src/trainer.py \
    --train_shards "datasets/train/*.tar" \
    --resume_from output/run1/checkpoint-5000 \
    --output_dir output/run1
```

## Merging Weights

After training, use `src/merge.py` to merge the trained decoder weights with the base model into a standard HuggingFace model that can be loaded without `trust_remote_code`:

```bash
uv run python src/merge.py \
    --base_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --checkpoint output/run1/checkpoint-best \
    --output_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz
```

The resulting model uses `model_type: qwen3_tts_tokenizer_12hz` and is fully compatible with `AutoModel.from_pretrained()`.

## Acknowledgements

- **[xcodec2](https://github.com/zhenye234/xcodec2)** — The GAN discriminator modules (`HiFiGANMultiPeriodDiscriminator`, `SpecDiscriminator`) and multi-resolution Mel loss are used directly from this repository.
- **[inworld-ai/tts](https://github.com/inworld-ai/tts)** — The `global_rms_loss` implementation and MPD/MSD hyperparameters are adopted from this work.

## License

Apache 2.0

# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Loss Functions
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

"""
Global RMS Energy Loss in dB

Computes per-track global RMS, converts to dB, and takes MSE of the
dB difference between predicted and target. This captures overall loudness
matching rather than frame-level energy.
Ported from inworld-ai/tts decoder.py (compute_generator_loss).
"""


def global_rms_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred: Predicted waveform (batch, samples)
        target: Target waveform (batch, samples)

    Returns:
        loss: MSE of the per-track dB RMS difference
    """
    # Add eps before sqrt to avoid NaN gradients when input is exactly zero
    # (sqrt(0) has inf gradient, leading to 0/0=NaN in backprop).
    # sqrt(1e-8) = 1e-4 -> 20*log10(1e-4) = -80 dB, consistent with clamp floor.
    pred_rms = torch.sqrt(torch.mean(pred**2, dim=-1) + 1e-8)
    target_rms = torch.sqrt(torch.mean(target**2, dim=-1) + 1e-8)
    pred_rms_db = (20 * torch.log10(pred_rms)).clamp(min=-80.0)
    target_rms_db = (20 * torch.log10(target_rms)).clamp(min=-80.0)
    return torch.mean((pred_rms_db - target_rms_db) ** 2)


def generator_adversarial_loss(disc_outputs: List[List[torch.Tensor]]) -> torch.Tensor:
    """LSGAN generator loss.

    Args:
        disc_outputs: List of discriminator outputs for fake samples.

    Returns:
        Generator adversarial loss.
    """
    loss = torch.tensor(0.0, device=disc_outputs[0][0].device)
    for dg_list in disc_outputs:
        dg = dg_list[-1]  # Final output layer for fake samples
        loss = loss + F.mse_loss(dg, torch.ones_like(dg))
    return loss


def discriminator_loss(
    disc_real_outputs: List[List[torch.Tensor]],
    disc_fake_outputs: List[List[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LSGAN discriminator loss.

    Args:
        disc_real_outputs: List of discriminator outputs for real samples.
        disc_fake_outputs: List of discriminator outputs for fake samples.

    Returns:
        total_loss: sum of r_loss + g_loss over all discriminators.
    """
    device = disc_real_outputs[0][0].device
    loss = torch.tensor(0.0, device=device)
    dr_total = torch.tensor(0.0, device=device)
    dg_total = torch.tensor(0.0, device=device)
    n = len(disc_real_outputs)
    for dr_list, dg_list in zip(disc_real_outputs, disc_fake_outputs):
        dr = dr_list[-1]  # Final output layer for real samples
        dg = dg_list[-1]  # Final output layer for fake samples
        r_loss = F.mse_loss(dr, torch.ones_like(dr))
        g_loss = F.mse_loss(dg, torch.zeros_like(dg))
        loss = loss + r_loss + g_loss
        dr_total = dr_total + torch.mean(dr.float())
        dg_total = dg_total + torch.mean(dg.float())
    return loss, dr_total / n, dg_total / n


def d_r1_loss(
    disc_real_outputs: List[List[torch.Tensor]],
    real_input: torch.Tensor,
) -> torch.Tensor:
    """R1 gradient penalty for discriminator regularization.

    Penalizes the L2 norm of the discriminator's output gradient w.r.t. the
    real input waveform. Applied lazily every d_reg_every steps (StyleGAN2-style).

    Args:
        disc_real_outputs: Return value of mpd(real_input) or msd(real_input).
                           List[List[Tensor]] — outer: sub-discriminators,
                           inner: feature maps, last element is final output.
        real_input: Waveform tensor passed to the discriminator,
                    must have requires_grad=True.

    Returns:
        Scalar R1 penalty.
    """
    total_real = sum(sub_out[-1].sum() for sub_out in disc_real_outputs)
    (grad_real,) = torch.autograd.grad(
        outputs=total_real, inputs=real_input, create_graph=True
    )
    # grad_real: (B, 1, T)
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()


def gpu_mcd(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    order: int = 24,
) -> torch.Tensor:
    """GPU-approximate Mel Cepstral Distortion in dB (lower is better).

    Replaces WORLD + pysptk with STFT → mel filterbank → DCT-II.
    Values correlate with but are not identical to the CPU WORLD-based MCD.

    Args:
        pred: Predicted waveform [B, T]
        target: Target waveform [B, T]
        sample_rate: Audio sample rate in Hz
        order: Cepstral order (default 24, matching CPU implementation)

    Returns:
        Scalar MCD value in dB.
    """
    hop_length = sample_rate // 200  # 5ms frames
    win_length = sample_rate // 40   # 25ms window
    n_fft = 2048 if sample_rate >= 32000 else 1024
    n_mels = order + 1  # 25 mel bands → c0..c24

    window = torch.hann_window(win_length, device=pred.device, dtype=pred.dtype)

    def _mcep(x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, n_fft, hop_length, win_length, window, return_complex=True
        )
        power = spec.abs().pow(2)  # [B, n_fft//2+1, T_frames]

        mel_fb = AF.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
        ).to(device=pred.device, dtype=pred.dtype)  # [n_fft//2+1, n_mels]

        mel_power = power.transpose(-1, -2) @ mel_fb  # [B, T_frames, n_mels]
        log_mel = torch.log(mel_power.clamp(min=1e-10))

        # DCT-II matrix
        n = torch.arange(n_mels, device=pred.device, dtype=pred.dtype)
        k = torch.arange(n_mels, device=pred.device, dtype=pred.dtype)
        dct_mat = torch.cos(math.pi * k.unsqueeze(1) * (n.unsqueeze(0) + 0.5) / n_mels)
        # [n_mels, n_mels]

        return log_mel @ dct_mat.T  # [B, T_frames, n_mels]

    mcep_pred = _mcep(pred)
    mcep_target = _mcep(target)

    min_frames = min(mcep_pred.shape[1], mcep_target.shape[1])
    if min_frames == 0:
        return pred.new_tensor(float("nan"))

    # Exclude c0 (index 0)
    diff = mcep_pred[:, :min_frames, 1:] - mcep_target[:, :min_frames, 1:]
    frame_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-16)
    mcd = (10.0 * math.sqrt(2.0) / math.log(10.0)) * frame_dist.mean()
    return mcd


def feature_matching_loss(
    disc_real_outputs: List[List[torch.Tensor]],
    disc_fake_outputs: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Feature matching loss between real and fake feature maps.

    Args:
        disc_real_outputs: List of feature map lists from discriminator on real samples.
        disc_fake_outputs: List of feature map lists from discriminator on fake samples.

    Returns:
        Feature matching loss.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0][0].device)
    for dr_list, dg_list in zip(disc_real_outputs, disc_fake_outputs):
        for dr, dg in zip(dr_list[:-1], dg_list[:-1]):  # Exclude final output layer
            loss = loss + F.l1_loss(dg, dr.detach())
    return loss

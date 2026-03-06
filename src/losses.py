"""
Loss Functions
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F

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
) -> torch.Tensor:
    """LSGAN discriminator loss.

    Args:
        disc_real_outputs: List of discriminator outputs for real samples.
        disc_fake_outputs: List of discriminator outputs for fake samples.

    Returns:
        total_loss: sum of r_loss + g_loss over all discriminators.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0][0].device)
    for dr_list, dg_list in zip(disc_real_outputs, disc_fake_outputs):
        dr = dr_list[-1]  # Final output layer for real samples
        dg = dg_list[-1]  # Final output layer for fake samples
        r_loss = F.mse_loss(dr, torch.ones_like(dr))
        g_loss = F.mse_loss(dg, torch.zeros_like(dg))
        loss = loss + r_loss + g_loss
    return loss


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

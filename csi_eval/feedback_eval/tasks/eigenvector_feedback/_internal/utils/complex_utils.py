"""Complex tensor helpers for real/imag CSI representations.

This file is a copy of ``utils/complex_utils.py`` from the original
CSIFeedback-Evaluation-1 project, vendored into csi_feedback_eval
to make the package self-contained.
"""

from __future__ import annotations

import numpy as np
import torch


def complex_to_ri_np(z: np.ndarray, channel_first: bool = True) -> np.ndarray:
    """Complex array -> real/imag representation.

    If z is [N, K, Nt], returns [N, 2, K, Nt] when channel_first=True.
    """
    if channel_first:
        return np.stack([z.real, z.imag], axis=1).astype(np.float32)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def ri_to_complex_np(x: np.ndarray, channel_first: bool = True) -> np.ndarray:
    if channel_first:
        return x[:, 0, ...] + 1j * x[:, 1, ...]
    return x[..., 0] + 1j * x[..., 1]


def ri_to_complex_torch(x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    real = x.select(channel_dim, 0)
    imag = x.select(channel_dim, 1)
    return torch.complex(real, imag)


def complex_to_ri_torch(z: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=channel_dim)


def normalize_complex_np(z: np.ndarray, axis=-1, eps: float = 1e-12) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(z) ** 2, axis=axis, keepdims=True) + eps)
    return z / norm


def normalize_complex_torch(z: torch.Tensor, dim=-1, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(torch.abs(z) ** 2, dim=dim, keepdim=True) + eps)
    return z / norm

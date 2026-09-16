"""Low-level math helpers for the metric layer.

These are intentionally lightweight and only depend on numpy/torch, so
they can be unit-tested and reused by any category.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _to_complex_ri(x: torch.Tensor) -> torch.Tensor:
    """[B, 2, ...] real/imag -> [B, ...] complex."""
    if x.ndim < 3 or x.shape[1] != 2:
        raise ValueError(f"Expected [B,2,...], got {tuple(x.shape)}")
    return torch.complex(x[:, 0], x[:, 1])


def nmse_db_ri(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """NMSE in dB for a real/imag tensor of shape [B, 2, ...]."""
    err = torch.sum((y_true - y_pred) ** 2, dim=tuple(range(1, y_true.ndim)))
    pwr = torch.sum(y_true ** 2, dim=tuple(range(1, y_true.ndim))).clamp_min(eps)
    nmse = err / pwr
    nmse_db = 10.0 * torch.log10(nmse.clamp_min(eps))
    return {
        "mean": float(nmse_db.mean().item()),
        "std": float(nmse_db.std(unbiased=False).item()),
        "min": float(nmse_db.min().item()),
        "max": float(nmse_db.max().item()),
        "linear_mean": float(nmse.mean().item()),
    }


def mse_ri(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """Plain MSE per-sample."""
    se = ((y_true - y_pred) ** 2).flatten(1).mean(dim=1)
    return {
        "mean": float(se.mean().item()),
        "std": float(se.std(unbiased=False).item()),
        "min": float(se.min().item()),
        "max": float(se.max().item()),
    }


def correlation_ri(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Non-squared complex cosine similarity over the last dim."""
    wt = _to_complex_ri(y_true)
    wp = _to_complex_ri(y_pred)
    inner = torch.sum(torch.conj(wt) * wp, dim=-1)
    denom = torch.sqrt(
        torch.sum(torch.abs(wt) ** 2, dim=-1) * torch.sum(torch.abs(wp) ** 2, dim=-1)
    ).clamp_min(eps)
    val = (torch.abs(inner) / denom).clamp(0.0, 1.0)
    return {
        "mean": float(val.mean().item()),
        "std": float(val.std(unbiased=False).item()),
        "min": float(val.min().item()),
        "max": float(val.max().item()),
    }


def sgcs_ri(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Per-subband squared generalized cosine similarity (auto layout)."""
    if y_true.ndim == 4 and y_true.shape[1] == 2:
        if y_true.shape[2] > y_true.shape[3]:
            wc = torch.complex(y_true[:, 0], y_true[:, 1])
            wp = torch.complex(y_pred[:, 0], y_pred[:, 1])
        else:
            wc = torch.complex(y_true[:, :, 0], y_true[:, :, 1])
            wp = torch.complex(y_pred[:, :, 0], y_pred[:, :, 1])
    else:
        raise ValueError(f"Expected [B,2,Nt,K] or [B,2,K,Nt], got {tuple(y_true.shape)}")

    inner = torch.sum(torch.conj(wc) * wp, dim=1)
    n_true = torch.sum(torch.abs(wc) ** 2, dim=1)
    n_pred = torch.sum(torch.abs(wp) ** 2, dim=1)
    rho2 = (torch.abs(inner) ** 2 / (n_true * n_pred + eps)).clamp(0.0, 1.0)
    return {
        "mean": float(rho2.mean().item()),
        "std": float(rho2.std(unbiased=False).item()),
        "min": float(rho2.min().item()),
        "max": float(rho2.max().item()),
    }


def evm_ri(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Error Vector Magnitude: sqrt(MSE) / sqrt(signal power).

    Defined as EVM (%) = 100 * sqrt( mean(|y - y_hat|^2) / mean(|y|^2) ).
    Useful for complex-valued CSI reconstruction; lower is better.
    """
    diff = y_true - y_pred
    err = (diff ** 2).flatten(1).mean(dim=1).clamp_min(0.0)
    pwr = (y_true ** 2).flatten(1).mean(dim=1).clamp_min(eps)
    ratio = (err / pwr).clamp_min(0.0)
    evm_pct = 100.0 * torch.sqrt(ratio)
    return {
        "mean_pct": float(evm_pct.mean().item()),
        "std_pct": float(evm_pct.std(unbiased=False).item()),
        "min_pct": float(evm_pct.min().item()),
        "max_pct": float(evm_pct.max().item()),
        "linear_mean": float(ratio.mean().item()),
    }

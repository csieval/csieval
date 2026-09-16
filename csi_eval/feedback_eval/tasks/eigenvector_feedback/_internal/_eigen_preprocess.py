"""Eigenvector preprocessing for WAIR-D CSI tensors.

This file is a copy of ``dataset/eigen_preprocess.py`` from the original
CSIFeedback-Evaluation-1 project, vendored into csi_feedback_eval
to make the package self-contained. The only changes are:
  * ``from utils.complex_utils import ...`` is rewritten to a relative
    import ``from .utils.complex_utils import ...`` so it can live under
    the package's ``_internal`` namespace.
  * Docstring header updated to note vendoring.

Input convention: complex CSI H has shape [N, Nt, Nr, Nf] or [Nt, Nr, Nf].
For each subband k, H_k is converted to [Nr, Nt], then the dominant right
singular vector is obtained from H_k^H H_k.

GPU acceleration: when a CUDA device is available, computation is offloaded to
GPU which provides ~10-100x speedup for large datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from csi_eval.progress import progress

from .utils.complex_utils import complex_to_ri_np, normalize_complex_np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EigenConfig:
    nt: int
    nr: int
    nf: int
    subband_size: int
    phase_fix: bool = True
    normalize_input_power: bool = False

    @property
    def n_subbands(self) -> int:
        if self.nf % self.subband_size != 0:
            raise ValueError(f"nf={self.nf} must be divisible by subband_size={self.subband_size}")
        return self.nf // self.subband_size


# ---------------------------------------------------------------------------
# NumPy reference implementation (CPU fallback)
# ---------------------------------------------------------------------------

def fix_eigenvector_phase(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Remove arbitrary eigenvector phase for reproducible MSE training.

    The entry with the largest magnitude is rotated to the positive real axis.
    Shape: [..., Nt].
    """
    idx = np.argmax(np.abs(w), axis=-1)
    flat = w.reshape(-1, w.shape[-1])
    idx_flat = idx.reshape(-1)
    phase = np.angle(flat[np.arange(flat.shape[0]), idx_flat] + eps)
    flat = flat * np.exp(-1j * phase)[:, None]
    return flat.reshape(w.shape)


def compute_subband_eigenvectors(
    H: np.ndarray,
    subband_size: int,
    phase_fix: bool = True,
    normalize_input_power: bool = False,
    batch_size: int = 64,
    subband_aggregation: str = "stack",
) -> np.ndarray:
    """Compute dominant eigenvectors for all samples and subbands.

    Per EVCsiNet (Liu et al., 2021) and 3GPP TypeI codebook convention, each
    subband contains ``subband_size`` RBs. The spatial covariance of the
    subband channel is:

        R_k = H_k^H H_k,    H_k ∈ C^{(L·Nr) × Nt}

    where L is the number of RBs in the subband. This function offers two
    ways of forming H_k from the (subband_size, Nr, Nt) block:

    - "stack"  (default, 3GPP-conformant): concatenate the subband_size RBs
              along the Rx axis, so H_k is L·Nr × Nt and uses all frequency
              samples within the subband. Recommended for EVCsiNet, where the
              subband is the basic feedback granularity.
    - "average": mean-pool the subband_size RBs down to a single Nr × Nt
              matrix (assumes the channel is flat within the subband).

    Args:
        H: complex ndarray, [N, Nt, Nr, Nf] or [Nt, Nr, Nf].
        subband_size: number of sampled carriers/RBs per feedback subband.
        phase_fix: make eigenvector phase deterministic.
        normalize_input_power: normalize each raw CSI sample before extracting eigenvectors.
        batch_size: samples processed in one eigendecomposition chunk.
        subband_aggregation: "stack" (default) or "average".

    Returns:
        W: complex64 ndarray, [N, K, Nt], K=Nf/subband_size.
    """
    if subband_aggregation not in ("stack", "average"):
        raise ValueError(
            f"subband_aggregation must be 'stack' or 'average', got {subband_aggregation!r}"
        )
    if H.ndim == 3:
        H = H[None, ...]
    if H.ndim != 4:
        raise ValueError(f"Expected H shape [N,Nt,Nr,Nf] or [Nt,Nr,Nf], got {H.shape}")
    if not np.iscomplexobj(H):
        H = H.astype(np.complex64)

    N, Nt, Nr, Nf = H.shape
    if Nf % subband_size != 0:
        raise ValueError(f"Nf={Nf} must be divisible by subband_size={subband_size}")
    K = Nf // subband_size
    W = np.empty((N, K, Nt), dtype=np.complex64)

    batch_starts = range(0, N, batch_size)
    for start in progress(
        batch_starts, total=(N + batch_size - 1) // batch_size,
        desc="feedback preprocess: eigenvectors CPU", unit="batch", leave=False
    ):
        end = min(start + batch_size, N)
        Hb = H[start:end].astype(np.complex64, copy=False)
        if normalize_input_power:
            p = np.sqrt(np.mean(np.abs(Hb) ** 2, axis=(1, 2, 3), keepdims=True) + 1e-12)
            Hb = Hb / p

        for k in range(K):
            f0 = k * subband_size
            f1 = f0 + subband_size
            Hsb = Hb[:, :, :, f0:f1]                              # [B, Nt, Nr, L]
            if subband_aggregation == "stack":
                # [B, Nt, L*Nr] → transpose to [B, L*Nr, Nt]
                Hsb = Hsb.reshape(Hb.shape[0], Nt, Nr * subband_size)
                Hmat = np.transpose(Hsb, (0, 2, 1))
            else:  # "average"
                Hsb = np.mean(Hsb, axis=-1)                      # [B, Nt, Nr]
                Hmat = np.transpose(Hsb, (0, 2, 1))              # [B, Nr, Nt]
            R = np.einsum("brt,bru->btu", np.conj(Hmat), Hmat, optimize=True)
            eigvals, eigvecs = np.linalg.eigh(R)
            wk = eigvecs[:, :, -1]
            wk = normalize_complex_np(wk, axis=-1)
            if phase_fix:
                wk = fix_eigenvector_phase(wk)
            W[start:end, k, :] = wk.astype(np.complex64)
    return W


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

def _torch_fix_eigenvector_phase(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Remove arbitrary eigenvector phase on GPU.

    The entry with the largest magnitude is rotated to the positive real axis.
    w shape: [B, Nt]
    """
    abs_w = torch.abs(w)
    idx = torch.argmax(abs_w, dim=-1)               # [B]
    gather = w[torch.arange(w.shape[0], device=w.device), idx]  # [B]
    phase = torch.angle(gather + eps)               # [B]
    w = w * torch.exp(-1j * phase.unsqueeze(-1))     # [B, Nt]
    return w


def _eigh_fallback_svd(Hmat: torch.Tensor) -> torch.Tensor:
    """SVD-based dominant eigenvector fallback for ill-conditioned batches.

    Hmat: [B, Nr, Nt], returns eigenvectors of H^H H [B, Nt, Nt].
    Uses torch.linalg.svd which is more numerically stable than eigh for
    near-singular or ill-conditioned matrices.
    """
    _, _, Vh = torch.linalg.svd(Hmat, full_matrices=False)
    return Vh.conj().transpose(-1, -2)


def compute_subband_eigenvectors_gpu(
    H: np.ndarray,
    subband_size: int,
    phase_fix: bool = True,
    normalize_input_power: bool = False,
    batch_size: int = 1024,
    device: Optional[str] = None,
    subband_aggregation: str = "stack",
) -> np.ndarray:
    """GPU-accelerated dominant eigenvector computation.

    Processes all K subbands in a single kernel launch per batch by padding
    subband dimension to a fixed width and using large-batch matmul.

    Args:
        H: complex ndarray, [N, Nt, Nr, Nf] or [Nt, Nr, Nf].
        subband_size: number of sampled carriers/RBs per feedback subband.
        phase_fix: make eigenvector phase deterministic.
        normalize_input_power: normalize each raw CSI sample before extracting eigenvectors.
        batch_size: samples per GPU batch. Default 1024 (GPU memory friendly).
        device: CUDA device string (e.g. "cuda:0"). If None, auto-detect.
        subband_aggregation: "stack" (default, 3GPP-conformant) or "average".

    Returns:
        W: complex64 ndarray, [N, K, Nt], K=Nf/subband_size.
    """
    if subband_aggregation not in ("stack", "average"):
        raise ValueError(
            f"subband_aggregation must be 'stack' or 'average', got {subband_aggregation!r}"
        )
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Use compute_subband_eigenvectors() for CPU.")
        device = "cuda:0"
    elif not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}' but CUDA is not available.")

    device_obj = torch.device(device)

    if H.ndim == 3:
        H = H[None, ...]
    if H.ndim != 4:
        raise ValueError(f"Expected H shape [N,Nt,Nr,Nf] or [Nt,Nr,Nf], got {H.shape}")
    if not np.iscomplexobj(H):
        H = H.astype(np.complex64)

    N, Nt, Nr, Nf = H.shape
    if Nf % subband_size != 0:
        raise ValueError(f"Nf={Nf} must be divisible by subband_size={subband_size}")
    K = Nf // subband_size

    W = np.empty((N, K, Nt), dtype=np.complex64)

    batch_starts = range(0, N, batch_size)
    for start in progress(
        batch_starts, total=(N + batch_size - 1) // batch_size,
        desc="feedback preprocess: eigenvectors GPU", unit="batch", leave=False
    ):
        end = min(start + batch_size, N)
        B = end - start
        Hb = torch.from_numpy(H[start:end]).to(device_obj)

        if normalize_input_power:
            p = torch.sqrt(torch.mean(torch.abs(Hb) ** 2, dim=(1, 2, 3), keepdim=True) + 1e-12)
            Hb = Hb / p

        Hb = Hb.to(torch.complex64)

        for k in range(K):
            f0 = k * subband_size
            f1 = f0 + subband_size
            Hsb = Hb[:, :, :, f0:f1]                          # [B, Nt, Nr, L]
            if subband_aggregation == "stack":
                Hsb = Hsb.reshape(B, Nt, Nr * subband_size)   # [B, Nt, L*Nr]
                Hmat = Hsb.transpose(-1, -2)                  # [B, L*Nr, Nt]
            else:  # "average"
                Hsb = Hsb.mean(dim=-1)                        # [B, Nt, Nr]
                Hmat = Hsb.transpose(-1, -2)                  # [B, Nr, Nt]
            R = torch.matmul(Hmat.conj().transpose(-1, -2), Hmat)  # [B, Nt, Nt]
            try:
                _, eigvecs = torch.linalg.eigh(R)                # [B, Nt, Nt]
            except RuntimeError:
                eigvecs = _eigh_fallback_svd(Hmat)           # safe fallback for ill-conditioned batches
            wk = eigvecs[:, :, -1]                           # [B, Nt]
            norm = torch.norm(wk, dim=-1, keepdim=True) + 1e-12
            wk = wk / norm
            if phase_fix:
                wk = _torch_fix_eigenvector_phase(wk)
            W[start:end, k, :] = wk.cpu().numpy().astype(np.complex64)

    return W


def compute_subband_signal_covariances(
    H: np.ndarray,
    subband_size: int,
    normalize_input_power: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    """Pre-compute signal covariance R_s = H_s @ H_s^H for every sample and subband.

    This replaces the O(n) eigendecomposition inside the per-SNR eval loop with a
    single O(1) lookup, dramatically accelerating noise robustness evaluation.

    Args:
        H: complex ndarray [N, Nt, Nr, Nf].
        subband_size: number of carriers per subband.
        normalize_input_power: if True, normalize each H by its power before computing.
        batch_size: samples processed per batch (GPU memory friendly).

    Returns:
        R: float32 ndarray [N, K, Nt, Nt], where K = Nf // subband_size.
    """
    if H.ndim != 4:
        raise ValueError(f"Expected H [N,Nt,Nr,Nf], got {H.shape}")
    if not np.iscomplexobj(H):
        H = H.astype(np.complex64)

    N, Nt, Nr, Nf = H.shape
    K = Nf // subband_size
    R = np.empty((N, K, Nt, Nt), dtype=np.complex64)

    batch_starts = range(0, N, batch_size)
    for start in progress(
        batch_starts, total=(N + batch_size - 1) // batch_size,
        desc="feedback preprocess: covariances", unit="batch", leave=False
    ):
        end = min(start + batch_size, N)
        Hb = H[start:end].astype(np.complex64, copy=False)
        if normalize_input_power:
            p = np.sqrt(np.mean(np.abs(Hb) ** 2, axis=(1, 2, 3), keepdims=True) + 1e-12)
            Hb = Hb / p

        for k in range(K):
            f0 = k * subband_size
            f1 = f0 + subband_size
            Hsb = np.mean(Hb[:, :, :, f0:f1], axis=-1)          # [B, Nt, Nr]
            Hmat = np.transpose(Hsb, (0, 2, 1))                  # [B, Nr, Nt]
            R[start:end, k] = np.einsum("brt,bru->btu", Hmat.conj(), Hmat, optimize=True)  # [B, Nt, Nt]

    return R


def compute_eigenvectors_from_covariance_batch(
    R: np.ndarray,
    phase_fix: bool = True,
) -> np.ndarray:
    """Eigendecompose pre-computed covariance matrices [N, K, Nt, Nt] -> [N, K, Nt].

    Args:
        R: complex covariance [N, K, Nt, Nt].
        phase_fix: apply deterministic phase fix to eigenvectors.

    Returns:
        W: complex eigenvectors [N, K, Nt], one dominant eigenvector per (sample, subband).
    """
    N, K, Nt = R.shape[0], R.shape[1], R.shape[2]
    W = np.empty((N, K, Nt), dtype=np.complex64)
    for n in progress(range(N), total=N,
                      desc="feedback preprocess: covariance eig", unit="sample", leave=False):
        for k in range(K):
            eigvals, eigvecs = np.linalg.eigh(R[n, k])
            wk = eigvecs[:, -1]
            wk = wk / (np.linalg.norm(wk) + 1e-12)
            if phase_fix:
                idx = np.argmax(np.abs(wk))
                wk = wk * np.exp(-1j * np.angle(wk[idx]))
            W[n, k] = wk.astype(np.complex64)
    return W


def compute_subband_eigenvectors_auto(
    H: np.ndarray,
    subband_size: int,
    phase_fix: bool = True,
    normalize_input_power: bool = False,
    batch_size: int = 1024,
    gpu: bool = True,
    subband_aggregation: str = "stack",
) -> np.ndarray:
    """Auto-dispatched eigenvector computation.

    Args:
        H: complex ndarray, [N, Nt, Nr, Nf] or [Nt, Nr, Nf].
        subband_size: number of sampled carriers/RBs per feedback subband.
        phase_fix: make eigenvector phase deterministic.
        normalize_input_power: normalize each raw CSI sample before extracting eigenvectors.
        batch_size: samples per batch (affects GPU memory usage).
        gpu: if True, use GPU when available; if False, always use CPU.
        subband_aggregation: "stack" (default, 3GPP-conformant) or "average".

    Returns:
        W: complex64 ndarray, [N, K, Nt], K=Nf/subband_size.
    """
    if gpu and torch.cuda.is_available():
        return compute_subband_eigenvectors_gpu(
            H, subband_size, phase_fix, normalize_input_power, batch_size,
            subband_aggregation=subband_aggregation,
        )
    return compute_subband_eigenvectors(
        H, subband_size, phase_fix, normalize_input_power,
        batch_size=max(batch_size, 64),  # ensure CPU path uses reasonable default
        subband_aggregation=subband_aggregation,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def eigenvectors_to_model_input(
    W: np.ndarray,
    layout: str = "paper",
) -> np.ndarray:
    """Convert W [N,K,Nt] complex to neural network input.

    Args:
        W: complex eigenvector array [N, K, Nt].
        layout: "paper" (default) -> [N, 2, Nt, K] matching paper Fig 2
                "image"           -> [N, 2, K, Nt] (legacy spatial layout)
    """
    ri = complex_to_ri_np(W, channel_first=True)          # [N, 2, K, Nt]
    if layout == "paper":
        return np.transpose(ri, (0, 1, 3, 2))              # [N, 2, Nt, K]
    if layout == "image":
        return ri
    raise ValueError("layout must be 'paper' or 'image'")


def model_output_to_eigenvectors(X: np.ndarray, layout: str = "paper") -> np.ndarray:
    """Convert network output to normalized complex W [N,K,Nt].

    Args:
        X: model output [N, 2, Nt, K] (paper layout) or [N, 2, K, Nt] (image layout).
        layout: "paper" (default) expects [N, 2, Nt, K].
    """
    if layout == "paper":
        X = np.transpose(X, (0, 1, 3, 2))               # [N, 2, K, Nt]
    W = X[:, 0] + 1j * X[:, 1]                         # [N, K, Nt]
    return normalize_complex_np(W, axis=-1).astype(np.complex64)

"""
任务性能指标
============
在默认设置（task=joint, mask_mode=comb, mask_ratio=0.5, max_samples=15000）下
计算任务相关的各项指标。

指标包括:
- NMSE (dB): 全局归一化均方误差
- SGCS: 谱广义余弦相似度
- per-mask NMSE/SGCS: 在各 mask 模式下的表现
- per-subcarrier NMSE: 每个子载波上的 NMSE
- per-port NMSE: 每个发送端口上的 NMSE
"""
import os
import sys
import numpy as np
import torch
from typing import Dict, List, Optional

from csi_eval.progress import progress

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ..params.params_7GHz import NT_PORT, NR, NF
from ..data_processing.mask_strategies import apply_mask
from ..model_registry import infer as registry_infer
from ..model_registry import ModelWrapper

# ----------------------------------------------------------------------
# 核心指标计算
# ----------------------------------------------------------------------

def _to_complex(H: np.ndarray) -> np.ndarray:
    """(2, ...) -> 复数"""
    if isinstance(H, torch.Tensor):
        return torch.complex(H[0], H[1])
    return H[0] + 1j * H[1]


def nmse_np(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """NMSE (dB) — numpy 版本，跨 batch 全局求和"""
    pred_c = _to_complex(H_pred)
    true_c = _to_complex(H_true)
    num = np.sum(np.abs(pred_c - true_c) ** 2)
    den = np.sum(np.abs(true_c) ** 2) + 1e-12
    return 10.0 * np.log10(num / den)


# ----------------------------------------------------------------------
# 3GPP SGCS (TR 38.843)
# ----------------------------------------------------------------------
def _rb_to_subband_indices(nf: int, rb_per_subband: int = 4) -> List[np.ndarray]:
    """Split ``nf`` RBs into contiguous frequency subbands.

    The last subband keeps only the actually available RBs instead of padding
    by duplicating the final RB.  With the default prediction data
    ``Nf=52`` and ``rb_per_subband=4`` this yields exactly 13 subbands.
    """
    if nf <= 0:
        raise ValueError(f"nf must be positive, got {nf}")
    if rb_per_subband <= 0:
        raise ValueError(f"rb_per_subband must be positive, got {rb_per_subband}")
    return [
        np.arange(start, min(start + rb_per_subband, nf), dtype=np.int64)
        for start in range(0, nf, rb_per_subband)
    ]


def _rb_covariances(H: np.ndarray) -> np.ndarray:
    """Compute one transmit-side covariance matrix for every RB.

    Args:
        H: complex channel with shape ``(Nt, Nr, Nf)``.

    Returns:
        ``R_rb`` with shape ``(Nf, Nt, Nt)`` where
        ``R_rb[f] = H_f @ H_f^H / Nr``.
    """
    H = np.asarray(H)
    if H.ndim != 3:
        raise ValueError(f"Expected H shape (Nt,Nr,Nf), got {H.shape}")
    Nt, Nr, Nf = H.shape
    # (Nt, Nr, Nf) -> (Nf, Nt, Nr)
    Hf = np.transpose(H, (2, 0, 1))
    R_rb = Hf @ np.swapaxes(Hf.conj(), -1, -2)
    return R_rb / max(int(Nr), 1)


def _topk_eigenvectors_from_covariance(
    R: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SVD a covariance matrix and return the top-k modes in descending order.

    The returned rows are the right singular/eigen vectors associated with the
    largest singular values.  Explicit sorting is kept even though NumPy SVD
    already returns singular values in descending order; this makes the stream
    ordering requirement unambiguous and robust to implementation changes.

    Returns:
        ``(V, values)`` where ``V.shape == (k, Nt)`` and ``values`` are in
        descending order.
    """
    R = np.asarray(R)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"Expected square covariance matrix, got {R.shape}")
    if k <= 0 or k > R.shape[0]:
        raise ValueError(f"top_k must be in [1,{R.shape[0]}], got {k}")

    # Numerical round-off may introduce tiny non-Hermitian components.
    R_h = 0.5 * (R + R.conj().T)
    _, singular_values, Vh = np.linalg.svd(R_h, full_matrices=False)
    order = np.argsort(singular_values)[::-1][:k]
    # Each row is one v_i^H conjugated back to v_i^T, i.e. vector in C^Nt.
    V = Vh[order, :].conj()
    return V, singular_values[order]


def sgcs_3gpp_per_subband(
    H_pred: np.ndarray,
    H_true: np.ndarray,
    top_k: int = 4,
    rb_per_subband: int = 4,
    eps: float = 1e-12,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute per-subband, per-stream SGCS using the requested 38.843 flow.

    Processing for each sample is exactly:

    1. Compute the transmit covariance on every RB, producing
       ``(RB, Tx, Tx)``.
    2. Average those covariance matrices over each frequency subband, producing
       ``(subband, Tx, Tx)``.
    3. SVD every averaged covariance matrix.
    4. Select the first ``top_k`` vectors corresponding to the largest
       covariance singular/eigen values and compare the same ordered stream
       between prediction and target using squared generalized cosine
       similarity.

    ``stream 1`` therefore means the dominant covariance mode, ``stream 2``
    the second mode, etc.  This fixes eigenvector-disorder caused by comparing
    vectors without an explicit dominant-mode ordering.

    Args:
        H_pred: ``(B,2,Nt,Nr,Nf)`` or single-sample ``(2,Nt,Nr,Nf)``.
        H_true: same shape as ``H_pred``.
        top_k: number of ordered streams/layers, normally 4.
        rb_per_subband: RBs averaged into one subband; default 4, giving 13
            subbands for the package's 52-RB prediction data.
        show_progress: show a per-sample covariance/SVD progress bar when a
            batch contains more than one sample.  This is useful because the
            exact 256x256 SVD path is relatively expensive.

    Returns:
        ``(B*n_subbands, top_k)`` array.  Column ``j`` is SGCS for stream
        ``j+1`` in descending covariance-eigenvalue order.
    """
    H_pred = np.asarray(H_pred)
    H_true = np.asarray(H_true)
    if H_pred.shape != H_true.shape:
        raise ValueError(
            f"H_pred and H_true must have identical shape, got "
            f"{H_pred.shape} vs {H_true.shape}"
        )

    if H_pred.ndim == 4:
        H_pred = H_pred[None, ...]
        H_true = H_true[None, ...]
    if H_pred.ndim != 5 or H_pred.shape[1] != 2:
        raise ValueError(
            "Expected H shape (B,2,Nt,Nr,Nf) or (2,Nt,Nr,Nf), "
            f"got {H_pred.shape}"
        )

    pred_c = H_pred[:, 0] + 1j * H_pred[:, 1]
    true_c = H_true[:, 0] + 1j * H_true[:, 1]
    n_samples, Nt, _, Nf = pred_c.shape
    if top_k > Nt:
        raise ValueError(f"top_k={top_k} exceeds Nt={Nt}")

    subbands = _rb_to_subband_indices(Nf, rb_per_subband=rb_per_subband)
    sgcs_rows: List[np.ndarray] = []

    sample_iter = range(n_samples)
    if show_progress and n_samples > 1:
        sample_iter = progress(
            sample_iter, total=n_samples, desc="prediction SGCS: covariance/SVD",
            unit="sample", leave=False
        )

    for b in sample_iter:
        # Required intermediate: (RB, Tx, Tx)
        R_pred_rb = _rb_covariances(pred_c[b])
        R_true_rb = _rb_covariances(true_c[b])

        for rb_idx in subbands:
            # Required intermediate: average RB covariance -> (Tx, Tx)
            R_pred_sub = R_pred_rb[rb_idx].mean(axis=0)
            R_true_sub = R_true_rb[rb_idx].mean(axis=0)

            V_pred, _ = _topk_eigenvectors_from_covariance(R_pred_sub, top_k)
            V_true, _ = _topk_eigenvectors_from_covariance(R_true_sub, top_k)

            inner = np.sum(np.conj(V_true) * V_pred, axis=-1)
            n_true = np.linalg.norm(V_true, axis=-1)
            n_pred = np.linalg.norm(V_pred, axis=-1)
            denom = n_true * n_pred + eps
            sgcs_pair = np.clip(np.abs(inner) / denom, 0.0, 1.0) ** 2
            sgcs_rows.append(sgcs_pair.astype(np.float64, copy=False))

    if not sgcs_rows:
        return np.empty((0, top_k), dtype=np.float64)
    return np.stack(sgcs_rows, axis=0)


def sgcs_3gpp_per_subband_torch(
    H_pred: torch.Tensor,
    H_true: torch.Tensor,
    top_k: int = 4,
    rb_per_subband: int = 4,
    eps: float = 1e-12,
) -> torch.Tensor:
    """GPU-batched implementation of the 3GPP SGCS definition.

    This preserves the NumPy implementation's operation order: per-RB
    transmit covariance, arithmetic mean inside each contiguous subband, then
    the ordered top-k covariance eigenvectors.  It returns ``(B*S, top_k)``
    on the same device, where ``S`` is the number of subbands.
    """
    if H_pred.shape != H_true.shape:
        raise ValueError(f"H_pred and H_true must have identical shape, got {H_pred.shape} vs {H_true.shape}")
    if H_pred.ndim == 4:
        H_pred = H_pred.unsqueeze(0)
        H_true = H_true.unsqueeze(0)
    if H_pred.ndim != 5 or H_pred.shape[1] != 2:
        raise ValueError(f"Expected H shape (B,2,Nt,Nr,Nf), got {tuple(H_pred.shape)}")

    _, _, nt, nr, nf = H_pred.shape
    if top_k <= 0 or top_k > nt:
        raise ValueError(f"top_k must be in [1,{nt}], got {top_k}")

    def covariance_by_subband(h: torch.Tensor) -> torch.Tensor:
        # (B, 2, Nt, Nr, Nf) -> (B, Nf, Nt, Nr).  Einsum directly computes
        # the mean of the required per-RB covariances for each subband.
        h_complex = torch.complex(h[:, 0], h[:, 1]).permute(0, 3, 1, 2)
        chunks = []
        for start in range(0, nf, rb_per_subband):
            rb = h_complex[:, start:start + rb_per_subband]
            chunks.append(torch.einsum('brtn,brun->btu', rb, rb.conj()) / (rb.shape[1] * nr))
        return torch.stack(chunks, dim=1)

    r_pred = covariance_by_subband(H_pred)
    r_true = covariance_by_subband(H_true)

    # Floating-point accumulation can leave an otherwise positive-semidefinite
    # covariance nearly singular.  Restore exact Hermitian symmetry and apply
    # scale-aware diagonal loading before the GPU eigensolver.
    def regularize_covariance(r: torch.Tensor) -> torch.Tensor:
        r = 0.5 * (r + r.mH)
        real_dtype = r.real.dtype
        diagonal_scale = r.diagonal(dim1=-2, dim2=-1).real.abs().mean(dim=-1)
        loading = (diagonal_scale * 1e-6).clamp_min(torch.finfo(real_dtype).eps)
        identity = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
        return r + loading[..., None, None] * identity

    r_pred = regularize_covariance(r_pred)
    r_true = regularize_covariance(r_true)
    # Covariances are Hermitian. eigh is substantially cheaper and more
    # numerically appropriate than a full SVD while producing the same modes.
    _, v_pred = torch.linalg.eigh(r_pred)
    _, v_true = torch.linalg.eigh(r_true)
    v_pred = v_pred[..., -top_k:].flip(-1)
    v_true = v_true[..., -top_k:].flip(-1)
    inner = (v_true.conj() * v_pred).sum(dim=-2)
    denom = torch.linalg.vector_norm(v_true, dim=-2) * torch.linalg.vector_norm(v_pred, dim=-2) + eps
    return (inner.abs() / denom).clamp_(0.0, 1.0).square().reshape(-1, top_k)


def sgcs_stream_means(sgcs_values: np.ndarray, top_k: int = 4) -> List[float]:
    """Return one mean SGCS for each ordered covariance stream/layer."""
    arr = np.asarray(sgcs_values, dtype=np.float64)
    if arr.size == 0:
        return [float("nan")] * top_k
    arr = arr.reshape(-1, top_k)
    return [float(x) for x in np.nanmean(arr, axis=0)]


def sgcs_np(
    H_pred: np.ndarray,
    H_true: np.ndarray,
    top_k: int = 4,
    rb_per_subband: int = 4,
    eps: float = 1e-12,
) -> float:
    """Mean SGCS over samples, subbands, and the ordered top-k streams."""
    arr = sgcs_3gpp_per_subband(
        H_pred,
        H_true,
        top_k=top_k,
        rb_per_subband=rb_per_subband,
        eps=eps,
    )
    return float(arr.mean()) if arr.size else float("nan")


def nmse_masked_np(H_pred: np.ndarray, H_true: np.ndarray,
                   mask: np.ndarray) -> float:
    """NMSE (dB) — 仅在 mask=False 位置（待预测）"""
    pred_c = _to_complex(H_pred)
    true_c = _to_complex(H_true)
    inv_mask = ~mask
    num = np.sum(np.abs(pred_c - true_c) ** 2 * inv_mask)
    den = np.sum(np.abs(true_c) ** 2 * inv_mask) + 1e-12
    if den < 1e-10:
        return np.nan   # mask 全 True，无可预测位置
    return 10.0 * np.log10(num / den)


def sgcs_masked_np(H_pred: np.ndarray, H_true: np.ndarray,
                   mask: np.ndarray) -> float:
    """SGCS — 仅在 mask=False 位置"""
    pred_c = _to_complex(H_pred)
    true_c = _to_complex(H_true)
    inv_mask = ~mask
    num = np.sum(np.conj(pred_c) * true_c * inv_mask)
    den = (np.sqrt(np.sum(np.abs(pred_c) ** 2 * inv_mask)) *
           np.sqrt(np.sum(np.abs(true_c) ** 2 * inv_mask)) + 1e-12)
    return np.abs(num) / den


# ----------------------------------------------------------------------
# 逐子载波 / 逐端口分析
# ----------------------------------------------------------------------

def per_subcarrier_nmse(H_pred: np.ndarray, H_true: np.ndarray) -> np.ndarray:
    """
    计算每个子载波的 NMSE (dB)。

    对所有 Nt_port 和 Nr 取平均后，逐 Nf 返回。

    Returns:
        nmse_per_sc: (Nf,) = (52,)
    """
    pred_c = _to_complex(H_pred)   # (Nt, Nr, Nf)
    true_c = _to_complex(H_true)
    # 沿 (Nt, Nr) 维度求和
    diff_sq = np.abs(pred_c - true_c) ** 2          # (Nt, Nr, Nf)
    true_sq = np.abs(true_c) ** 2                    # (Nt, Nr, Nf)
    nmse_per_sc = 10.0 * np.log10(
        diff_sq.sum(axis=(0, 1)) / (true_sq.sum(axis=(0, 1)) + 1e-12)
    )  # (Nf,)
    return nmse_per_sc


def per_port_nmse(H_pred: np.ndarray, H_true: np.ndarray) -> np.ndarray:
    """
    计算每个发送端口的 NMSE (dB)。

    对所有 Nr 和 Nf 取平均后，逐 Nt 返回。

    Returns:
        nmse_per_port: (Nt_port,) = (256,)
    """
    pred_c = _to_complex(H_pred)
    true_c = _to_complex(H_true)
    diff_sq = np.abs(pred_c - true_c) ** 2
    true_sq = np.abs(true_c) ** 2
    nmse_per_port = 10.0 * np.log10(
        diff_sq.sum(axis=(1, 2)) / (true_sq.sum(axis=(1, 2)) + 1e-12)
    )  # (Nt_port,)
    return nmse_per_port


# ----------------------------------------------------------------------
# 综合评估函数
# ----------------------------------------------------------------------

def compute_task_metrics(model,
                         samples: List[dict],
                         task: str = 'joint',
                         mask_mode: str = 'comb',
                         mask_ratio: float = 0.5,
                         device: str = 'cuda',
                         max_samples: int = 15000,
                         batch_size: int = 256,
                         sgcs_top_k: int = 4,
                         sgcs_rb_per_subband: int = 4,
                         ) -> Dict:
    """
    计算任务性能指标。

    Args:
        model: 已加载权重的模型实例
        samples: sample dict 列表
        task, mask_mode, mask_ratio: 评估配置
        device: 模型所在设备
        max_samples: 最大评估样本数
        batch_size: 推理批大小
        sgcs_top_k: 3GPP SGCS 计算时 SVD 取的主特征向量数（默认 4）。
        sgcs_rb_per_subband: 3GPP SGCS 计算时每个子带包含的 RB 数（默认 4，对应
            30kHz SCS、20MHz 带宽，13 个子带）。

    Returns:
        results: {
            'nmse_db': float,          # 全局 NMSE (dB)
            'sgcs': float,              # 全局 SGCS（3GPP 标准：对每个子带的
                                         # 平均协方差矩阵 SVD 后取前 top_k
                                         # 个主特征向量，再与真实特征向量求
                                         # 平方广义余弦相似度）
            'per_subcarrier_nmse': np.array(Nf),   # 逐子载波
            'per_port_nmse': np.array(Nt_port),    # 逐端口
            'n_samples': int,
        }
    """
    from ..data_loader import prepare_batch
    from ..data_processing.mask_strategies import get_mask_strategy



    # 仅对 nn.Module 调用 eval()，传统算法无需此步
    if isinstance(model, torch.nn.Module):
        model.eval()
    elif hasattr(model, 'eval'):
        model.eval()

    assert task in ('joint', 'spatial', 'frequency'), \
        f'compute_task_metrics 收到非法 task: {task!r}'

    # 数据集样本截断
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    is_wrapped = isinstance(model, ModelWrapper)
    is_nn      = isinstance(model, torch.nn.Module)
    is_trad    = not is_nn and not is_wrapped

    # 累加器
    total_nmse_num = 0.0
    total_nmse_den = 0.0
    # 3GPP SGCS：累积逐 (样本, 子带, 特征向量) 的 SGCS 平方值，最后取全局平均。
    sgcs_global_sum = 0.0
    sgcs_global_count = 0
    sgcs_stream_sum = np.zeros(sgcs_top_k, dtype=np.float64)
    sgcs_stream_count = 0
    n_samples = 0

    # GPU 上的累加张量（避免每次迭代都同步回 CPU）
    per_sc_num_t = torch.zeros(NF, device=device if not is_trad else 'cpu')
    per_sc_den_t = torch.zeros(NF, device=device if not is_trad else 'cpu')
    per_port_num_t = torch.zeros(NT_PORT, device=device if not is_trad else 'cpu')
    per_port_den_t = torch.zeros(NT_PORT, device=device if not is_trad else 'cpu')

    per_sc_num = np.zeros(NF)
    per_sc_den = np.zeros(NF)
    per_port_num = np.zeros(NT_PORT)
    per_port_den = np.zeros(NT_PORT)

    with torch.no_grad():
        if is_trad:
            # 传统算法：每个样本单独调用（接口只接受单样本）
            for sample in progress(
                samples, desc=f"prediction {task}/{mask_mode}: samples",
                unit="sample", leave=False
            ):
                H_known, H_target, mask_np = prepare_batch(
                    sample, task, mask_mode, mask_ratio)
                H_known_complex = (H_known[0] + 1j * H_known[1])
                H_pred_complex = model(H_known_complex, mask_np, task)
                H_pred_np = np.stack(
                    [H_pred_complex.real, H_pred_complex.imag], axis=0)

                pred_c = H_pred_np[0] + 1j * H_pred_np[1]
                true_c = H_target[0] + 1j * H_target[1]
                # **关键修正**：仅在 mask=True（待预测）位置评估
                if task == 'frequency':
                    mask_3d = mask_np[None, None, :]
                elif task == 'spatial':
                    mask_3d = mask_np[:, None, None]
                else:
                    mask_3d = mask_np[:, None, :]
                diff_sq = np.abs(pred_c - true_c) ** 2 * mask_3d
                true_sq = np.abs(true_c) ** 2  # 全值：分母不 mask

                total_nmse_num += diff_sq.sum()
                total_nmse_den += true_sq.sum()

                # 3GPP SGCS：mask=False 是已知区域，用 target 回填为完美预测，
                # 使 SGCS 只反映 mask=True 待预测区域，与神经网络分支保持一致。
                pred_eval_c = pred_c.copy()
                if task == 'frequency':
                    pred_eval_c[:, :, ~mask_np] = true_c[:, :, ~mask_np]
                elif task == 'spatial':
                    pred_eval_c[~mask_np, :, :] = true_c[~mask_np, :, :]
                else:
                    pred_eval_c = pred_eval_c * mask_3d + true_c * (~mask_3d)
                H_pred_for_sgcs = np.stack([pred_eval_c.real, pred_eval_c.imag], axis=0)
                sgcs_vals = sgcs_3gpp_per_subband(
                    H_pred_for_sgcs[None], H_target[None],
                    top_k=sgcs_top_k, rb_per_subband=sgcs_rb_per_subband,
                )
                if sgcs_vals.size:
                    sgcs_global_sum += float(sgcs_vals.sum())
                    sgcs_global_count += int(sgcs_vals.size)
                    sgcs_stream_sum += sgcs_vals.sum(axis=0)
                    sgcs_stream_count += int(sgcs_vals.shape[0])

                per_sc_num += diff_sq.sum(axis=(0, 1))
                per_sc_den += true_sq.sum(axis=(0, 1))
                per_port_num += diff_sq.sum(axis=(1, 2))
                per_port_den += true_sq.sum(axis=(1, 2))
                n_samples += 1
        else:
            bs = max(1, int(batch_size))
            batch_starts = range(0, len(samples), bs)
            for start in progress(
                batch_starts, total=(len(samples) + bs - 1) // bs,
                desc=f"prediction {task}/{mask_mode}: inference",
                unit="batch", leave=False
            ):
                chunk = samples[start:start + bs]
                B = len(chunk)
                H_known_list, H_target_list, mask_list = [], [], []
                for sample in chunk:
                    H_k, H_t, m = prepare_batch(sample, task, mask_mode, mask_ratio)
                    H_known_list.append(H_k)
                    H_target_list.append(H_t)
                    mask_list.append(m)
                H_known_np = np.stack(H_known_list, axis=0)
                H_target_np = np.stack(H_target_list, axis=0)
                # 所有样本共用同一个 mask（取第一个）
                mask_np = mask_list[0]
                H_known_t = torch.from_numpy(H_known_np).to(device, non_blocking=True)
                H_target_t = torch.from_numpy(H_target_np).to(device, non_blocking=True)
                # 统一推理：自动适配 ModelWrapper / 普通 nn.Module 的任意签名
                H_pred_t = registry_infer(model, H_known_t, mask_np, task, device)

                # GPU 上算复数指标 (B, 2, Nt, Nr, Nf) -> (B, Nt, Nr, Nf) complex
                pred_c = torch.complex(H_pred_t[:, 0], H_pred_t[:, 1])  # (B, Nt, Nr, Nf)
                true_c = torch.complex(H_target_t[:, 0], H_target_t[:, 1])
                diff_c = pred_c - true_c
                # **关键修正**：NMSE / SGCS 只在 mask=True（待预测）位置计算。
                # mask=False 是已知输入，模型 copy 该位置 H_true，
                # 不应纳入预测质量评估（否则指标会被人为压低）。
                if task == 'frequency':
                    mask_3d = torch.from_numpy(mask_np).reshape(1, 1, 1, -1).to(device)
                elif task == 'spatial':
                    mask_3d = torch.from_numpy(mask_np).reshape(1, -1, 1, 1).to(device)
                else:  # joint: mask_np 是 (Nt_port, Nf) → (1, Nt_port, 1, Nf)
                    mask_3d = torch.from_numpy(mask_np).reshape(
                        1, mask_np.shape[0], 1, mask_np.shape[1]
                    ).to(device)
                mask_3d_f = mask_3d.to(diff_c.dtype)
                diff_sq = (diff_c.real ** 2 + diff_c.imag ** 2) * mask_3d_f
                true_sq = (true_c.real ** 2 + true_c.imag ** 2)  # 全值：分母不 mask

                total_nmse_num += diff_sq.real.sum().item()
                total_nmse_den += true_sq.real.sum().item()

                # 3GPP SGCS stays on the GPU.  Fill known positions from the
                # target so its evaluation scope remains mask=True only.
                pred_eval_t = torch.where(mask_3d.unsqueeze(1), H_pred_t, H_target_t)
                sgcs_vals_t = sgcs_3gpp_per_subband_torch(
                    pred_eval_t, H_target_t,
                    top_k=sgcs_top_k, rb_per_subband=sgcs_rb_per_subband,
                )
                if sgcs_vals_t.numel():
                    sgcs_global_sum += float(sgcs_vals_t.sum().item())
                    sgcs_global_count += int(sgcs_vals_t.numel())
                    sgcs_stream_sum += sgcs_vals_t.sum(dim=0).cpu().numpy()
                    sgcs_stream_count += int(sgcs_vals_t.shape[0])

                # per-subcarrier (Nf) 和 per-port (Nt) 在 GPU 上聚合
                per_sc_num_t += diff_sq.real.sum(dim=(0, 1, 2))   # (Nf,)
                per_sc_den_t += true_sq.real.sum(dim=(0, 1, 2))
                per_port_num_t += diff_sq.real.sum(dim=(0, 2, 3))  # (Nt,)
                per_port_den_t += true_sq.real.sum(dim=(0, 2, 3))

                n_samples += B

            # 把 per-sc/per-port 张量拷回 CPU
            per_sc_num = per_sc_num_t.cpu().numpy()
            per_sc_den = per_sc_den_t.cpu().numpy()
            per_port_num = per_port_num_t.cpu().numpy()
            per_port_den = per_port_den_t.cpu().numpy()

    # 汇总
    if total_nmse_num.real < 1e-10:
        global_nmse = float('-inf')
    else:
        global_nmse = 10.0 * np.log10(total_nmse_num.real / (total_nmse_den.real + 1e-12))

    if sgcs_global_count > 0:
        global_sgcs = sgcs_global_sum / sgcs_global_count
    else:
        global_sgcs = float('nan')
    if sgcs_stream_count > 0:
        sgcs_streams = (sgcs_stream_sum / sgcs_stream_count).astype(np.float64).tolist()
    else:
        sgcs_streams = [float('nan')] * sgcs_top_k

    # per-subcarrier / per-port：避免除零产生 -inf
    per_subcarrier = np.full(NF, np.nan)
    valid_sc = per_sc_den > 1e-10
    zero_error_sc = valid_sc & (per_sc_num <= 0.0)
    per_subcarrier[zero_error_sc] = -np.inf
    positive_error_sc = valid_sc & ~zero_error_sc
    if positive_error_sc.any():
        per_subcarrier[positive_error_sc] = 10.0 * np.log10(
            per_sc_num[positive_error_sc] /
            (per_sc_den[positive_error_sc] + 1e-12))

    per_port = np.full(NT_PORT, np.nan)
    valid_pt = per_port_den > 1e-10
    zero_error_pt = valid_pt & (per_port_num <= 0.0)
    per_port[zero_error_pt] = -np.inf
    positive_error_pt = valid_pt & ~zero_error_pt
    if positive_error_pt.any():
        per_port[positive_error_pt] = 10.0 * np.log10(
            per_port_num[positive_error_pt] /
            (per_port_den[positive_error_pt] + 1e-12))

    result = {
        'nmse_db': float(global_nmse),
        'sgcs': float(global_sgcs),
        'sgcs_streams': [float(x) for x in sgcs_streams],
        'sgcs_stream_order': 'descending_covariance_eigenvalue',
        'sgcs_top_k': int(sgcs_top_k),
        'sgcs_rb_per_subband': int(sgcs_rb_per_subband),
        'sgcs_n_subbands': len(_rb_to_subband_indices(NF, sgcs_rb_per_subband)),
        'per_subcarrier_nmse': per_subcarrier.tolist(),
        'per_port_nmse': per_port.tolist(),
        'n_samples': n_samples,
    }
    for i, value in enumerate(sgcs_streams, start=1):
        result[f'sgcs_stream_{i}'] = float(value)
    return result


def compute_per_mask_metrics(model,
                              samples: List[dict],
                              mask_modes: List[str],
                              task: str = 'joint',
                              mask_ratio: float = 0.5,
                              device: str = 'cuda',
                              max_samples: int = 15000) -> Dict[str, Dict]:
    """
    在各 mask 模式下分别评估 NMSE / SGCS。

    Returns:
        {
            'comb':  {'nmse_db': ..., 'sgcs': ...},
            'grid':  {'nmse_db': ..., 'sgcs': ...},
            ...
        }
    """
    results = {}
    for mode in progress(mask_modes, desc="prediction: mask modes", unit="mode", leave=False):
        # 对 grid/random_2d/comb 各模式单独生成 mask
        # 注意：若某模式+ratio 组合导致 mask 全 True（无可预测位置），NMSE 会是 nan
        r = compute_task_metrics(
            model, samples, task=task, mask_mode=mode,
            mask_ratio=mask_ratio, device=device,
            max_samples=max_samples)
        results[mode] = {
            'nmse_db': r.get('nmse_db'),
            'sgcs': r.get('sgcs'),
            'sgcs_streams': r.get('sgcs_streams'),
            'sgcs_stream_order': r.get('sgcs_stream_order'),
        }
        for i, value in enumerate(r.get('sgcs_streams') or [], start=1):
            results[mode][f'sgcs_stream_{i}'] = value
    return results

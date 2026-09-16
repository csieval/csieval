"""
数据预处理工具
==============

提供复数化、归一化、归一化逆变换等工具。

数据格式说明:
- 存储: (2048, 52) complex64 (不再分实部/虚部通道)
- 模型输入: DataLoader 中转 (B, 2, 256, 8, 52) 实部+虚部
- 内存/模型内部: 复数或 (2, ...) 实部+虚部
"""

import numpy as np
import torch


def complex_from_npy(npy_array):
    """(2048, 52) 复数 -> (2048, 52) complex64

    从 .npy 文件读取后还原复数 (不拆分通道)
    """
    if isinstance(npy_array, torch.Tensor):
        npy_array = npy_array.cpu().numpy()
    # npy_array 本身是 complex64 复数数组，直接返回
    return np.asarray(npy_array, dtype=np.complex64)


def npy_from_complex(H_complex):
    """复数 (Nt, Nf) / (2048, 52) -> (2, 2048, 52) 实部+虚部

    用于 DataLoader 包装: 从存储的复数转为模型输入格式
    """
    if isinstance(H_complex, torch.Tensor):
        H_complex = H_complex.cpu().numpy()
    return np.stack([H_complex.real, H_complex.imag], axis=0).astype(np.float32)


def channel_to_3d(H_flat, nt_port=256, nr=8):
    """(2048, 52) -> (256, 8, 52)"""
    nt_total, nf = H_flat.shape
    assert nt_total == nt_port * nr, \
        f'shape mismatch: {nt_total} != {nt_port}*{nr}'
    return H_flat.reshape(nt_port, nr, nf)


def normalize_complex(H, norm_factor=None):
    """归一化复数信道

    Returns:
        H_norm: 归一化后的数组
        norm_factor: 反归一化系数 (乘回去)
    """
    if norm_factor is None:
        norm_factor = np.sqrt(np.mean(np.abs(H) ** 2)) + 1e-8
    return H / norm_factor, norm_factor


def denormalize_complex(H_norm, norm_factor):
    """归一化逆变换"""
    return H_norm * norm_factor


def to_real_imag_tensors(H_complex, device='cpu'):
    """复数 numpy -> (2, ...) 实部+虚部 torch tensor

    Args:
        H_complex: (B, 2048, 52) complex64 或 (2048, 52) complex64
        device: 输出设备

    Returns:
        torch.Tensor (2, B, 2048, 52) 或 (2, 2048, 52)
    """
    if isinstance(H_complex, np.ndarray):
        H_complex = torch.from_numpy(H_complex)
    real = H_complex.real.float()
    imag = H_complex.imag.float()
    if H_complex.dim() == 2:
        return torch.stack([real, imag], dim=0).to(device)
    else:  # B, 2048, 52
        return torch.stack([real, imag], dim=1).to(device)


def from_real_imag_tensors(H_real_imag):
    """(B, 2, 2048, 52) / (2, 2048, 52) torch tensor -> (B, 2048, 52) / (2048, 52) 复数

    用于模型输出转复数
    """
    if isinstance(H_real_imag, np.ndarray):
        H_real_imag = torch.from_numpy(H_real_imag)
    if H_real_imag.dim() == 4:
        return torch.complex(H_real_imag[:, 0], H_real_imag[:, 1])
    return torch.complex(H_real_imag[0], H_real_imag[1])


def complex_magnitude_batch(H):
    """计算复数批量幅度"""
    if isinstance(H, torch.Tensor):
        return torch.sqrt(H.real ** 2 + H.imag ** 2 + 1e-8)
    return np.sqrt(H.real ** 2 + H.imag ** 2 + 1e-8)


def normalize_per_sample(H_batch, eps=1e-8):
    """对每个样本独立归一化 (按 |H|^2 平均)

    Args:
        H_batch: (B, 2, ...) 实部虚部格式
    Returns:
        H_norm: 归一化
        norms: (B,) 归一化系数
    """
    if isinstance(H_batch, torch.Tensor):
        H_complex = torch.complex(H_batch[:, 0], H_batch[:, 1])
        power = torch.mean(torch.abs(H_complex) ** 2, dim=tuple(range(1, H_complex.ndim)))
        norms = torch.sqrt(power + eps)
        H_norm = H_batch / norms.view(-1, *([1] * (H_batch.ndim - 1)))
        return H_norm, norms
    else:
        H_complex = H_batch[:, 0] + 1j * H_batch[:, 1]
        power = np.mean(np.abs(H_complex) ** 2, axis=tuple(range(1, H_complex.ndim)))
        norms = np.sqrt(power + eps)
        H_norm = H_batch / norms.reshape(-1, *([1] * (H_batch.ndim - 1)))
        return H_norm, norms

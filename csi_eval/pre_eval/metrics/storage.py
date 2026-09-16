"""
存储与部署指标
==============
- Params (M): 可训练参数量
- FP32 体积 (MB): float32 = 4 bytes / param
- 量化位宽: FP16 (2B), INT8 (1B), INT4 (0.5B, 估算)
"""
import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def count_parameters(model) -> int:
    """统计可训练参数量（不含 frozen BN 等）"""
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0  # 传统算法无参数


def model_size_bytes(model, dtype: str = 'fp32') -> int:
    """
    模型存储体积（字节数）。

    Args:
        model: nn.Module
        dtype: 'fp32' | 'fp16' | 'int8' | 'int4'
    """
    n_params = count_parameters(model)
    BYTES = {'fp32': 4, 'fp16': 2, 'int8': 1, 'int4': 0.5}
    return int(n_params * BYTES.get(dtype, 4))


def compute_storage_metrics(model) -> dict:
    """
    计算存储与部署指标。

    Returns:
        {
            'params_M': float,          # 参数量 (M)
            'fp32_MB': float,           # FP32 体积 (MB)
            'fp16_MB': float,           # FP16 体积 (MB)
            'int8_MB': float,           # INT8 体积 (MB)
            'int4_MB': float,           # INT4 估算体积 (MB)
            'n_params': int,            # 参数量原始值
        }
    """
    n = count_parameters(model)
    MB = 1024 * 1024
    return {
        'params_M': round(n / 1_000_000, 3),
        'fp32_MB': round(model_size_bytes(model, 'fp32') / MB, 3),
        'fp16_MB': round(model_size_bytes(model, 'fp16') / MB, 3),
        'int8_MB': round(model_size_bytes(model, 'int8') / MB, 3),
        'int4_MB': round(model_size_bytes(model, 'int4') / MB, 3),
        'n_params': n,
    }


def estimate_int8_compression_ratio(model) -> float:
    """
    估算 INT8 量化后的压缩率（相对于 FP32）。

    INT8 非线性量化通常可达到 ~4x 压缩率（去除量化误差后约 3.5-4x）。
    这里返回理论值 4.0。
    """
    return 4.0


def estimate_int4_compression_ratio(model) -> float:
    """INT4 理论压缩率 ~8x"""
    return 8.0

"""
统一数据加载器
==============
提供一套接口加载 Scenario1 / Scenario2 数据集，支持任意 task / mask_mode / mask_ratio。
未来新增数据集只需在此文件中添加加载逻辑。
"""
import os
import sys
import numpy as np
import torch
from typing import List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from .params.params_7GHz import NT_PORT, NR, NF, NPY_SHAPE
from .data_processing.dataset import (
    scan_dataset, split_dataset,
    CSIPredictionDataset,
)
from .data_processing.mask_strategies import apply_mask
from .data_processing.preprocessing import complex_from_npy

# ----------------------------------------------------------------------
# 加载数据
# ----------------------------------------------------------------------

def load_scenario1(data_dir: str = None,
                   max_samples: int = None,
                   split: str = 'test',
                   train_envs: int = 1000,
                   seed: int = 42) -> List[dict]:
    """
    加载 Scenario1 数据。

    Args:
        data_dir: 数据目录，默认使用 S1_DATA_DIR
        max_samples: 最大样本数（用于评估快速验证）
        split: 'train' | 'val' | 'test' | 'all'
        train_envs: 前多少个 map ID 作为训练集（默认 1000）
        seed: 随机种子（用于 train/val/test 划分）

    Returns:
        samples: list of dict {env, link, path}
    """
    if data_dir is None:
        from . import config as cfg
        data_dir = cfg.S1_DATA_DIR
    samples = scan_dataset(data_dir, max_samples=max_samples, scenario=1)
    train_s, val_s, test_s = split_dataset(samples, train_envs=train_envs, scenario=1)

    if split == 'train':
        return train_s
    elif split == 'val':
        return val_s
    elif split == 'test':
        return test_s
    else:
        return train_s + val_s + test_s


# def load_scenario2(data_dir: str = None,
#                    max_samples: int = None) -> List[dict]:
#     """
#     加载 Scenario2 数据（用于跨场景评估）。

#     Returns:
#         samples: list of dict {env, link, path}
#     """
#     if data_dir is None:
#         data_dir = os.path.join(PROJECT_ROOT, 'data',
#                                 'generated_scenario_2_6_0_100_2_16_32_2_4_1_18_52')
#     samples = scan_dataset(data_dir, max_samples=max_samples, scenario=2)
#     return samples
def load_scenario2(data_dir=None, max_samples=None):
    if data_dir is None:
        from . import config as cfg
        data_dir = cfg.S2_DATA_DIR
    # 生成器已统一为 env/array/*.npy 结构，扫描逻辑与 S1 相同
    samples = scan_dataset(data_dir, max_samples=max_samples, scenario=1)
    return samples


def get_dataset(samples: List[dict],
                task: str = 'joint',
                mask_mode: str = 'comb',
                mask_ratio: float = 0.5,
                random_mask: bool = False,
                cache_in_memory: bool = False,
                max_samples: int = None) -> CSIPredictionDataset:
    """
    构建 CSIPredictionDataset。

    数据处理流程（每个 sample）:
        .npy (2048, 52) complex64
            -> complex_from_npy
            -> reshape (256, 8, 52) = (Nt_port, Nr, Nf)
            -> apply_mask -> H_known_3d + H_target_3d
            -> stack real+imag -> (2, 256, 8, 52)

    Returns:
        CSIPredictionDataset 实例
    """
    # 截断到 max_samples
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    ds = CSIPredictionDataset(
        samples=samples,
        task=task,
        mask_mode=mask_mode,
        mask_ratio=mask_ratio,
        random_mask=random_mask,
        cache_in_memory=cache_in_memory,
    )
    return ds


# ----------------------------------------------------------------------
# 批处理推理数据准备
# ----------------------------------------------------------------------

def prepare_batch(sample: dict,
                  task: str = 'joint',
                  mask_mode: str = 'comb',
                  mask_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从单个 sample dict 准备模型输入（H_known, H_target, mask）。

    用于传统算法评估（不需要 DataLoader）。

    Returns:
        H_known: (2, Nt_port, Nr, Nf) float32
        H_target: (2, Nt_port, Nr, Nf) float32
        mask: (Nt_port, Nf) bool（joint）或对应维度的 bool
    """
    # 加载 .npy
    npy_data = np.load(sample['path']).astype(np.complex64)
    assert npy_data.shape == NPY_SHAPE, f'{npy_data.shape} vs {NPY_SHAPE}'

    # reshape -> (Nt_port, Nr, Nf)
    H_complex = npy_data.reshape(NT_PORT, NR, NF)

    # 生成 mask
    from .data_processing.mask_strategies import get_mask_strategy
    if task == 'frequency':
        mask_shape = (NF,)
    elif task == 'spatial':
        mask_shape = (NT_PORT,)
    elif task == 'joint':
        mask_shape = (NT_PORT, NF)
    else:
        raise ValueError(f'Unknown task: {task}')

    strategy = get_mask_strategy(task, mask_mode)
    # hash() 可能返回负数，用 abs() 保证非负
    seed = abs(hash(str(sample.get('env', sample.get('path', '')))))
    mask = strategy.generate_mask(mask_shape, mask_ratio, seed=seed)

    # apply mask
    H_known_3d, H_target_3d = apply_mask(H_complex, mask, task)

    # real+imag
    H_known = np.stack([H_known_3d.real, H_known_3d.imag], axis=0).astype(np.float32)
    H_target = np.stack([H_target_3d.real, H_target_3d.imag], axis=0).astype(np.float32)

    return H_known, H_target, mask


def add_noise(H: np.ndarray, snr_db: float) -> np.ndarray:
    """
    向 CSI 信号添加 AWGN 噪声。

    Args:
        H: (2, Nt, Nr, Nf) 实数数组（已经过 real+imag 分离）
        snr_db: 信噪比（dB）
    Returns:
        带噪版本，同形状
    """
    # 转复数
    H_complex = H[0] + 1j * H[1]
    power_signal = np.mean(np.abs(H_complex) ** 2)
    power_noise = power_signal / (10 ** (snr_db / 10))
    noise = np.sqrt(power_noise / 2) * (
        np.random.randn(*H_complex.shape) +
        1j * np.random.randn(*H_complex.shape)
    )
    H_noisy = H_complex + noise
    # 转回实虚分离格式
    return np.stack([H_noisy.real, H_noisy.imag], axis=0).astype(np.float32)


# ----------------------------------------------------------------------
# 数据集信息
# ----------------------------------------------------------------------

def get_data_info(data_dir: str = None) -> dict:
    """返回数据集基本信息（样本数、目录等）"""
    from csi_pre_eval import config as cfg
    base_dir = data_dir or cfg.DATA_DIR
    s1_dir = cfg._resolve_scenario_dir(base_dir, '1')
    s2_dir = cfg._resolve_scenario_dir(base_dir, '2')

    def _count(d):
        if os.path.exists(d):
            return len(scan_dataset(d, max_samples=None, scenario=1))
        return 0

    return {
        's1_total': _count(s1_dir),
        's2_total': _count(s2_dir),
        's1_dir': s1_dir,
        's2_dir': s2_dir,
        'shape_3d': (NT_PORT, NR, NF),
        'npy_shape': NPY_SHAPE,
    }

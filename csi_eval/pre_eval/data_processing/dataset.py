"""
CSI 空频域预测 Dataset
=====================

数据格式 (统一):
- 存储: (2048, 52) complex64 .npy 文件 (来自生成器)
- 内存: (256, 8, 52) 复数
- 模型输入/输出: (2, 256, 8, 52) 实部+虚部

数据形状 (EVM 1驱4平均后):
- H_3d: (Nt_port=256, Nr=8, Nf=52) 复数
- 标准归一化: 已除以 sqrt(mean(|H|^2))
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from ..params.params_7GHz import (
    GENERATED_FOLDER, NT_PORT, NR, NF, NPY_SHAPE,
)
from .preprocessing import complex_from_npy
from .mask_strategies import (
    get_mask_strategy, apply_mask,
)


# def scan_dataset(root_dir, max_samples=None, scenario=1):
#     """扫描所有 .npy 样本

#     Args:
#         root_dir: 数据集根目录
#         max_samples: 限制最多扫描多少样本 (减少 NFS RPC, 加快启动)
#         scenario: 1 = Scenario 1 目录结构 (root_dir/00001/array/xxx.npy)
#                   2 = Scenario 2 目录结构 (root_dir/repeat_N/00032/array/xxx.npy)
#                   None = 自动探测 (有 repeat_N 子目录则按 s2, 否则 s1)

#     Returns:
#         samples: list of dict {env, link, path, repeat}
#     """
#     import os
#     samples = []

#     # 自动探测
#     if scenario is None:
#         if any(d.startswith('repeat_') for d in os.listdir(root_dir)
#                if os.path.isdir(os.path.join(root_dir, d))):
#             scenario = 2
#         else:
#             scenario = 1

#     if scenario == 1:
#         # Scenario 1: root_dir/00001/array/xxx.npy
#         envs = sorted([d for d in os.listdir(root_dir)
#                        if os.path.isdir(os.path.join(root_dir, d))],
#                        key=lambda x: int(x) if x.isdigit() else float('inf'))
#         for env_dir in envs:
#             array_dir = os.path.join(root_dir, env_dir, 'array')
#             if not os.path.isdir(array_dir):
#                 continue
#             with os.scandir(array_dir) as it:
#                 for entry in it:
#                     if not entry.name.endswith('.npy'):
#                         continue
#                     link_str = entry.name[:-4]
#                     samples.append({
#                         'env': env_dir,
#                         'link': link_str,
#                         'path': os.path.join(array_dir, entry.name),
#                         'repeat': 0,
#                     })
#                     if max_samples is not None and len(samples) >= max_samples:
#                         return samples

#     elif scenario == 2:
#         # Scenario 2: root_dir/repeat_N/00032/array/xxx.npy
#         repeats = sorted([d for d in os.listdir(root_dir)
#                           if os.path.isdir(os.path.join(root_dir, d))
#                           and d.startswith('repeat_')],
#                          key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
#         for repeat_dir in repeats:
#             repeat_path = os.path.join(root_dir, repeat_dir)
#             repeat_num = int(repeat_dir.split('_')[1])
#             envs = sorted([d for d in os.listdir(repeat_path)
#                            if os.path.isdir(os.path.join(repeat_path, d))],
#                           key=lambda x: int(x) if x.isdigit() else float('inf'))
#             for env_dir in envs:
#                 array_dir = os.path.join(repeat_path, env_dir, 'array')
#                 if not os.path.isdir(array_dir):
#                     continue
#                 with os.scandir(array_dir) as it:
#                     for entry in it:
#                         if not entry.name.endswith('.npy'):
#                             continue
#                         link_str = entry.name[:-4]
#                         samples.append({
#                             'env': env_dir,
#                             'link': link_str,
#                             'path': os.path.join(array_dir, entry.name),
#                             'repeat': repeat_num,
#                         })
#                         if max_samples is not None and len(samples) >= max_samples:
#                             return samples

#     return samples
def scan_dataset(root_dir, max_samples=None, scenario=1):
    """扫描数据集目录，按 env/array/*.npy 收集样本"""
    samples = []
    if not os.path.isdir(root_dir):
        return samples

    envs = sorted([d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))])
    for env in envs:
        env_path = os.path.join(root_dir, env, 'array')
        if not os.path.isdir(env_path):
            continue
        files = sorted([f for f in os.listdir(env_path) if f.endswith('.npy')])
        for f in files:
            samples.append({
                'env': env,
                'file': f,
                'path': os.path.join(env_path, f),
            })
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples

def split_dataset(samples, train_envs=1000, scenario=1):
    """按 env 划分 train/val/test

    Scenario 1: 前 train_envs 个地图作为训练集，剩下的作为泛化测试
    Scenario 2: 全部作为泛化测试集（不做 8:1:1 划分）

    训练: 前 train_envs 个地图
    验证: 训练地图里随机 10%
    测试: 训练地图里随机 10% (与验证不重叠)
    """
    train_set, val_set, test_set = [], [], []
    train_envs_set = set(str(i).zfill(5) for i in range(1, train_envs + 1))

    if scenario == 2:
        # Scenario 2: 全部作为泛化测试集 (Scenario 2 本身就是 holdout)
        # 训练集为空，用户在 train 时用 s1 训练，s2 评测泛化
        return [], [], list(samples)

    # Scenario 1: 训练地图内的样本
    train_samples = [s for s in samples if s['env'] in train_envs_set]

    # 简单划分: 按样本索引 8:1:1
    n = len(train_samples)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(n // 10, 1)
    n_test = max(n // 10, 1)
    val_idx = set(perm[:n_val])
    test_idx = set(perm[n_val:n_val + n_test])

    for i, s in enumerate(train_samples):
        if i in val_idx:
            val_set.append(s)
        elif i in test_idx:
            test_set.append(s)
        else:
            train_set.append(s)

    return train_set, val_set, test_set


class CSIPredictionDataset(Dataset):
    """CSI 空频域预测数据集

    数据流程:
        .npy (2048, 52) complex64
        -> complex_from_npy -> (2048, 52) 复数
        -> reshape -> (256, 8, 52) 复数
        -> apply_mask -> (256, 8, 52) 已知 + (256, 8, 52) 目标
        -> to_real_imag -> (2, 256, 8, 52) 模型输入/目标

    Args:
        samples: list of sample dict (from scan_dataset)
        task: 'frequency' / 'spatial' / 'joint'
        mask_mode: 掩码模式 (如 'comb', 'random', 'grid' 等)
        mask_ratio: 掩码保留比例 (如 0.5)
        random_mask: 训练时是否每轮随机生成掩码
        cache_in_memory: 是否一次性加载到内存 (小数据集可开启)
    """
    def __init__(self, samples, task='frequency', mask_mode='comb',
                 mask_ratio=0.5, random_mask=True, cache_in_memory=False):
        self.samples = samples
        self.task = task
        self.mask_ratio = mask_ratio
        self.random_mask = random_mask
        self.mask_strategy = get_mask_strategy(task, mask_mode)
        self.cache_in_memory = cache_in_memory

        if cache_in_memory:
            self._cache = {}
            for i, s in enumerate(samples):
                self._cache[i] = np.load(s['path']).astype(np.complex64)

    def __len__(self):
        return len(self.samples)

    def _get_npy(self, idx):
        if self.cache_in_memory:
            return self._cache[idx]
        # .npy 存的就是 complex64，直接读进来就是复数数组
        return np.load(self.samples[idx]['path']).astype(np.complex64)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # .npy: (2048, 52) complex64 -> 直接就是复数
        npy_data = self._get_npy(idx)
        assert npy_data.shape == NPY_SHAPE, \
            f'shape mismatch: {npy_data.shape} vs {NPY_SHAPE}'

        # reshape 到 3D: (2048, 52) -> (256, 8, 52)
        H_complex = complex_from_npy(npy_data)         # (2048, 52) 复数
        H_3d = H_complex.reshape(NT_PORT, NR, NF)       # (256, 8, 52) 复数

        # 生成掩码
        if self.task == 'frequency':
            mask_shape = (NF,)
        elif self.task == 'spatial':
            mask_shape = (NT_PORT,)
        elif self.task == 'joint':
            mask_shape = (NT_PORT, NF)
        else:
            raise ValueError(self.task)

        mask = self.mask_strategy.generate_mask(
            mask_shape, self.mask_ratio, seed=None if self.random_mask else idx
        )

        # 应用掩码: 复数操作
        H_known_3d, H_target_3d = apply_mask(H_3d, mask, self.task)

        # 复数 -> (2, 256, 8, 52) 实部+虚部 (模型输入格式)
        H_known = np.stack(
            [H_known_3d.real, H_known_3d.imag], axis=0
        ).astype(np.float32)
        H_target = np.stack(
            [H_target_3d.real, H_target_3d.imag], axis=0
        ).astype(np.float32)
        mask_t = torch.from_numpy(mask.astype(np.float32))

        return {
            'H_known': torch.from_numpy(H_known),     # (2, 256, 8, 52)
            'H_target': torch.from_numpy(H_target),   # (2, 256, 8, 52)
            'mask': mask_t,                           # 掩码
            'task': self.task,
            'env': sample['env'],
            'link': sample['link'],
        }


class ThreeChannelDataset(Dataset):
    """支持三种任务混合训练的数据集

    每次随机选择一种任务及其对应的掩码。
    """
    def __init__(self, samples, mask_ratios=(0.5,), tasks=('frequency',),
                 task_weights=None, cache_in_memory=False):
        self.samples = samples
        self.mask_ratios = mask_ratios
        self.tasks = tasks
        self.task_weights = task_weights or [1.0 / len(tasks)] * len(tasks)
        self.cache_in_memory = cache_in_memory

        self._datasets = {}
        for task in tasks:
            from .mask_strategies import (
                FrequencyCombMask, FrequencyRandomMask,
                SpatialCombMask, SpatialSubsetMask,
                JointGridMask, JointRandomMask,
            )
            task_modes = {
                'frequency': [('comb', FrequencyCombMask()),
                              ('random', FrequencyRandomMask())],
                'spatial':   [('comb', SpatialCombMask()),
                              ('random_subset', SpatialSubsetMask())],
                'joint':     [('grid', JointGridMask()),
                              ('random_2d', JointRandomMask())],
            }
            self._datasets[task] = [
                (mode, strategy, ratio)
                for mode, strategy in task_modes[task]
                for ratio in mask_ratios
            ]

        if cache_in_memory:
            self._cache = {}
            for i, s in enumerate(samples):
                self._cache[i] = np.load(s['path']).astype(np.complex64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.cache_in_memory:
            npy_data = self._cache[idx]
        else:
            # .npy 存的就是 complex64
            npy_data = np.load(sample['path']).astype(np.complex64)

        # 随机选任务
        task_idx = np.random.choice(len(self.tasks), p=self.task_weights)
        task = self.tasks[task_idx]
        # 随机选该任务下的 (mode, ratio)
        cfg = self._datasets[task][np.random.randint(len(self._datasets[task]))]
        mode, strategy, ratio = cfg

        # .npy -> 3D 复数
        H_complex = complex_from_npy(npy_data)           # (2048, 52)
        H_3d = H_complex.reshape(NT_PORT, NR, NF)         # (256, 8, 52)

        if task == 'frequency':
            mask_shape = (NF,)
        elif task == 'spatial':
            mask_shape = (NT_PORT,)
        else:
            mask_shape = (NT_PORT, NF)

        mask = strategy.generate_mask(mask_shape, ratio, seed=None)
        H_known_3d, H_target_3d = apply_mask(H_3d, mask, task)

        H_known = np.stack([H_known_3d.real, H_known_3d.imag], axis=0).astype(np.float32)
        H_target = np.stack([H_target_3d.real, H_target_3d.imag], axis=0).astype(np.float32)
        mask_t = torch.from_numpy(mask.astype(np.float32))

        return {
            'H_known': torch.from_numpy(H_known),
            'H_target': torch.from_numpy(H_target),
            'mask': mask_t,
            'task': task,
            'mode': mode,
            'ratio': ratio,
            'env': sample['env'],
            'link': sample['link'],
        }

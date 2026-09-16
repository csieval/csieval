"""
掩码策略
========

三类预测任务共享同一 3D 张量 (Nt_port, Nr, Nf):
- 频域预测: 沿 Nf 维度掩码
- 空域预测: 沿 Nt_port 维度掩码
- 联合预测: 对 (Nt_port, Nf) 平面做 2D 掩码

统一语义（重要）：
    mask = True  → 待预测位置（模型需要输出的区域）
    mask = False → 已知位置（输入中保留真实值的区域）

    mask_ratio = 待预测比例（每维独立）：
        - 频率预测: ratio=0.25 表示 25% 频点待预测，75% 已知
        - 空间预测: ratio=0.25 表示 25% 端口待预测，75% 已知
        - 联合预测: ratio=0.25 表示两维各自 25% 待预测，
                   总已知 cell 比例 = (1-ratio)²，待预测 = 1-(1-ratio)²

依赖: (Nt_port, Nr, Nf) = (256, 8, 52)
"""

import numpy as np
from abc import ABC, abstractmethod


class MaskStrategy(ABC):
    """掩码策略基类"""
    @abstractmethod
    def generate_mask(self, shape, ratio, seed=None):
        """生成掩码

        Args:
            shape: 各任务对应维度
                - 频域: (Nf,)
                - 空域: (Nt_port,)
                - 联合: (Nt_port, Nf)
            ratio: 待预测比例（0~1）
            seed: 随机种子
        Returns:
            mask: bool array, True=待预测, False=已知
        """
        pass


# ============== 频域预测 (Frequency-only) ==============

class FrequencyCombMask(MaskStrategy):
    """频域均匀间隔掩码: 沿 Nf 维度均匀选取已知位置。

    语义: mask_ratio = 待预测比例（被 mask 的子载波占比）。
          已知比例 = 1 - mask_ratio。
    实现: n_keep = round(Nf * (1 - ratio))，用 np.linspace 均匀分布已知位置。
    """
    def generate_mask(self, shape, ratio, seed=None):
        assert len(shape) == 1, f'1D shape expected, got {shape}'
        nf = shape[0]
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio))
        n_keep = max(1, min(nf, int(round(nf * keep_ratio))))
        mask = np.ones(nf, dtype=bool)  # 默认全 True（待预测）
        if n_keep >= 1:
            idx = np.linspace(0, nf - 1, n_keep, dtype=int)
            mask[idx] = False             # 均匀间隔位置为已知 (False)
        return mask


class FrequencyBlockMask(MaskStrategy):
    """频域块状掩码: 前面部分已知，后面待预测"""
    def generate_mask(self, shape, ratio, seed=None):
        nf = shape[0]
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio))
        n_keep = max(0, min(nf, int(round(nf * keep_ratio))))
        mask = np.ones(nf, dtype=bool)
        mask[:n_keep] = False
        return mask


class FrequencyRandomMask(MaskStrategy):
    """频域随机掩码: 随机选取已知位置"""
    def generate_mask(self, shape, ratio, seed=None):
        nf = shape[0]
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio))
        n_keep = max(0, min(nf, int(round(nf * keep_ratio))))
        rng = np.random.default_rng(seed)
        idx = rng.choice(nf, size=n_keep, replace=False)
        mask = np.ones(nf, dtype=bool)
        mask[idx] = False
        return mask


# ============== 空域预测 (Spatial-only) ==============

class SpatialCombMask(MaskStrategy):
    """空域均匀间隔掩码。

    语义: mask_ratio = 待预测比例（被 mask 的端口占比）。
          已知比例 = 1 - mask_ratio。
    实现: n_keep = round(Nt_port * (1 - ratio))，用 np.linspace 均匀分布。
    """
    def generate_mask(self, shape, ratio, seed=None):
        nt = shape[0]
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio))
        n_keep = max(1, min(nt, int(round(nt * keep_ratio))))
        mask = np.ones(nt, dtype=bool)  # 默认全 True（待预测）
        if n_keep >= 1:
            idx = np.linspace(0, nt - 1, n_keep, dtype=int)
            mask[idx] = False             # 均匀间隔位置为已知 (False)
        return mask


class SpatialSubsetMask(MaskStrategy):
    """空域随机子集掩码"""
    def generate_mask(self, shape, ratio, seed=None):
        nt = shape[0]
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio))
        n_keep = max(0, min(nt, int(round(nt * keep_ratio))))
        rng = np.random.default_rng(seed)
        idx = rng.choice(nt, size=n_keep, replace=False)
        mask = np.ones(nt, dtype=bool)
        mask[idx] = False
        return mask


# ============== 联合预测 (Joint Spatial-Frequency) ==============

class JointGridMask(MaskStrategy):
    """联合 2D 规则网格掩码: 两维各自均匀间隔选取已知位置。

    语义: mask_ratio = 待预测比例（每维独立）。
    实现: n_keep_t = round(Nt_port * (1 - ratio)), n_keep_f = round(Nf * (1 - ratio))
          已知 cells = n_keep_t × n_keep_f（笛卡尔积）
    """
    def generate_mask(self, shape, ratio, seed=None):
        """联合 2D 网格掩码：保留**整行整列**，形成 grid（十字网格）形状。

        与 JointCombMask 的区别：grid 保留整行/列（mask=False 沿整轴扩展），
        comb 仅在笛卡尔积点（稀疏点阵）。
        """
        nt, nf = shape
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio))
        n_keep_t = max(1, min(nt, int(round(nt * keep_ratio))))
        n_keep_f = max(1, min(nf, int(round(nf * keep_ratio))))
        mask = np.ones((nt, nf), dtype=bool)  # 默认全 True（待预测）
        idx_t = np.linspace(0, nt - 1, n_keep_t, dtype=int)
        idx_f = np.linspace(0, nf - 1, n_keep_f, dtype=int)
        # grid: 保留**整行 + 整列**（mask=False），形成十字网格
        mask[idx_t, :] = False   # 选中行整行已知
        mask[:, idx_f] = False   # 选中列整列已知
        return mask


class JointRandomMask(MaskStrategy):
    """联合 2D 随机掩码: 在 (Nt_port, Nf) 平面随机选已知位置

    语义: mask_ratio = 待预测比例。
          已知比例 = 1 - mask_ratio。
    """
    def generate_mask(self, shape, ratio, seed=None):
        nt, nf = shape
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio))
        n_keep = max(0, min(nt * nf, int(round(nt * nf * keep_ratio))))
        rng = np.random.default_rng(seed)
        flat_idx = rng.choice(nt * nf, size=n_keep, replace=False)
        mask = np.ones((nt, nf), dtype=bool)
        mask.flat[flat_idx] = False
        return mask


class JointCombMask(MaskStrategy):

    """联合 2D 梳状掩码: 在 (Nt_port, Nf) 平面**两维各自独立**做均匀间隔采样。

    语义说明 (joint+comb):
        mask_ratio = 待预测比例（每维独立）。
        已知比例 = 1 - mask_ratio（每维独立）。

    实现：
        - 空域: n_keep_t = round(Nt_port * (1 - ratio))，均匀分布
        - 频域: n_keep_f = round(Nf * (1 - ratio))，均匀分布
        - 已知 cells = n_keep_t × n_keep_f（笛卡尔积）

    与 spatial+comb / frequency+comb 的关键区别：
        - spatial+comb    : 频域全保留（所有频点），只 mask 空域
        - frequency+comb : 空域全保留（所有端口），只 mask 频域
        - joint+comb     : **两维各自都 mask**，已知 cell 比例 = (1-ratio)²
                          → 待预测比例 = 1 - (1-ratio)²
                          → joint 任务难度最大

    例如 (Nt=256, Nf=52)：
        mask_ratio=0.25 → 已知比例=75% → n_keep_t=192, n_keep_f=39
                          → 已知 cells = 192×39=7488 (56.25%), 待预测=43.75%
        mask_ratio=0.5  → 已知比例=50% → n_keep_t=128, n_keep_f=26
                          → 已知 cells = 128×26=3328 (25%), 待预测=75%
        mask_ratio=0.75 → 已知比例=25% → n_keep_t=64, n_keep_f=13
                          → 已知 cells = 64×13=832 (6.25%), 待预测=93.75%
    """
    def generate_mask(self, shape, ratio, seed=None):
        nt, nf = shape
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio))
        n_keep_t = max(1, min(nt, int(round(nt * keep_ratio))))
        n_keep_f = max(1, min(nf, int(round(nf * keep_ratio))))

        mask = np.ones((nt, nf), dtype=bool)  # 默认全 True（待预测）
        idx_t = np.linspace(0, nt - 1, n_keep_t, dtype=int)
        idx_f = np.linspace(0, nf - 1, n_keep_f, dtype=int)
        # **与 Grid 区分**：comb 在两维上**各自**做"梳状"采样，
        # 用相位偏移 (n_keep_t-1)/2 让已知位不在两端对齐，避免与 grid 同形。
        if n_keep_t > 1:
            phase_t = (nt / n_keep_t) / 2
            idx_t = np.round(np.arange(n_keep_t) * (nt / n_keep_t) + phase_t).astype(int)
            idx_t = np.clip(idx_t, 0, nt - 1)
            idx_t = np.unique(idx_t)
        if n_keep_f > 1:
            phase_f = (nf / n_keep_f) / 2
            idx_f = np.round(np.arange(n_keep_f) * (nf / n_keep_f) + phase_f).astype(int)
            idx_f = np.clip(idx_f, 0, nf - 1)
            idx_f = np.unique(idx_f)
        mask[np.ix_(idx_t, idx_f)] = False   # 笛卡尔积为已知 (False)
        return mask


# ============== 工厂函数 ==============

def get_mask_strategy(task, mode):
    """根据任务+模式获取掩码策略"""
    registry = {
        ('frequency', 'comb'): FrequencyCombMask,
        ('frequency', 'block'): FrequencyBlockMask,
        ('frequency', 'random'): FrequencyRandomMask,
        ('spatial', 'comb'): SpatialCombMask,
        ('spatial', 'random_subset'): SpatialSubsetMask,
        ('joint', 'grid'): JointGridMask,
        ('joint', 'random_2d'): JointRandomMask,
        ('joint', 'comb'): JointCombMask,
    }
    key = (task, mode)
    if key not in registry:
        raise ValueError(f'Unknown task/mode: {key}. '
                         f'Available: {list(registry.keys())}')
    return registry[key]()


def apply_mask(H_3d, mask, task):
    """将掩码应用到 3D 张量 (Nt_port, Nr, Nf)

    语义: mask=True → 待预测（置零）；mask=False → 已知（保留）

    Returns:
        H_known: 已知部分（mask=True 位置置零）
        H_target: 待预测完整真实值
    """
    if task == 'frequency':
        # mask: (Nf,), True=待预测
        H_known = H_3d.copy()
        H_known[:, :, mask] = 0
        return H_known, H_3d
    elif task == 'spatial':
        # mask: (Nt_port,), True=待预测
        H_known = H_3d.copy()
        H_known[mask, :, :] = 0
        return H_known, H_3d
    elif task == 'joint':
        # mask: (Nt_port, Nf), True=待预测
        H_known = H_3d.copy()
        H_known = H_known * (~mask)[:, None, :]  # mask=False 位置保留
        return H_known, H_3d
    else:
        raise ValueError(f'Unknown task: {task}')


def extract_pred_region(H_pred, H_true, mask, task):
    """提取预测区域(待预测位置)的预测值与真实值"""
    if task == 'frequency':
        return H_pred[:, :, mask], H_true[:, :, mask]
    elif task == 'spatial':
        return H_pred[mask, :, :], H_true[mask, :, :]
    elif task == 'joint':
        return H_pred[:, :, mask], H_true[:, :, mask]
    else:
        raise ValueError(f'Unknown task: {task}')

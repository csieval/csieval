"""
统一评估配置
============
所有评估相关的超参数、路径、默认值集中在此文件。
"""
import os
from dataclasses import dataclass, field
from typing import List

# ---------- 路径 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'pre_eval')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'results', 'checkpoints')

# ---------- 数据路径自动探测 ----------
# data_pre / data 下直接就是 generated_scenario_* 子目录（勿重复嵌套）。
# _resolve_data_dir 找顶层数据目录；_resolve_scenario_dir 在顶层目录内找对应场景子目录。
_DATA_DIR_CANDIDATES = ('data_pre', 'data')

def _resolve_data_dir():
    """返回真实存在的数据根目录；都不存在时返回 'data'（保底）。"""
    for candidate in _DATA_DIR_CANDIDATES:
        p = os.path.join(PROJECT_ROOT, candidate)
        if os.path.isdir(p):
            return p
    return os.path.join(PROJECT_ROOT, 'data')

def _resolve_scenario_dir(base_dir, scenario_marker):
    """在 base_dir 内查找名为 generated_scenario_<marker>_* 的子目录并返回。

    如果 base_dir 本身就是场景目录（即已包含 generated_scenario_*），
    直接返回 base_dir。这保证了两层目录结构（data_pre/scenario_*）和
    扁层结构（data_pre/*）都能正常工作。
    """
    if base_dir and f'generated_scenario_{scenario_marker}' in os.path.basename(base_dir):
        return base_dir  # base_dir 已是场景目录，无需再探
    if not os.path.isdir(base_dir):
        return base_dir
    marker_prefix = f'generated_scenario_{scenario_marker}'
    try:
        candidates = os.listdir(base_dir)
    except OSError:
        return base_dir
    for name in candidates:
        if name.startswith(marker_prefix):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full):
                return full
    return base_dir  # 找不到时回退原值

DATA_DIR = _resolve_data_dir()

# 场景目录：自动在 DATA_DIR 下找对应 generated_scenario_* 子目录
S1_DATA_DIR = _resolve_scenario_dir(DATA_DIR, '1')
S2_DATA_DIR = _resolve_scenario_dir(DATA_DIR, '2')

# RESULTS_DIR 由实际使用的子评估包在需要时创建，此处不提前 mkdir
# （避免 import 时就在 results/ 下创建空目录）
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'pre_eval')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'results', 'checkpoints')

# ---------- 硬件 ----------
DEFAULT_DEVICE = 'cuda'

# ---------- 评估默认设置 ----------
EVAL_DEFAULT = {
    'task': 'joint',
    'mask_mode': 'comb',
    'mask_ratio': 0.5,
    'max_samples': 15000,   # 评估样本数
    'batch_size': 256,
    'device': DEFAULT_DEVICE,
    'seed': 42,
    'num_workers': 4,
}

# ---------- 任务 & 掩码 ----------
TASK = 'joint'
MASK_MODES = ['grid', 'random_2d', 'comb']  # 联合预测只保留这三种模式
MASK_RATIOS = [0.5, 0.25, 0.125, 0.0625]    # ratio 评估范围
SNR_LEVELS = [5, 10, 15, 20, 25, 30]        # 噪声鲁棒性 SNR (dB)

# ---------- Task Performance 9 指标网格 ----------
# 任务 → 默认 mask 模式（spatial/frequency/joint 各匹配一种 mask strategy）
TASK_GRID_TASKS = ('joint', 'spatial', 'frequency')
TASK_GRID_MASK_MODES = {
    'joint':     'comb',         # 联合：沿 Nt_port 梳状采样，频域全保留
    'spatial':   'comb',         # 空域：沿 Nt_port 梳状
    'frequency': 'comb',         # 频域：沿 Nf 梳状
}
TASK_GRID_MASK_RATIOS = (0.25, 0.5, 0.75)  # 与 README 中的规格对齐

# ---------- 参数量化位宽 ----------
QUANT_BITS = [32, 16, 8]                     # FP32 / FP16 / INT8

# ---------- 泛化鲁棒性配置 ----------
CROSS_MASK_PAIRS = [
    ('comb', 'grid'),
    ('comb', 'random_2d'),
]

CROSS_RATIO_DEGRADATION = [
    (0.75, 0.5),
    (0.5,  0.25),
    (0.25, 0.125),
]

# ---------- 跨场景 ----------
S1_TO_S2_MAX_SAMPLES = 15000

# ---------- 模型列表 ----------
# 神经网络模型（需要训练）
DL_MODELS = ['csinet', 'crnet', 'wifo', 'csi_conformer', 'stcmixer']
# 传统算法（无需训练）
TRAD_MODELS = ['linear', 'cubic', 'nearest', 'fft_extrapolation']
ALL_MODELS = DL_MODELS + TRAD_MODELS

# ---------- 指标名称 ----------
METRIC_NMSE = 'nmse'
METRIC_SGCS = 'sgcs'
METRIC_EVM  = 'evm'

# ---------- 数据维度 (configs/params_7GHz.py) ----------
NT_PORT = 256
NR      = 8
NF      = 52
INPUT_SHAPE = (2, NT_PORT, NR, NF)   # (real, imag, Nt_port, Nr, Nf) — 注意和模型保持一致


@dataclass
class EvalConfig:
    """评估配置容器"""
    task: str = TASK
    mask_mode: str = 'comb'
    mask_ratio: float = 0.5
    max_samples: int = 15000
    batch_size: int = 256
    device: str = DEFAULT_DEVICE
    s1_data_dir: str = ''
    s2_data_dir: str = ''
    seed: int = 42
    num_workers: int = 4

    # 结果输出
    output_dir: str = RESULTS_DIR

    # 哪些模块要跑
    run_task_metrics: bool = True
    run_storage_metrics: bool = True
    run_compute_metrics: bool = True
    run_robustness: bool = True
    run_noise_robustness: bool = True
    run_cross_scenario: bool = True

    # 鲁棒性
    cross_mask_pairs: List[tuple] = field(default_factory=lambda: CROSS_MASK_PAIRS)
    cross_ratio_degradation: List[tuple] = field(default_factory=lambda: CROSS_RATIO_DEGRADATION)
    snr_levels: List[int] = field(default_factory=lambda: SNR_LEVELS)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
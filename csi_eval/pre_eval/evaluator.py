"""
主评估编排器
============
按用户定义的指标体系，对指定模型进行全量评估，并统一输出结果。

评估流程:
    1. 数据加载（S1 测试集 / S2 全量）
    2. 模型构建 + 权重加载
    3. 任务性能指标
    4. 存储与部署指标
    5. 计算效率指标
    6. 泛化鲁棒性（Cross-mask, Cross-ratio, SNR, 跨场景）
    7. 结果汇总输出（统一 EvalReport，可直接 report.save("json"|"html"|"markdown")）
"""
import os
import sys
import time
import datetime
import torch
import numpy as np
from typing import Dict, List, Optional

from csi_eval.progress import progress

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from . import config as cfg
from .model_registry import build_from_bundle
from .data_loader import load_scenario1, load_scenario2
from .metrics.task_metrics import compute_task_metrics, compute_per_mask_metrics
from .metrics.storage import compute_storage_metrics
from .metrics.compute import measure_latency, measure_flops, measure_peak_memory
from .metrics.robustness import compute_all_robustness

# 直接复用 feedback 包的 EvalReport + MetricRecord，保持接口完全统一
from csi_eval.feedback_eval.core.report import EvalReport, MetricRecord


# ----------------------------------------------------------------------
# Task Performance 9-指标网格（spatial / frequency / joint × 3 mask 比例）
# ----------------------------------------------------------------------
def _load_samples_to_3d(samples) -> np.ndarray:
    """一次性把所有 sample 的 .npy 加载为 (N, Nt_port, Nr, Nf) 复数 numpy 数组。

    用于 9 指标网格：避免每个 (task, ratio) 都重复做 np.load 的 IO。
    """
    from .data_loader import NPY_SHAPE, NT_PORT, NR, NF
    arrs = []
    for s in progress(samples, desc="prediction grid: load CSI", unit="sample", leave=False):
        d = np.load(s['path']).astype(np.complex64)
        arrs.append(d.reshape(NT_PORT, NR, NF))
    return np.stack(arrs, axis=0)        # (N, Nt_port, Nr, Nf) complex64


def _make_input_for(H_3d: np.ndarray, mask: np.ndarray, task: str) -> np.ndarray:
    """把 3D 复数张量 + mask 转成 (N, 2, Nt_port, Nr, Nf) 实部/虚部堆叠的输入。"""
    if task == 'frequency':
        # mask: (Nf,) → mask[None, None, :] broadcast
        keep = mask[None, None, :]
    elif task == 'spatial':
        # mask: (Nt_port,) → mask[:, None, None] broadcast
        keep = mask[:, None, None]
    elif task == 'joint':
        # mask: (Nt_port, Nf) → mask[:, None, :]
        keep = mask[:, None, :]
    else:
        raise ValueError(f'Unknown task: {task}')
    H_known_complex = H_3d * keep
    out = np.stack([H_known_complex.real, H_known_complex.imag], axis=1).astype(np.float32)
    return out


def compute_task_metric_grid(
    model,
    samples,
    device: str = 'cuda',
    max_samples: int = 1000,
    tasks: List[str] = None,
    mask_modes: List[str] = None,
    mask_ratios: List[float] = None,
    batch_size: int = None,
    sgcs_top_k: int = 4,
    sgcs_rb_per_subband: int = 4,
):
    """在 (task, mask_ratio) 的笛卡尔积上计算 NMSE / SGCS，组成 9 指标网格。

    默认: tasks = ('joint', 'spatial', 'frequency')，
          mask_ratios = (0.25, 0.5, 0.75)，
          mask_modes 由 ``get_mask_strategy`` 自动适配每个 (task, ratio)。

    性能优化：
      - 每个 sample 的 .npy 只读 1 次（而不是 9 次），复用同一份 ``H_3d`` numpy。
      - 大 batch 推理，一次把整 batch 放到 GPU 后顺序跑完所有 (task, ratio) 的 forward，
        让 GPU 始终处于计算密集状态。
      - SGCS 计算在 batch 级只搬一次 H_pred/H_true 到 CPU。
      - ``batch_size`` 默认取 ``cfg.EVAL_DEFAULT['batch_size']``（256），可按显存调大。

    Returns:
        dict:
          - ``metrics``: {f"{task}@{ratio}": {'nmse_db': float, 'sgcs': float,
                                              'task': str, 'mask_mode': str,
                                              'mask_ratio': float}}
            注：key 采用字符串以保证 JSON 可序列化。访问可用 ``metrics["joint@0.5"]``。
          - ``grid``:   list[dict]，与 ``metrics`` 等长，按行 (task) × 列 (ratio) 顺序排列。
          - ``tasks`` / ``mask_ratios``: 入参透传，便于 HTML 渲染表头。
          - ``n_samples``: 本次评估使用的样本数。
    """
    from .data_processing.mask_strategies import get_mask_strategy
    from .metrics.task_metrics import sgcs_3gpp_per_subband, sgcs_3gpp_per_subband_torch, _rb_to_subband_indices
    from .model_registry import infer as registry_infer

    if tasks is None:
        tasks = list(cfg.TASK_GRID_TASKS)
    if mask_ratios is None:
        mask_ratios = list(cfg.TASK_GRID_MASK_RATIOS)
    if mask_modes is None:
        mask_modes = dict(cfg.TASK_GRID_MASK_MODES)
    if batch_size is None:
        batch_size = int(cfg.EVAL_DEFAULT.get('batch_size', 128))

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    # ---------- 1. 一次性加载所有 H_3d（共享数据池） ----------
    n_total = len(samples)
    print(f'    [grid] 预加载 {n_total} 个 sample 的 .npy (一次性 IO)...')
    t_io = time.time()
    H_3d_all = _load_samples_to_3d(samples)        # (N, Nt_port, Nr, Nf) complex64
    print(f'    [grid] IO 完成 ({time.time()-t_io:.1f}s, 内存 {H_3d_all.nbytes / 1e9:.2f} GB)')

    # ---------- 2. 准备 9 个 (task, ratio) 的 mask ----------
    masks = {}
    for task in tasks:
        mode_for_task = mask_modes.get(task, 'comb')
        for ratio in mask_ratios:
            try:
                strat = get_mask_strategy(task, mode_for_task)
            except ValueError as e:
                print(f'    [skip] task={task}, mode={mode_for_task}: {e}')
                continue
            if task == 'frequency':
                shape = (cfg.NF,)
            elif task == 'spatial':
                shape = (cfg.NT_PORT,)
            else:  # joint
                shape = (cfg.NT_PORT, cfg.NF)
            seed = abs(hash(f"{task}|{mode_for_task}|{ratio}"))
            masks[(task, ratio)] = strat.generate_mask(shape, ratio, seed=seed)

    if isinstance(model, torch.nn.Module) or hasattr(model, 'eval'):
        try:
            model.eval()
        except Exception:
            pass

    # ---------- 3. 在 GPU 上分批跑 9 种 (task, ratio) ----------
    grid_metrics: Dict[str, Dict] = {}
    grid_rows = []

    # 与原 compute_task_metrics 一致的三态判断：
    #   is_wrapped: ModelWrapper（自动适配 1/2/3 参数签名）
    #   is_nn:      裸 torch.nn.Module
    #   is_trad:    既不是 ModelWrapper 也不是 nn.Module 的 callable（传统算法）
    from .model_registry import ModelWrapper
    is_wrapped = isinstance(model, ModelWrapper)
    is_nn      = isinstance(model, torch.nn.Module)
    is_trad    = not is_nn and not is_wrapped

    # 计算设备：传统算法只在 CPU 上跑
    use_device = 'cpu' if is_trad else device

    # 复数 -> GPU 复数的辅助
    to_complex = lambda t: torch.complex(t[:, 0], t[:, 1])

    with torch.no_grad():
        batch_starts = range(0, n_total, batch_size)
        for start in progress(
            batch_starts, total=(n_total + batch_size - 1) // batch_size,
            desc="prediction grid: 9 settings", unit="batch", leave=False
        ):
            end = min(start + batch_size, n_total)
            H_3d_b = H_3d_all[start:end]                       # (B, Nt_port, Nr, Nf) complex
            B = end - start

            # H_target 在 GPU 上常驻：每个 ratio 都用它做 NMSE/SGCS 真值
            H_target_t = torch.from_numpy(
                np.stack([H_3d_b.real, H_3d_b.imag], axis=1)
            ).to(use_device, non_blocking=True)

            # 9 个 (task, ratio) 顺序推理
            for task in tasks:
                for ratio in mask_ratios:
                    if (task, ratio) not in masks:
                        continue
                    mask = masks[(task, ratio)]
                    # 构造该 (task, ratio) 下的输入张量（H_known，shape (B, 2, Nt_port, Nr, Nf)）
                    H_known_np = _make_input_for(H_3d_b, mask, task)
                    H_known_t = torch.from_numpy(H_known_np).to(use_device, non_blocking=True)

                    if is_trad:
                        # 传统算法：单样本调用
                        H_pred_t = None
                        diff_sq_sum = 0.0
                        true_sq_sum = 0.0
                        per_sc_num_np = np.zeros(cfg.NF, dtype=np.float64)
                        per_sc_den_np = np.zeros(cfg.NF, dtype=np.float64)
                        per_port_num_np = np.zeros(cfg.NT_PORT, dtype=np.float64)
                        per_port_den_np = np.zeros(cfg.NT_PORT, dtype=np.float64)
                        sgcs_pred_list = []
                        sgcs_target_list = []
                        for i in range(B):
                            h_in = H_known_np[i]
                            h_target = H_3d_b[i]
                            pred = model(h_in[0] + 1j * h_in[1], mask, task)
                            pred_np = np.stack([pred.real, pred.imag], axis=0)
                            pc = pred_np[0] + 1j * pred_np[1]
                            tc = h_target
                            # 仅在 mask=True 位置评估
                            if task == 'frequency':
                                m3 = mask[None, None, :]
                            elif task == 'spatial':
                                m3 = mask[:, None, None]
                            else:
                                m3 = mask[:, None, :]
                            d2 = np.abs(pc - tc) ** 2 * m3
                            t2 = np.abs(tc) ** 2 * m3
                            diff_sq_sum += d2.sum()
                            true_sq_sum += t2.sum()
                            per_sc_num_np += d2.sum(axis=(0, 1))
                            per_sc_den_np += t2.sum(axis=(0, 1))
                            per_port_num_np += d2.sum(axis=(1, 2))
                            per_port_den_np += t2.sum(axis=(1, 2))
                            # SGCS: mask=True 保留模型预测，mask=False 用 target 回填。
                            # m3 可广播到 (Nt,Nr,Nf)，兼容 frequency/spatial/joint。
                            pc_eval = np.where(m3, pc, tc)
                            sgcs_pred_list.append(
                                np.stack([pc_eval.real, pc_eval.imag], axis=0).astype(np.float32)
                            )
                            sgcs_target_list.append(np.stack([h_target.real, h_target.imag], axis=0).astype(np.float32))
                        H_pred_for_sgcs = np.stack(sgcs_pred_list, axis=0)
                        H_target_for_sgcs = np.stack(sgcs_target_list, axis=0)
                        sgcs_arr = sgcs_3gpp_per_subband(
                            H_pred_for_sgcs, H_target_for_sgcs,
                            top_k=sgcs_top_k, rb_per_subband=sgcs_rb_per_subband,
                        )
                        key = f"{task}@{ratio}"
                        entry = grid_metrics.get(key)
                        if entry is None:
                            entry = {'task': task, 'mask_mode': mask_modes.get(task, 'comb'),
                                     'mask_ratio': ratio,
                                     'nmse_num': 0.0, 'nmse_den': 0.0,
                                     'sgcs_sum': 0.0, 'sgcs_count': 0,
                                     'sgcs_stream_sum': np.zeros(sgcs_top_k, dtype=np.float64),
                                     'sgcs_stream_count': 0,
                                     'n_samples': 0,
                                     'per_sc_num': np.zeros(cfg.NF, dtype=np.float64),
                                     'per_sc_den': np.zeros(cfg.NF, dtype=np.float64),
                                     'per_port_num': np.zeros(cfg.NT_PORT, dtype=np.float64),
                                     'per_port_den': np.zeros(cfg.NT_PORT, dtype=np.float64)}
                            grid_metrics[key] = entry
                        entry['nmse_num'] += diff_sq_sum
                        entry['nmse_den'] += true_sq_sum
                        entry['per_sc_num'] += per_sc_num_np
                        entry['per_sc_den'] += per_sc_den_np
                        entry['per_port_num'] += per_port_num_np
                        entry['per_port_den'] += per_port_den_np
                        if sgcs_arr.size:
                            entry['sgcs_sum'] += float(sgcs_arr.sum())
                            entry['sgcs_count'] += int(sgcs_arr.size)
                            entry['sgcs_stream_sum'] += sgcs_arr.sum(axis=0)
                            entry['sgcs_stream_count'] += int(sgcs_arr.shape[0])
                        entry['n_samples'] += B
                        continue

                    # GPU 推理
                    H_pred_t = registry_infer(model, H_known_t, mask, task, use_device)
                    pred_c = to_complex(H_pred_t)
                    true_c = to_complex(H_target_t)

                    # **关键修正**：NMSE / SGCS 只在 mask=True（待预测）位置计算。
                    # mask=False 是已知输入，模型 copy 该位置 H_true，
                    # 不应纳入预测质量评估（否则指标会被人为压低）。
                    if task == 'frequency':
                        mask_3d_t = torch.from_numpy(mask).reshape(1, 1, 1, -1).to(use_device)
                    elif task == 'spatial':
                        mask_3d_t = torch.from_numpy(mask).reshape(1, -1, 1, 1).to(use_device)
                    else:  # joint
                        mask_3d_t = torch.from_numpy(mask).reshape(
                            1, mask.shape[0], 1, mask.shape[1]
                        ).to(use_device)
                    mask_3d_f = mask_3d_t.to(pred_c.dtype)

                    diff_sq = (pred_c.real ** 2 + pred_c.imag ** 2) * mask_3d_f
                    true_sq = (true_c.real ** 2 + true_c.imag ** 2)  # 全值：分母不 mask

                    # SGCS stays on the model GPU: unknown locations are filled
                    # from the target exactly as in the previous NumPy path.
                    pred_eval_t = torch.where(mask_3d_t.unsqueeze(1), H_pred_t, H_target_t)
                    sgcs_arr_t = sgcs_3gpp_per_subband_torch(
                        pred_eval_t, H_target_t,
                        top_k=sgcs_top_k, rb_per_subband=sgcs_rb_per_subband,
                    )
                    sgcs_sum_t = sgcs_arr_t.sum()
                    sgcs_stream_sum_t = sgcs_arr_t.sum(dim=0)
                    sgcs_count = int(sgcs_arr_t.numel())
                    sgcs_stream_count = int(sgcs_arr_t.shape[0])

                    # Accumulate only scalar/vector reductions on the host.
                    nmse_num = float(diff_sq.real.sum().item())
                    nmse_den = float(true_sq.real.sum().item())
                    per_sc_num = diff_sq.real.sum(dim=(0, 1, 2)).cpu().numpy()
                    per_sc_den = true_sq.real.sum(dim=(0, 1, 2)).cpu().numpy()
                    per_port_num = diff_sq.real.sum(dim=(0, 2, 3)).cpu().numpy()
                    per_port_den = true_sq.real.sum(dim=(0, 2, 3)).cpu().numpy()

                    key = f"{task}@{ratio}"
                    entry = grid_metrics.get(key)
                    if entry is None:
                        entry = {'task': task, 'mask_mode': mask_modes.get(task, 'comb'),
                                 'mask_ratio': ratio,
                                 'nmse_num': 0.0, 'nmse_den': 0.0,
                                 'sgcs_sum': 0.0, 'sgcs_count': 0,
                                 'sgcs_stream_sum': np.zeros(sgcs_top_k, dtype=np.float64),
                                 'sgcs_stream_count': 0,
                                 'n_samples': 0,
                                 'per_sc_num': np.zeros(cfg.NF, dtype=np.float64),
                                 'per_sc_den': np.zeros(cfg.NF, dtype=np.float64),
                                 'per_port_num': np.zeros(cfg.NT_PORT, dtype=np.float64),
                                 'per_port_den': np.zeros(cfg.NT_PORT, dtype=np.float64)}
                        grid_metrics[key] = entry
                    entry['nmse_num'] += nmse_num
                    entry['nmse_den'] += nmse_den
                    if sgcs_count:
                        entry['sgcs_sum'] += float(sgcs_sum_t.item())
                        entry['sgcs_count'] += sgcs_count
                        entry['sgcs_stream_sum'] += sgcs_stream_sum_t.cpu().numpy()
                        entry['sgcs_stream_count'] += sgcs_stream_count
                    entry['n_samples'] += B
                    entry['per_sc_num'] += per_sc_num
                    entry['per_sc_den'] += per_sc_den
                    entry['per_port_num'] += per_port_num
                    entry['per_port_den'] += per_port_den

    # ---------- 4. 汇总每个 (task, ratio) 的最终 NMSE / SGCS ----------
    for key, entry in grid_metrics.items():
        if entry['nmse_den'] > 1e-10:
            entry['nmse_db'] = 10.0 * np.log10(entry['nmse_num'] / (entry['nmse_den'] + 1e-12))
        else:
            entry['nmse_db'] = float('-inf')
        if entry['sgcs_count'] > 0:
            entry['sgcs'] = entry['sgcs_sum'] / entry['sgcs_count']
        else:
            entry['sgcs'] = float('nan')
        if entry['sgcs_stream_count'] > 0:
            streams = (entry['sgcs_stream_sum'] / entry['sgcs_stream_count']).tolist()
        else:
            streams = [float('nan')] * sgcs_top_k
        entry['sgcs_streams'] = [float(x) for x in streams]
        entry['sgcs_stream_order'] = 'descending_covariance_eigenvalue'
        entry['sgcs_top_k'] = int(sgcs_top_k)
        entry['sgcs_rb_per_subband'] = int(sgcs_rb_per_subband)
        entry['sgcs_n_subbands'] = len(_rb_to_subband_indices(cfg.NF, sgcs_rb_per_subband))
        for i, value in enumerate(streams, start=1):
            entry[f'sgcs_stream_{i}'] = float(value)
        # per-sc / per-port 转 list
        sc_valid = entry['per_sc_den'] > 1e-10
        per_sc = np.full(cfg.NF, np.nan)
        if sc_valid.any():
            per_sc[sc_valid] = 10.0 * np.log10(entry['per_sc_num'][sc_valid] / entry['per_sc_den'][sc_valid] + 1e-12)
        pt_valid = entry['per_port_den'] > 1e-10
        per_pt = np.full(cfg.NT_PORT, np.nan)
        if pt_valid.any():
            per_pt[pt_valid] = 10.0 * np.log10(entry['per_port_num'][pt_valid] / entry['per_port_den'][pt_valid] + 1e-12)
        entry['per_subcarrier_nmse'] = per_sc.tolist()
        entry['per_port_nmse'] = per_pt.tolist()

        # 清理大数组以减小返回体积
        for k in ('nmse_num', 'nmse_den', 'sgcs_sum', 'sgcs_count',
                  'sgcs_stream_sum', 'sgcs_stream_count', 'per_sc_num', 'per_sc_den', 'per_port_num', 'per_port_den'):
            entry.pop(k, None)

    # ---------- 5. 按行 (task) × 列 (ratio) 顺序生成 grid_rows ----------
    for task in tasks:
        for ratio in mask_ratios:
            key = f"{task}@{ratio}"
            if key in grid_metrics:
                grid_rows.append(grid_metrics[key])

    return {
        'metrics': grid_metrics,
        'grid': grid_rows,
        'tasks': tasks,
        'mask_modes': mask_modes,
        'mask_ratios': mask_ratios,
        'n_samples': n_total,
    }


class CSIPreEvaluator:
    """
    CSI 预训练模型统一评估器。

    用法:
        evaluator = CSIPreEvaluator(output_dir='./results/my_eval')
        evaluator.add_model('wifo', checkpoint='path/to/best.pth')
        evaluator.add_model('csinet', checkpoint='path/to/best.pth')
        results = evaluator.run_all()
        # 与 feedback 包完全统一的 save() 接口
        evaluator.report.save("json")
        evaluator.report.save("html")
        evaluator.report.save("markdown")
    """

    def __init__(self,
                 output_dir: str = None,
                 device: str = 'cuda',
                 max_samples: int = 15000,
                 eval_cfg: cfg.EvalConfig = None,
                 s1_data_dir: str = None,
                 s2_data_dir: str = None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_samples = max_samples
        self.output_dir = output_dir or cfg.RESULTS_DIR
        self.eval_cfg = eval_cfg or cfg.EvalConfig()

        if s1_data_dir is not None and not self.eval_cfg.s1_data_dir:
            self.eval_cfg.s1_data_dir = s1_data_dir
        if s2_data_dir is not None and not self.eval_cfg.s2_data_dir:
            self.eval_cfg.s2_data_dir = s2_data_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self._models = []          # 待评估模型列表
        self._ckpt_map = {}        # model_name -> checkpoint_path
        self._model_type_map = {}
        self._model_map = {}       # model_name -> (wrapper, device)
        self._data_loaded = False

        # 统一 EvalReport：直接复用 feedback 包结构
        self._report = EvalReport(
            config=self.eval_cfg,
            meta={
                "task": cfg.TASK,
                "device": self.device,
                "output_dir": self.output_dir,
                "timestamp": (datetime.datetime.now(datetime.timezone.utc)
                              + datetime.timedelta(hours=8)
                              ).strftime('%Y-%m-%d %H:%M:%S CST'),
            },
        )

        self._s1_test = None
        self._s2_samples = None
        self._load_data()

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _load_data(self):
        """加载 S1 测试集和 S2 数据（路径优先用 eval_cfg.s*_data_dir）。"""
        s1_dir = cfg._resolve_scenario_dir(
            self.eval_cfg.s1_data_dir or cfg.DATA_DIR, '1')
        s2_dir = cfg._resolve_scenario_dir(
            self.eval_cfg.s2_data_dir or cfg.DATA_DIR, '2')

        print(f'[Evaluator] 加载 Scenario1 测试数据... (dir={s1_dir})')
        self._s1_test = load_scenario1(
            data_dir=s1_dir,
            split='test',
            max_samples=None,
            train_envs=cfg.EVAL_DEFAULT.get('train_envs', 1000)
        )
        print(f'  S1 测试样本数: {len(self._s1_test)}')

        print(f'[Evaluator] 加载 Scenario2 数据... (dir={s2_dir})')
        self._s2_samples = load_scenario2(
            data_dir=s2_dir,
            max_samples=cfg.S1_TO_S2_MAX_SAMPLES,
        )
        print(f'  S2 样本数: {len(self._s2_samples)}')

        self._data_loaded = True

    # ------------------------------------------------------------------
    # 模型注册
    # ------------------------------------------------------------------
    def add_model(self, model_type: str,
                  checkpoint: str = None,
                  alias: str = None):
        """添加一个待评估模型"""
        name = alias or model_type
        self._models.append(name)
        self._ckpt_map[name] = checkpoint
        self._model_type_map[name] = model_type

        try:
            wrapper, model_device, model_meta = build_from_bundle(
                checkpoint, device=self.device)
            print(f'  [✓] 模型构建成功 (device: {model_device})')
            self._model_map[name] = (wrapper, model_device, model_meta)
        except Exception as e:
            print(f'  [✗] 模型构建失败: {e}')
            self._model_map[name] = None

    def add_models(self, model_types: List[str]):
        for mt in model_types:
            self.add_model(mt)

    # ------------------------------------------------------------------
    # 核心评估
    # ------------------------------------------------------------------
    def _eval_single_model(self, model_name: str) -> Dict:
        """评估单个模型，将指标写入 self._report"""
        print(f'\n{"="*60}')
        print(f'[Evaluator] 评估模型: {model_name}')
        print(f'{"="*60}')

        t_start = time.time()

        cached = self._model_map.get(model_name)
        if cached is None:
            print(f'  [✗] 模型 {model_name} 未构建')
            return {}
        model, model_device, model_meta = cached
        print(f'  [✓] 模型已就绪 (device: {model_device})')

        # 任务性能指标
        if self.eval_cfg.run_task_metrics:
            print('  [→] 任务性能指标（spatial / frequency / joint × '
                  '0.25 / 0.5 / 0.75 mask ratio，共 9 项）...')
            t0 = time.time()
            grid = compute_task_metric_grid(
                model, self._s1_test,
                device=model_device,
                max_samples=self.max_samples,
                tasks=cfg.TASK_GRID_TASKS,
                mask_modes=cfg.TASK_GRID_MASK_MODES,
                mask_ratios=cfg.TASK_GRID_MASK_RATIOS,
            )
            n_total_samples = grid['n_samples']
            print(f'    ({time.time()-t0:.1f}s, {n_total_samples} 样本，'
                  f'9 个 (NMSE, SGCS) 指标)')
            # 保存子结果（不向 records 注册 n_samples；下游 HTML 按表格渲染）
            self._report.add_sub_dict("task_performance")[model_name] = grid

        # 各 mask 模式性能
        if self.eval_cfg.run_task_metrics:
            print('  [→] 各 mask 模式性能...')
            per_mask = compute_per_mask_metrics(
                model, self._s1_test,
                task=cfg.TASK,
                mask_ratio=cfg.EVAL_DEFAULT['mask_ratio'],
                device=model_device,
                max_samples=self.max_samples,
                mask_modes=cfg.MASK_MODES,
            )
            self._report.add_sub_dict("per_mask")[model_name] = per_mask

        # 存储与部署
        if self.eval_cfg.run_storage_metrics:
            print('  [→] 存储与部署指标...')
            storage_m = compute_storage_metrics(model)
            print(f'    Params={storage_m["params_M"]}M, FP32={storage_m["fp32_MB"]}MB, '
                  f'INT8={storage_m["int8_MB"]}MB')
            self._report.add(MetricRecord(
                name="Parameters", category="storage",
                value=storage_m["params_M"], higher_is_better=True,
                unit="M",
            ))
            self._report.add(MetricRecord(
                name="FP32 Size", category="storage",
                value=storage_m["fp32_MB"], higher_is_better=False,
                unit="MB",
            ))
            self._report.add(MetricRecord(
                name="INT8 Size", category="storage",
                value=storage_m["int8_MB"], higher_is_better=False,
                unit="MB",
            ))
            self._report.add_sub_dict("storage")[model_name] = storage_m

        # 计算效率
        if self.eval_cfg.run_compute_metrics:
            print('  [→] 计算效率指标（推理延迟 / FLOPs / 峰值显存）...')
            bs = cfg.EVAL_DEFAULT['batch_size']
            # 延迟/吞吐用配置的批大小，FLOPs 固定用 batch=1（模型固有复杂度）
            lat = measure_latency(model, device=model_device,
                                 batch_size=bs, task=cfg.TASK)
            flops = measure_flops(model, device=model_device, batch_size=1)
            mem = measure_peak_memory(model, device=model_device,
                                     batch_size=bs, task=cfg.TASK)
            compute_m = {
                'latency_mean_ms': lat['latency_mean_ms'],
                'latency_std_ms': lat['latency_std_ms'],
                'latency_min_ms': lat['latency_min_ms'],
                'latency_max_ms': lat['latency_max_ms'],
                'throughput_fps': lat['throughput_fps'],
                'flops_G': flops.get('flops_G'),
                'flops_method': flops.get('flops_method'),
                'flops_note': flops.get('note', ''),
                'peak_memory_MB': mem.get('peak_memory_MB', 0.0),
            }
            print(f'    Latency={compute_m["latency_mean_ms"]:.2f}ms, '
                  f'FLOPs={compute_m["flops_G"]}G, '
                  f'PeakMem={compute_m["peak_memory_MB"]:.1f}MB')
            self._report.add(MetricRecord(
                name="Latency Mean", category="computation",
                value=compute_m["latency_mean_ms"], higher_is_better=False,
                unit="ms",
            ))
            self._report.add(MetricRecord(
                name="Latency Std", category="computation",
                value=compute_m.get("latency_std_ms", 0), higher_is_better=False,
                unit="ms",
            ))
            self._report.add(MetricRecord(
                name="FLOPs", category="computation",
                value=compute_m["flops_G"], higher_is_better=False,
                unit="G",
            ))
            self._report.add(MetricRecord(
                name="Peak Memory", category="computation",
                value=compute_m["peak_memory_MB"], higher_is_better=False,
                unit="MB",
            ))
            self._report.add_sub_dict("computation")[model_name] = compute_m

        # 泛化鲁棒性
        if self.eval_cfg.run_robustness or self.eval_cfg.run_cross_scenario:
            print('  [→] 泛化鲁棒性...')
            rob_dir = os.path.join(self.output_dir, 'robustness_figs', model_name)
            robustness = compute_all_robustness(
                model,
                s1_samples=self._s1_test,
                s2_samples=self._s2_samples,
                task=cfg.TASK,
                mask_mode=cfg.EVAL_DEFAULT['mask_mode'],
                mask_ratio=cfg.EVAL_DEFAULT['mask_ratio'],
                cross_mask_pairs=cfg.CROSS_MASK_PAIRS,
                snr_levels=cfg.SNR_LEVELS,
                device=model_device,
                max_samples=self.max_samples,
                output_dir=rob_dir,
            )
            self._report.add_sub_dict("robustness")[model_name] = robustness

            cm = robustness.get('cross_mask', {})
            for pair, vals in cm.items():
                dn = vals.get('delta_nmse')
                ds = vals.get('delta_sgcs')
                if dn is not None and not np.isnan(dn):
                    print(f'    {pair}: ΔNMSE={dn:.4f}dB, ΔSGCS={ds:.6f}')
                else:
                    print(f'    {pair}: 数据不足')

            cs = robustness.get('cross_scenario', {})
            if cs and 'delta_nmse' in cs and 'delta_sgcs_percent' in cs:
                dn = cs['delta_nmse']
                ds = cs['delta_sgcs_percent']
                if not np.isnan(dn):
                    print(f'    S1→S2: ΔNMSE={dn:.4f}dB, ΔSGCS%={ds:.3f}%')
                else:
                    print(f'    S1→S2: 数据不足')
            elif not self._s2_samples:
                print('    [skip] cross_scenario：S2 样本数为 0，跳过泛化评估')

        # 更新 meta
        self._report.meta["model_name"] = model_name
        if model_meta:
            self._report.meta.setdefault("checkpoint", model_meta.get("checkpoint_path", ""))

        elapsed = time.time() - t_start
        print(f'  [✓] {model_name} 评估完成 ({elapsed:.1f}s)')
        return {}

    def run_all(self) -> Dict[str, Dict]:
        """对所有注册模型运行完整评估"""
        print(f'\n{"="*60}')
        print(f'CSI Pre-Evaluation 开始')
        print(f'{"="*60}')
        print(f'评估模型: {self._models}')
        print(f'默认配置: task={cfg.TASK}, mask_mode={cfg.EVAL_DEFAULT["mask_mode"]}, '
              f'mask_ratio={cfg.EVAL_DEFAULT["mask_ratio"]}')
        print(f'评估样本数: {self.max_samples}')
        print(f'输出目录: {self.output_dir}')

        for name in progress(self._models, desc="prediction: models", unit="model", leave=True):
            self._eval_single_model(name)

        print(f'\n{"="*60}')
        print(f'所有模型评估完成！')
        print(f'{"="*60}')
        return {}

    # ------------------------------------------------------------------
    # 结果查询（向后兼容）
    # ------------------------------------------------------------------
    @property
    def report(self) -> EvalReport:
        """统一报告对象，直接复用 feedback 包的 EvalReport 接口。

        支持 report.save("json"|"html"|"markdown")，与 feedback 任务接口完全一致。
        """
        return self._report

    @property
    def results(self) -> Dict[str, Dict]:
        """向后兼容：返回各模型的原始指标字典"""
        out = {}
        for name in self._models:
            rob_sub = self._report.sub_results.get("robustness", {}).get(name, {})
            out[name] = {
                "task_metrics": self._report.sub_results.get("task_performance", {}).get(name, {}),
                "per_mask_metrics": self._report.sub_results.get("per_mask", {}).get(name, {}),
                "storage_metrics": self._report.sub_results.get("storage", {}).get(name, {}),
                "compute_metrics": self._report.sub_results.get("computation", {}).get(name, {}),
                "robustness": rob_sub,
            }
        return out

    def get_nmse_table(self) -> Dict[str, float]:
        """向后兼容：从 grid 中取 joint × 0.5 模式的 NMSE。"""
        nmse_map = {}
        for name in self._models:
            entry = self._grid_lookup(name, task='joint', ratio=0.5)
            nmse_map[name] = entry.get('nmse_db', float('inf')) if entry else float('inf')
        return nmse_map

    def _grid_lookup(self, model_name: str, task: str, ratio: float) -> Optional[Dict]:
        """在 ``task_performance`` 子结果中按 (task, ratio) 查找一条 grid 记录。"""
        tp = self._report.sub_results.get("task_performance", {}).get(model_name, {})
        if not tp:
            return None
        # 优先用 metrics dict（key 形如 "joint@0.5"），fallback 到 grid 列表
        key = f"{task}@{ratio}"
        metrics = tp.get("metrics") or {}
        if key in metrics:
            return metrics[key]
        for entry in tp.get("grid", []) or []:
            if entry.get('task') == task and entry.get('mask_ratio') == ratio:
                return entry
        return None

    def summary(self):
        """打印简洁汇总（向后兼容）。

        NMSE/SGCS 取 grid 中 (joint, mask_ratio=0.5) 的结果作为代表性指标。
        """
        print(f'\n{"Model":<20} {"NMSE(dB)":>10} {"SGCS":>8} '
              f'{"S1":>8} {"S2":>8} {"S3":>8} {"S4":>8} '
              f'{"Params(M)":>10} {"Lat(ms)":>8} {"ΔNMSE→grid":>12}')
        print('-' * 108)
        for name in self._models:
            entry = self._grid_lookup(name, task='joint', ratio=0.5) or {}
            sm = self._report.sub_results.get("storage", {}).get(name, {})
            cm = self._report.sub_results.get("computation", {}).get(name, {})
            rob = self._report.sub_results.get("robustness", {}).get(name, {})
            cm_delta = rob.get('cross_mask', {}).get('comb→grid', {}).get('delta_nmse', 0)
            nmse = entry.get('nmse_db', float('nan'))
            sgcs = entry.get('sgcs', float('nan'))
            streams = list(entry.get('sgcs_streams') or [])[:4]
            streams += [float('nan')] * (4 - len(streams))
            params = sm.get('params_M', float('nan'))
            lat = cm.get('latency_mean_ms', float('nan'))
            print(f'{name:<20} {nmse:>10.4f} {sgcs:>8.6f} '
                  f'{streams[0]:>8.6f} {streams[1]:>8.6f} {streams[2]:>8.6f} {streams[3]:>8.6f} '
                  f'{params:>10.3f} {lat:>8.2f} {cm_delta:>12.4f}')

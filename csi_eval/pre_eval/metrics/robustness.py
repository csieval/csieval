"""
泛化鲁棒性 & 噪声鲁棒性 & 跨场景评估
====================================
四个评估维度：
1. Cross-mask ΔNMSE / ΔSGCS：comb→grid, comb→random_2d
2. Cross-ratio ΔNMSE / ΔSGCS：ratio 退化曲线（折线图）
3. 噪声鲁棒性：多 SNR 下 NMSE/SGCS 曲线
4. 跨场景保留率：Scenario1 训练 → Scenario2 测试

生成图像：
- cross_ratio_nmse_curve.png
- cross_ratio_sgcs_curve.png
- noise_nmse_curve.png
- noise_sgcs_curve.png
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from csi_eval.progress import progress

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ..params.params_7GHz import NT_PORT, NR, NF
from ..metrics.task_metrics import (
    compute_task_metrics,
    sgcs_3gpp_per_subband,
    sgcs_3gpp_per_subband_torch,
)
from ..model_registry import infer as registry_infer, ModelWrapper


# ----------------------------------------------------------------------
# 1. Cross-mask 鲁棒性
# ----------------------------------------------------------------------

def compute_cross_mask_robustness(model,
                                    samples: List[dict],
                                    task: str = 'joint',
                                    mask_ratio: float = 0.5,
                                    cross_mask_pairs: List[Tuple[str, str]] = None,
                                    device: str = 'cuda',
                                    max_samples: int = 15000) -> Dict:
    """
    计算 Cross-mask ΔNMSE / ΔSGCS。

    cross_mask_pairs: [('comb', 'grid'), ('comb', 'random_2d')]

    Returns:
        {
            'comb→grid':  {'delta_nmse': float, 'delta_sgcs': float},
            'comb→random_2d': {'delta_nmse': float, 'delta_sgcs': float},
        }
    """
    if cross_mask_pairs is None:
        cross_mask_pairs = [('comb', 'grid'), ('comb', 'random_2d')]

    # 以 comb 为基线
    baseline = compute_task_metrics(
        model, samples, task=task, mask_mode='comb',
        mask_ratio=mask_ratio, device=device, max_samples=max_samples)

    results = {}
    for src, tgt in progress(
        cross_mask_pairs, desc="prediction robustness: cross-mask", unit="pair", leave=False
    ):
        r = compute_task_metrics(
            model, samples, task=task, mask_mode=tgt,
            mask_ratio=mask_ratio, device=device, max_samples=max_samples)
        delta_nmse = r['nmse_db'] - baseline['nmse_db']
        delta_sgcs = baseline['sgcs'] - r['sgcs']   # SGCS 降低 = 负向
        base_streams = np.asarray(baseline.get('sgcs_streams') or [], dtype=np.float64)
        target_streams = np.asarray(r.get('sgcs_streams') or [], dtype=np.float64)
        delta_streams = (base_streams - target_streams).tolist() if (
            base_streams.size and base_streams.shape == target_streams.shape
        ) else []
        results[f'{src}→{tgt}'] = {
            'delta_nmse': round(delta_nmse, 4),
            'delta_sgcs': round(delta_sgcs, 6),
            'delta_sgcs_streams': [round(float(x), 6) for x in delta_streams],
            f'{tgt}_nmse_db': r['nmse_db'],
            f'{tgt}_sgcs': r['sgcs'],
            f'{tgt}_sgcs_streams': r.get('sgcs_streams'),
            'sgcs_stream_order': r.get('sgcs_stream_order'),
        }
    return results


# ----------------------------------------------------------------------
# 2. Cross-ratio 退化曲线
# ----------------------------------------------------------------------

def compute_cross_ratio_robustness(model,
                                     samples: List[dict],
                                     task: str = 'joint',
                                     mask_mode: str = 'comb',
                                     ratios: List[float] = None,
                                     device: str = 'cuda',
                                     max_samples: int = 15000) -> Dict:
    """
    计算各 ratio 下的 NMSE / SGCS，用于折线图。

    ratios: [0.75, 0.5, 0.25, 0.125]
    """
    if ratios is None:
        ratios = [0.75, 0.5, 0.25, 0.125]

    nmse_list, sgcs_list, sgcs_stream_list = [], [], []
    for r in progress(ratios, desc="prediction robustness: mask-ratio", unit="ratio", leave=False):
        m = compute_task_metrics(
            model, samples, task=task, mask_mode=mask_mode,
            mask_ratio=r, device=device, max_samples=max_samples)
        nmse_list.append(m['nmse_db'])
        sgcs_list.append(m['sgcs'])
        sgcs_stream_list.append(m.get('sgcs_streams'))

    return {
        'ratios': ratios,
        'port_keep_ratio': ratios,        
        'nmse_db': [round(x, 4) for x in nmse_list],
        'sgcs': [round(x, 6) for x in sgcs_list],
        'sgcs_streams': [
            [round(float(x), 6) for x in streams] if streams else []
            for streams in sgcs_stream_list
        ],
        'sgcs_stream_order': 'descending_covariance_eigenvalue',
    }



def compute_cross_ratio_degradation(model,
                                     samples: List[dict],
                                     task: str = 'joint',
                                     mask_mode: str = 'comb',
                                     degradation_pairs: List[Tuple[float, float]] = None,
                                     device: str = 'cuda',
                                     max_samples: int = 15000,
                                     precomputed: Optional[Dict] = None) -> Dict:
    """
    计算 ratio 退化（阶梯）时的 ΔNMSE / ΔSGCS。

    degradation_pairs: [(0.75, 0.5), (0.5, 0.25), (0.25, 0.125)]
    """
    if degradation_pairs is None:
        degradation_pairs = [(0.75, 0.5), (0.5, 0.25), (0.25, 0.125)]

    required_ratios = list(dict.fromkeys(r for pair in degradation_pairs for r in pair))
    full = precomputed
    if full is None or not set(required_ratios).issubset(set(full.get('ratios', []))):
        full = compute_cross_ratio_robustness(
            model, samples, task=task, mask_mode=mask_mode,
            ratios=required_ratios,
            device=device, max_samples=max_samples)

    # 整理成 dict: ratio -> nmse/sgcs
    ratio_map = dict(zip(full['ratios'], full['nmse_db']))
    sgcs_map = dict(zip(full['ratios'], full['sgcs']))
    stream_map = dict(zip(full['ratios'], full.get('sgcs_streams', [])))

    results = {}
    for hi, lo in degradation_pairs:
        key = f'{hi}→{lo}'
        hi_streams = np.asarray(stream_map.get(hi, []), dtype=np.float64)
        lo_streams = np.asarray(stream_map.get(lo, []), dtype=np.float64)
        stream_delta = (hi_streams - lo_streams).tolist() if (
            hi_streams.size and hi_streams.shape == lo_streams.shape
        ) else []
        results[key] = {
            'delta_nmse': round(ratio_map[lo] - ratio_map[hi], 4),
            'delta_sgcs': round(sgcs_map[hi] - sgcs_map[lo], 6),
            'delta_sgcs_streams': [round(float(x), 6) for x in stream_delta],
        }
    return results


def plot_cross_ratio_curve(data: Dict,
                            output_path: str,
                            metric: str = 'nmse_db',
                            ylabel: str = 'NMSE (dB)'):
    """
    绘制 ratio 退化折线图。

    data: {
        'ratios': [0.75, 0.5, 0.25, 0.125],
        'nmse_db': [...],
        'sgcs': [...],
    }
    """
    ratios = data['ratios']
    values = data[metric]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ratios, values, marker='o', linewidth=2, markersize=7,
            color='#6C63FF', label=ylabel)
    ax.set_xscale('log', base=2)
    ax.set_xticks(ratios)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_xlabel('Port Keep Ratio', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Cross Port-Ratio {ylabel}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)





# ----------------------------------------------------------------------
# 3. 噪声鲁棒性
# ----------------------------------------------------------------------

def compute_noise_robustness(model,
                               samples: List[dict],
                               task: str = 'joint',
                               mask_mode: str = 'comb',
                               mask_ratio: float = 0.5,
                               snr_levels: List[int] = None,
                               device: str = 'cuda',
                               max_samples: int = 15000) -> Dict:
    """
    在多 SNR 下评估模型鲁棒性。

    snr_levels: [5, 10, 15, 20, 25, 30] dB
    """
    from ..data_loader import prepare_batch, add_noise

    if snr_levels is None:
        snr_levels = [5, 10, 15, 20, 25, 30]

    if hasattr(model, 'eval'):
        model.eval()
    if max_samples:
        samples = samples[:max_samples]

    results = {f'snr_{s}': {'nmse_db': None, 'sgcs': None, 'sgcs_streams': None} for s in snr_levels}
    results['snr_inf'] = {'nmse_db': None, 'sgcs': None, 'sgcs_streams': None}  # 无噪声基线

    snr_sweep = snr_levels + [None]
    for snr in progress(
        snr_sweep, desc="prediction robustness: SNR sweep", unit="level", leave=False
    ):
        total_num, total_den = 0.0, 0.0
        # 3GPP SGCS：累积逐 (样本, 子带, stream) 的 SGCS 平方值
        sgcs_global_sum = 0.0
        sgcs_global_count = 0
        sgcs_stream_sum = np.zeros(4, dtype=np.float64)
        sgcs_stream_count = 0
        n = 0

        snr_label = "inf" if snr is None else str(snr)
        is_trad = (not isinstance(model, torch.nn.Module)
                   and not isinstance(model, ModelWrapper))
        if is_trad:
            # Traditional algorithms expose a single-sample callable only.
            sample_batches = ((sample,) for sample in samples)
        else:
            batch_size = 256
            sample_batches = (
                samples[start:start + batch_size]
                for start in range(0, len(samples), batch_size)
            )

        for chunk in progress(
            sample_batches,
            total=len(samples) if is_trad else (len(samples) + batch_size - 1) // batch_size,
            desc=f"prediction SNR={snr_label}",
            unit="sample" if is_trad else "batch",
            leave=False,
        ):
            H_known_list, H_target_list, mask_np = [], [], None
            for sample in chunk:
                H_known, H_target, sample_mask = prepare_batch(
                    sample, task, mask_mode, mask_ratio)
                H_known_list.append(add_noise(H_known, snr) if snr is not None else H_known)
                H_target_list.append(H_target)
                mask_np = sample_mask

            H_known_np = np.stack(H_known_list, axis=0)
            H_target_np = np.stack(H_target_list, axis=0)
            if is_trad:
                H_known_complex = H_known_np[0, 0] + 1j * H_known_np[0, 1]
                H_pred_complex = model(H_known_complex, mask_np, task)
                H_pred_np = np.stack([H_pred_complex.real, H_pred_complex.imag], axis=0)[None]
                sgcs_vals = sgcs_3gpp_per_subband(H_pred_np, H_target_np)
                pred_c = H_pred_np[:, 0] + 1j * H_pred_np[:, 1]
                true_c = H_target_np[:, 0] + 1j * H_target_np[:, 1]
                total_num += float(np.sum(np.abs(pred_c - true_c) ** 2))
                total_den += float(np.sum(np.abs(true_c) ** 2))
                if sgcs_vals.size:
                    sgcs_global_sum += float(sgcs_vals.sum())
                    sgcs_global_count += int(sgcs_vals.size)
                    sgcs_stream_sum += sgcs_vals.sum(axis=0)
                    sgcs_stream_count += int(sgcs_vals.shape[0])
            else:
                H_known_t = torch.from_numpy(H_known_np).to(device, non_blocking=True)
                H_target_t = torch.from_numpy(H_target_np).to(device, non_blocking=True)
                with torch.no_grad():
                    H_pred_t = registry_infer(model, H_known_t, mask_np, task, device)
                    diff = H_pred_t - H_target_t
                    total_num += float(diff.square().sum().item())
                    total_den += float(H_target_t.square().sum().item())
                    sgcs_vals_t = sgcs_3gpp_per_subband_torch(H_pred_t, H_target_t)
                if sgcs_vals_t.numel():
                    sgcs_global_sum += float(sgcs_vals_t.sum().item())
                    sgcs_global_count += int(sgcs_vals_t.numel())
                    sgcs_stream_sum += sgcs_vals_t.sum(dim=0).cpu().numpy()
                    sgcs_stream_count += int(sgcs_vals_t.shape[0])

            n += len(chunk)

        nmse = 10.0 * np.log10(total_num / (total_den + 1e-12))
        sgcs = (sgcs_global_sum / sgcs_global_count) if sgcs_global_count > 0 else float('nan')
        streams = (sgcs_stream_sum / sgcs_stream_count).tolist() if sgcs_stream_count else [float('nan')] * 4
        key = f'snr_{snr}' if snr is not None else 'snr_inf'
        results[key] = {
            'nmse_db': round(float(nmse), 4),
            'sgcs': round(float(sgcs), 6),
            'sgcs_streams': [round(float(x), 6) for x in streams],
            'sgcs_stream_order': 'descending_covariance_eigenvalue',
            'n_samples': n,
        }
        for i, value in enumerate(streams, start=1):
            results[key][f'sgcs_stream_{i}'] = round(float(value), 6)

    return results


def plot_noise_curve(data: Dict,
                      output_path: str,
                      metric: str = 'nmse_db',
                      ylabel: str = 'NMSE (dB)'):
    """绘制噪声鲁棒性折线图"""
    snr_levels = [5, 10, 15, 20, 25, 30]
    values = [data.get(f'snr_{s}', {}).get(metric, None) for s in snr_levels]
    values = [v for v in values if v is not None]
    snr_levels = [s for s, v in zip(snr_levels, [data.get(f'snr_{s}', {}).get(metric) for s in snr_levels]) if v is not None]

    if not values:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snr_levels, values, marker='s', linewidth=2, markersize=7,
            color='#FF6B6B')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Noise Robustness — {ylabel}', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. 跨场景评估
# ----------------------------------------------------------------------

def compute_cross_scenario(model,
                           s1_samples, s2_samples,
                           task='joint',
                           mask_mode='comb',
                           mask_ratio=0.5,
                           max_samples=None,
                           device=None):
    """
    跨场景评估：
    - S1 训练模型在 S2 数据上的 NMSE / SGCS
    - 与 S1 测试集结果对比，报告差值 / 衰减百分比

    Returns:
        {
            's1_nmse_db': float,
            's1_sgcs': float,
            's2_nmse_db': float,
            's2_sgcs': float,
            'delta_nmse': float,          # S2 - S1 (dB 差值)
            'delta_sgcs_percent': float,  # (S1 - S2) / S1 * 100
            's2_n_samples': int,
        }
    """
    s1 = compute_task_metrics(model, s1_samples,
                              task=task, mask_mode=mask_mode, mask_ratio=mask_ratio,
                              max_samples=max_samples,
                              device=device)
    if not s2_samples:
        return {
            'error': 'S2 样本为空, 无法计算跨场景保留率',
            's1_nmse_db': s1['nmse_db'],
            's1_sgcs': s1['sgcs'],
            's1_sgcs_streams': s1.get('sgcs_streams'),
            'sgcs_stream_order': s1.get('sgcs_stream_order'),
        }
    s2_samples_use = s2_samples
    if max_samples and len(s2_samples) > max_samples:
        s2_samples_use = s2_samples[:max_samples]
    s2 = compute_task_metrics(model, s2_samples_use,
                              task=task, mask_mode=mask_mode, mask_ratio=mask_ratio,
                              max_samples=max_samples,
                              device=device)
    delta_nmse = s2['nmse_db'] - s1['nmse_db']
    delta_sgcs_pct = (s1['sgcs'] - s2['sgcs']) / (s1['sgcs'] + 1e-12) * 100

    return {
        's1_nmse_db': round(s1['nmse_db'], 4),
        's1_sgcs': round(s1['sgcs'], 6),
        's1_sgcs_streams': [round(float(x), 6) for x in (s1.get('sgcs_streams') or [])],
        's2_nmse_db': round(s2['nmse_db'], 4),
        's2_sgcs': round(s2['sgcs'], 6),
        's2_sgcs_streams': [round(float(x), 6) for x in (s2.get('sgcs_streams') or [])],
        'sgcs_stream_order': s1.get('sgcs_stream_order'),
        'delta_nmse': round(delta_nmse, 4),
        'delta_sgcs_percent': round(delta_sgcs_pct, 3),
        's1_n_samples': s1['n_samples'],
        's2_n_samples': s2['n_samples'],
    }


# ----------------------------------------------------------------------
# 综合鲁棒性
# ----------------------------------------------------------------------

def compute_all_robustness(model,
                           s1_samples: List[dict],
                           s2_samples: List[dict],
                           task: str = 'joint',
                           mask_mode: str = 'comb',
                           mask_ratio: float = 0.5,
                           cross_mask_pairs: List[Tuple[str, str]] = None,
                           snr_levels: List[int] = None,
                           device: str = 'cuda',
                           max_samples: int = 15000,
                           output_dir: str = None) -> Dict:
    """
    综合鲁棒性评估，生成图像。

    Returns:
        完整鲁棒性结果字典
    """
    if cross_mask_pairs is None:
        cross_mask_pairs = [('comb', 'grid'), ('comb', 'random_2d')]
    if snr_levels is None:
        snr_levels = [5, 10, 15, 20, 25, 30]

    os.makedirs(output_dir or '.', exist_ok=True)

    # 1. Cross-mask
    cross_mask = compute_cross_mask_robustness(
        model, s1_samples,
        task=task, mask_ratio=mask_ratio,
        cross_mask_pairs=cross_mask_pairs,
        device=device, max_samples=max_samples,
    )

    # 2. Cross-ratio
    cross_ratio = compute_cross_ratio_robustness(
        model, s1_samples,
        task=task, mask_mode=mask_mode,
        device=device, max_samples=max_samples,
    )

    cross_ratio_deg = compute_cross_ratio_degradation(
        model, s1_samples,
        task=task, mask_mode=mask_mode,
        device=device, max_samples=max_samples,
        precomputed=cross_ratio,
    )

    # 3. 噪声鲁棒性
    noise = compute_noise_robustness(
        model, s1_samples,
        task=task, mask_mode=mask_mode, mask_ratio=mask_ratio,
        snr_levels=snr_levels,
        device=device, max_samples=max_samples,
    )

    # 4. 跨场景
    cross_scenario = compute_cross_scenario(
        model,
        s1_samples=s1_samples,
        s2_samples=s2_samples,
        task=task, mask_mode=mask_mode, mask_ratio=mask_ratio,
        max_samples=max_samples,
        device=device,
    )

    # 生成图像
    if output_dir:
        plot_cross_ratio_curve(cross_ratio, os.path.join(output_dir, 'cross_ratio_nmse_curve.png'),
                               'nmse_db', 'NMSE (dB)')
        plot_cross_ratio_curve(cross_ratio, os.path.join(output_dir, 'cross_ratio_sgcs_curve.png'),
                               'sgcs', 'SGCS')
        plot_noise_curve(noise, os.path.join(output_dir, 'noise_nmse_curve.png'),
                          'nmse_db', 'NMSE (dB)')
        plot_noise_curve(noise, os.path.join(output_dir, 'noise_sgcs_curve.png'),
                          'sgcs', 'SGCS')

    return {
        'cross_mask': cross_mask,
        'cross_ratio': cross_ratio,
        'cross_ratio_degradation': cross_ratio_deg,
        'noise_robustness': noise,
        'cross_scenario': cross_scenario,
    }

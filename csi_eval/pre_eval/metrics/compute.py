"""CSI Prediction 计算效率指标：延迟、FLOPs、显存。"""

import os
import sys
import time
import inspect
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from csi_eval.progress import progress

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ..params.params_7GHz import NT_PORT, NR, NF
from ..model_registry import infer as registry_infer, ModelWrapper, _internal_model


# ----------------------------------------------------------------------
# 推理延迟
# ----------------------------------------------------------------------

def measure_latency(model,
                     device: str = 'cuda',
                     batch_size: int = 1,
                     task: str = 'joint',
                     n_warmup: int = 3,
                     n_runs: int = 100) -> Dict:
    """
    测量推理延迟（ms）和吞吐（FPS）。

    Returns:
        {
            'latency_mean_ms': float,
            'latency_std_ms': float,
            'latency_min_ms': float,
            'latency_max_ms': float,
            'throughput_fps': float,
        }
    """
    nn_model = _internal_model(model)
    # mask 形状按 task 决定
    if task == 'frequency':
        mask = torch.ones(NF)
    elif task == 'spatial':
        mask = torch.ones(NT_PORT)
    else:
        mask = torch.ones(NT_PORT, NF)
    H_known = torch.randn(batch_size, 2, NT_PORT, NR, NF).to(device)
    # Warmup
    with torch.no_grad():
        for _ in progress(range(n_warmup), total=n_warmup,
                          desc="prediction latency: warmup", unit="run", leave=False):
            _ = registry_infer(model, H_known, mask, task, device)
    if device == 'cuda':
        torch.cuda.synchronize()
    # 计时
    times = []
    with torch.no_grad():
        for _ in progress(range(n_runs), total=n_runs,
                          desc="prediction latency: benchmark", unit="run", leave=False):
            t0 = time.perf_counter()
            _ = registry_infer(model, H_known, mask, task, device)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    throughput = batch_size * 1000.0 / mean_ms if mean_ms > 0 else 0
    return {
        'latency_mean_ms': round(mean_ms, 3),
        'latency_std_ms': round(std_ms, 3),
        'latency_min_ms': round(float(np.min(times)), 3),
        'latency_max_ms': round(float(np.max(times)), 3),
        'throughput_fps': round(throughput, 2),
    }


# ----------------------------------------------------------------------
# FLOPs
# ----------------------------------------------------------------------

def measure_flops(model, device='cuda', batch_size=1):
    """估算 FLOPs (GFLOPs)，对任意自定义模型鲁棒。

    策略：
      1. 用 thop（最通用）：自动追踪所有标准 PyTorch 算子的 FLOPs。
      2. 用 register_forward_hook 手动统计 Linear/Conv/Matmul 等。
      3. 按参数量估算（1 param ≈ 3 FLOPs）。

    若测量过程出错，返回 {'flops_G': None, 'flops_method': 'unavailable', 'error': str}。
    """
    nn_model = _internal_model(model)
    sig = None
    try:
        sig = inspect.signature(nn_model.forward)
        params = [p for p in sig.parameters.values() if p.name != 'self']
        n_args = len(params)
    except Exception:
        n_args = 3

    H_known = torch.randn(batch_size, 2, NT_PORT, NR, NF).to(device)
    jnt_mask = torch.ones(NT_PORT, NF).to(device)

    def _build_args(_n_args):
        if _n_args == 1:
            return (H_known,)
        if _n_args == 2:
            return (H_known, jnt_mask)
        return (H_known, jnt_mask, 'joint')

    # -------- 1. thop --------
    try:
        from thop import profile
        class _ThopWrapper(nn.Module):
            def __init__(self, m, args):
                super().__init__()
                self.m = m
                self.args = args
            def forward(self, *a):
                return self.m(*self.args)
        args = _build_args(n_args)
        wrapped = _ThopWrapper(nn_model, args)
        wrapped.eval()
        flops, _ = profile(wrapped, inputs=(), verbose=False)
        if flops > 1e6:
            return {'flops_G': round(flops / 1e9, 3), 'flops_method': 'thop'}
    except Exception:
        pass

    # -------- 2. register_forward_hook --------
    try:
        total_macs = 0
        def _hook_linear(module, inp, out):
            nonlocal total_macs
            inp_tensor = inp[0] if isinstance(inp, tuple) else inp
            in_features = inp_tensor.shape[-1] if len(inp_tensor.shape) >= 2 else module.in_features
            total_macs += in_features * module.out_features

        def _hook_conv(module, inp, out):
            nonlocal total_macs
            inp_tensor = inp[0] if isinstance(inp, tuple) else inp
            c_in = inp_tensor.shape[1] if len(inp_tensor.shape) >= 2 else module.in_channels
            ks = 1
            for k in module.kernel_size:
                ks *= k
            if hasattr(out, 'shape') and len(out.shape) >= 2:
                spatial = 1
                for d in out.shape[2:]:
                    spatial *= d
                total_macs += spatial * c_in * ks * module.out_channels

        handles = []
        for m in nn_model.modules():
            if isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(_hook_linear))
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                handles.append(m.register_forward_hook(_hook_conv))

        args = _build_args(n_args)
        nn_model.eval()
        with torch.no_grad():
            _ = nn_model(*args)
        for h in handles:
            h.remove()
        if total_macs > 1e6:
            flops = 2 * total_macs
            return {'flops_G': round(flops / 1e9, 3), 'flops_method': 'profile'}
    except Exception:
        pass

    # -------- 3. 按参数量估算 --------
    try:
        n_params = sum(p.numel() for p in nn_model.parameters())
        flops = n_params * 3  # 经验值：推理时每个参数 ≈ 2-4 FLOPs
        return {'flops_G': round(flops / 1e9, 3), 'flops_method': 'params',
                'note': f'{n_params/1e6:.2f}M params × ~3 FLOPs/param ≈ {flops/1e9:.2f}G'}
    except Exception:
        pass

    return {'flops_G': None, 'flops_method': 'unavailable', 'error': '无法估算 FLOPs'}


# ----------------------------------------------------------------------
# 峰值显存
# ----------------------------------------------------------------------

def measure_peak_memory(model,
                        device: str = 'cuda',
                        batch_size: int = 1,
                        task: str = 'joint') -> Dict[str, float]:
    """
    测量推理峰值显存（MB）。

    Returns:
        {
            'peak_memory_MB': float,
            'input_memory_MB': float,   # 输入占用的显存
        }
    """
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        return {'peak_memory_MB': 0.0, 'input_memory_MB': 0.0}

    # 提前判断"传统算法"：它不走 torch 算子，测不到任何 CUDA 分配，直接返回 0
    is_trad = (not isinstance(model, torch.nn.Module)
               and not isinstance(model, ModelWrapper))
    if is_trad:
        return {'peak_memory_MB': 0.0, 'input_memory_MB': 0.0}

    # 关键：所有 CUDA 操作必须锁在模型所在的那张卡上。
    # 否则 reset_peak_memory_stats / max_memory_allocated 会默认作用于
    # torch.cuda.current_device()（通常 cuda:0），而模型在 cuda:1，
    # 导致读到 0.0 MB 的假象。
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        H_known = torch.randn(batch_size, 2, NT_PORT, NR, NF).to(device)
        input_bytes = H_known.element_size() * H_known.nelement()

        with torch.no_grad():
            mask = torch.ones(NT_PORT, NF, device=device)
            _ = registry_infer(model, H_known, mask, task, device)
        # 等前向彻底落盘，避免异步 kernel 导致峰值被低估
        torch.cuda.synchronize()

        peak_bytes = torch.cuda.max_memory_allocated()

    MB = 1024 * 1024
    return {
        'peak_memory_MB': round(peak_bytes / MB, 2),
        'input_memory_MB': round(input_bytes / MB, 2),
    }


# ----------------------------------------------------------------------
# 综合计算效率
# ----------------------------------------------------------------------

def compute_compute_metrics(model,
                            device: str = 'cuda',
                            batch_size: int = 1,
                            task: str = 'joint',
                            **kwargs) -> Dict:
    """
    综合计算效率指标。

    Returns:
        {
            'latency_mean_ms': float,
            'latency_std_ms': float,
            'throughput_fps': float,
            'flops_G': float,
            'flops_method': str,
            'peak_memory_MB': float,
        }
    """
    lat = measure_latency(model, device, batch_size, task=task, **kwargs)
    flops = measure_flops(model, device, batch_size)
    mem = measure_peak_memory(model, device, batch_size, task=task)
    return {
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

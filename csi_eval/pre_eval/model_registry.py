"""
模型加载器（黑盒版）
====================
自动适配用户模型的 forward 签名，无需用户修改模型代码。
"""
import os
import sys
import json
import importlib
import importlib.util
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from .params.params_7GHz import NT_PORT, NR, NF

EXPECTED_INPUT_SHAPE  = (None, 2, NT_PORT, NR, NF)
EXPECTED_OUTPUT_SHAPE = (None, 2, NT_PORT, NR, NF)


# ────────────────────────────────────────────────────────────────
# 1. 动态加载模型类
# ────────────────────────────────────────────────────────────────

def _load_class_from_meta(meta: Dict[str, Any], bundle_dir: str):
    """
    加载用户模型类。

    module_path 支持两种写法：
      A) 'models.transformer.wifo.WiFo'  — 从评估包代码 import
      B) 'my_model.MyModel'              — 从 bundle/my_model.py import
    """
    module_path = meta['module_path']
    class_name  = meta['model_class']
    build_params = meta.get('build_params', {})

    parts = module_path.split('.')
    if len(parts) > 1 and parts[-1] == class_name:
        module_name = '.'.join(parts[:-1])
        # 尝试从评估包 import
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # fallback：从 bundle 目录加载同名 .py
            file_basename = parts[-2]
            file_path = os.path.join(bundle_dir, f'{file_basename}.py')
            if os.path.isfile(file_path):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                raise ImportError(
                    f'无法 import {module_name!r}，'
                    f'且 bundle 目录 {bundle_dir} 中也没有 {file_basename}.py'
                )
    else:
        raise ValueError(
            f'module_path={module_path!r} 应为 "<module>.<Class>"，'
            f'且最后一段必须等于 model_class={class_name!r}'
        )

    ModelCls = getattr(module, class_name, None)
    if ModelCls is None:
        raise AttributeError(f'模块 {module_path!r} 中找不到类 {class_name!r}')
    return ModelCls, build_params


def _find_weight_file(bundle_dir: str) -> Optional[str]:
    for name in ('best.pth', 'model.pth', 'weights.pth', 'checkpoint.pth', 'ckpt.pth'):
        p = os.path.join(bundle_dir, name)
        if os.path.isfile(p):
            return p
    for f in os.listdir(bundle_dir):
        if f.endswith(('.pth', '.pt')):
            return os.path.join(bundle_dir, f)
    return None


def _load_state_dict(weight_path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(weight_path, map_location='cpu')
    if isinstance(state, dict):
        for key in ('model_state_dict', 'state_dict', 'model', 'net', 'ema_state_dict'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return state


# ────────────────────────────────────────────────────────────────
# 2. 签名探测 & 推理包装器（核心）
# ────────────────────────────────────────────────────────────────

def _probe_forward_signature(model: torch.nn.Module) -> Callable:
    """
    自动探测 model.forward 的参数签名，并返回一个兼容的推理函数。

    支持的签名（按优先级尝试）：
      (H_known, mask, task)  ← 标准三参
      (H_known, mask)        ← 两参
      (H_known,)             ← 单参（WiFo / 用户模型）

    Returns:
        一个函数 forward(H_known, mask, task)，内部调用模型的真实签名
    """
    import inspect

    sig = inspect.signature(model.forward)
    n_params = len([
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
    ])

    # 三参：(H_known, mask, task)
    if n_params >= 3:
        print(f'  [registry] forward 签名: (H_known, mask, task) — 标准接口')
        def forward_fn(H_known, mask, task):
            return model(H_known, mask, task)
        return forward_fn

    # 两参：(H_known, mask)
    if n_params >= 2:
        print(f'  [registry] forward 签名: (H_known, mask) — 两参数接口')
        def forward_fn(H_known, mask, task):
            return model(H_known, mask)
        return forward_fn

    # 单参：(H_known) — WiFo 等大多数模型走这里
    print(f'  [registry] forward 签名: (H_known) — 单参数接口（WiFo/用户模型）')
    def forward_fn(H_known, mask, task):
        return model(H_known)
    return forward_fn


# ────────────────────────────────────────────────────────────────
# 3. 统一推理接口（所有指标计算模块调用这里）
# ────────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    对用户模型做统一封装：
    - 自动探测 forward 签名
    - 输出形状校验
    - 自动设备迁移
    """
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._forward = _probe_forward_signature(model)
        self._device = next(model.parameters()).device if len(list(model.parameters())) else 'cpu'

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def device(self) -> str:
        return str(self._device)

    def eval(self) -> 'ModelWrapper':
        """把内部 nn.Module 切到 eval 模式（兼容 task_metrics/robustness 中 model.eval() 的调用）"""
        self._model.eval()
        return self

    # 透传到内部 nn.Module，让 named_modules / parameters / state_dict 等接口在
    # FLOPs 探测、thop/fvcore、checkpoint 加载等场景下继续工作
    def named_modules(self, *args, **kwargs):
        return self._model.named_modules(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self._model.named_parameters(*args, **kwargs)

    def parameters(self):
        return self._model.parameters()

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._model.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self._model.to(*args, **kwargs)
        return self

    def __call__(self, H_known: torch.Tensor,
                mask, task: str) -> torch.Tensor:
        """
        统一推理接口。

        Args:
            H_known: (B, 2, 256, 8, 52)
            mask:    评估包生成的 mask（shape 取决于 task）
            task:    'joint' | 'spatial' | 'frequency'
        Returns:
            torch.Tensor of shape (B, 2, 256, 8, 52)
        """
        return self._forward(H_known, mask, task)


# ────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────
# 4. 从 bundle 加载模型
# ────────────────────────────────────────────────────────────────

def build_from_bundle(bundle_dir: str, device: str = 'cuda') -> Tuple[ModelWrapper, str, Dict]:
    """
    从用户提供的 checkpoint bundle 加载模型。

    Args:
        bundle_dir: bundle 目录（包含 model_meta.json + *.pth + 模型代码）
        device: 'cuda' | 'cpu'

    Returns:
        (ModelWrapper, device_str, training_info_dict)
    """
    bundle_dir = os.path.abspath(bundle_dir)
    meta_path  = os.path.join(bundle_dir, 'model_meta.json')

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f'未在 {bundle_dir} 中找到 model_meta.json。\n'
            f'Bundle 必须包含:\n'
            f'  1. model_meta.json  — 模型元数据\n'
            f'  2. *.pth / *.pt     — 模型权重\n'
            f'  3. <model>.py       — （可选）若模型类不在评估包内\n'
            f'参考 docs/USER_GUIDE.md 与 examples/external_model_bundle/'
        )

    with open(meta_path) as f:
        meta = json.load(f)

    for k in ('model_class', 'module_path', 'build_params'):
        if k not in meta:
            raise ValueError(f'model_meta.json 缺少必填字段: {k!r}')

    # 加载类
    ModelCls, build_params = _load_class_from_meta(meta, bundle_dir)
    print(f'  [registry] 模型类: {ModelCls.__module__}.{ModelCls.__name__}')

    # 实例化
    try:
        model = ModelCls(**build_params)
    except TypeError as e:
        raise TypeError(
            f'用 build_params={build_params} 实例化 {ModelCls.__name__} 失败: {e}\n'
            f'请检查 model_meta.json 的 build_params 与模型 __init__ 签名是否一致。'
        ) from e

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f'{ModelCls.__name__} 不是 torch.nn.Module 子类。'
            f'评估包要求模型继承 nn.Module。'
        )

    # 加载权重
    weight_path = _find_weight_file(bundle_dir)
    if weight_path is None:
        raise FileNotFoundError(f'未在 {bundle_dir} 中找到任何 .pth / .pt 权重文件。')

    state = _load_state_dict(weight_path)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}

    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f'  [registry] ⚠ {len(result.missing_keys)} 个权重未加载 (示例: {result.missing_keys[:3]})')
    if result.unexpected_keys:
        print(f'  [registry] ⚠ {len(result.unexpected_keys)} 个多余权重 (示例: {result.unexpected_keys[:3]})')
    print(f'  [registry] ✓ 权重已加载: {os.path.basename(weight_path)}')

    # 设备
    target = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    model = model.to(target)
    model.eval()

    wrapper = ModelWrapper(model)
    training_info = meta.get('training_info', {})
    return wrapper, wrapper.device, training_info


# ────────────────────────────────────────────────────────────────
# 5. 统一 infer 接口（外部模块用，自动识别类型）
# ────────────────────────────────────────────────────────────────

def infer(model,
          H_known: torch.Tensor,
          mask,
          task: str,
          device: str = None) -> torch.Tensor:
    """
    统一推理入口，自动适配：
      - ModelWrapper         → 走 .__call__(H_known, mask, task)
      - torch.nn.Module      → 自动探测签名（1/2/3 参）
      - 传统算法（callable） → 调用 model(H_complex, mask_np, task)

    Args:
        model:   ModelWrapper / nn.Module / 传统算法对象
        H_known: (B, 2, 256, 8, 52) tensor
        mask:    与 task 形状匹配的 mask（可以是 None）
        task:    'joint' | 'spatial' | 'frequency'
        device:  目标设备；None 表示用模型自身设备

    Returns:
        H_pred: (B, 2, 256, 8, 52) tensor, float32
    """
    # 1) ModelWrapper：直接走封装好的签名适配逻辑
    if isinstance(model, ModelWrapper):
        return model(H_known, mask, task)

    # 2) torch.nn.Module：探测真实签名
    if isinstance(model, torch.nn.Module):
        target_device = device or (next(model.parameters()).device
                                   if len(list(model.parameters())) else 'cpu')
        H_known = H_known.to(target_device)
        with torch.no_grad():
            try:
                # 优先尝试三参
                return model(H_known, mask, task)
            except TypeError:
                try:
                    return model(H_known, mask)
                except TypeError:
                    return model(H_known)

    # 3) 传统算法：单样本复数接口
    H_known_np = H_known[0].cpu().numpy() if H_known.dim() == 5 else H_known.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    H_known_complex = H_known_np[0] + 1j * H_known_np[1]
    H_pred_complex = model(H_known_complex, mask_np, task)
    H_pred_np = np.stack([H_pred_complex.real, H_pred_complex.imag], axis=0)
    return torch.from_numpy(H_pred_np).unsqueeze(0).float()

def _internal_model(model) -> torch.nn.Module:
    """如果 model 是 ModelWrapper，返回内部 nn.Module；否则原样返回"""
    if isinstance(model, ModelWrapper):
        return model.model
    return model
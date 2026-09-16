"""ModelAdapter implementations.

A ``ModelAdapter`` wraps a torch ``nn.Module`` (or anything duck-type
compatible) and exposes a uniform interface to the framework. The
implementation probes the wrapped module for optional capabilities
(``encode``, ``get_compression_ratio``, ...) and records whether they
exist, so metrics can skip gracefully when the model does not provide
a particular feature.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..core.protocols import (
    CAP_MODEL_COMPRESSION_RATIO, CAP_MODEL_ENCODE, CAP_MODEL_INFO,
    CAP_MODEL_MACS, CAP_MODEL_QUANT_BITS,
)


def _state_dict_mb(model: nn.Module) -> float:
    """Return the in-memory state_dict size in MB (float32 assumption)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 ** 2)


class ModelAdapter:
    """Adapter that wraps a torch ``nn.Module``.

    Only ``forward`` and ``get_input_shape`` are strictly required from
    the underlying module. The adapter probes optional capabilities and
    reports them via the framework's capability system.
    """

    def __init__(self, module: nn.Module, task_name: str = ""):
        self._module = module
        self._task_name = task_name
        # Cache a copy of the model_info dict to avoid recomputing each call
        self._info_cache: Optional[Dict[str, Any]] = None

    # --- nn.Module passthroughs (so the adapter IS the module) ---
    def parameters(self):
        return self._module.parameters()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self._module.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self._module.to(*args, **kwargs)
        return self

    def cpu(self):
        self._module.cpu()
        return self

    def cuda(self, *args, **kwargs):
        self._module.cuda(*args, **kwargs)
        return self

    def eval(self):
        self._module.eval()
        return self

    def train(self, mode: bool = True):
        self._module.train(mode)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, name: str):
        # Forward unknown attribute lookups to the wrapped module
        # (so .parameters(), .named_modules() etc. still work transparently)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # If the wrapped module has the attribute, write through to it
        # so callers like ``model.quant_bits = 2`` actually take effect
        # on the underlying ``nn.Module`` (used by the quantization
        # robustness sweep). For adapter-only fields (e.g. ``_module``,
        # ``_task_name``) fall back to the default behaviour.
        if name in {"_module", "_task_name", "_info_cache"}:
            super().__setattr__(name, value)
            return
        inner = self.__dict__.get("_module", None)
        if inner is not None and hasattr(inner, name):
            setattr(inner, name, value)
            return
        super().__setattr__(name, value)

    # --- Required capability ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._module.forward(x)

    def get_input_shape(self) -> Tuple[int, ...]:
        if hasattr(self._module, "get_input_shape"):
            try:
                return tuple(self._module.get_input_shape())
            except Exception:
                pass
        if hasattr(self._module, "input_shape"):
            shp = self._module.input_shape
            if isinstance(shp, (tuple, list)):
                return tuple(shp)
        # Try to infer from the first Linear/Conv2d layer
        return self._infer_input_shape()

    def _infer_input_shape(self) -> Tuple[int, ...]:
        """Best-effort input shape inference from the first layer."""
        for m in self._module.modules():
            if isinstance(m, nn.Conv2d):
                return (m.in_channels, 1, 1)
            if isinstance(m, nn.Linear):
                return (m.in_features,)
        return (1,)

    def get_output_shape(self) -> Tuple[int, ...]:
        if hasattr(self._module, "get_output_shape"):
            try:
                return tuple(self._module.get_output_shape())
            except Exception:
                pass
        return self.get_input_shape()

    # --- Optional capabilities (probed) ---
    def encode(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if callable(getattr(self._module, "encode", None)):
            try:
                return self._module.encode(x)
            except Exception:
                return None
        return None

    def get_compression_ratio(self) -> Optional[float]:
        if callable(getattr(self._module, "get_compression_ratio", None)):
            try:
                return float(self._module.get_compression_ratio())
            except Exception:
                pass
        # Try to infer from model_info
        info = self.get_model_info()
        cr = info.get("compression_ratio")
        if cr is not None:
            return float(cr)
        return None

    def get_model_info(self) -> Dict[str, Any]:
        if self._info_cache is not None:
            return self._info_cache
        if callable(getattr(self._module, "get_model_info", None)):
            try:
                self._info_cache = dict(self._module.get_model_info())
                return self._info_cache
            except Exception:
                pass
        # Fallback: synthesize a minimal info dict
        params = sum(p.numel() for p in self._module.parameters())
        info: Dict[str, Any] = {
            "name": self._module.__class__.__name__,
            "type": self._module.__class__.__name__,
            "task": self._task_name,
            "params": int(params),
            "params_m": params / 1e6,
            "model_size_mb": _state_dict_mb(self._module),
        }
        # Try to fill in shapes
        try:
            info["input_shape"] = list(self.get_input_shape())
        except Exception:
            pass
        self._info_cache = info
        return info

    def get_quant_bits(self) -> int:
        if hasattr(self._module, "quant_bits"):
            try:
                return int(self._module.quant_bits)
            except Exception:
                return 0
        return 0

    def estimate_macs(self, batch_size: int = 1) -> Optional[int]:
        if callable(getattr(self._module, "estimate_macs", None)):
            try:
                return int(self._module.estimate_macs(batch_size=batch_size))
            except Exception:
                pass
        # Fallback: rough heuristic from Linear layer parameter counts
        macs = 0
        for m in self._module.modules():
            if isinstance(m, nn.Linear):
                macs += batch_size * m.in_features * m.out_features
            elif isinstance(m, nn.Conv2d):
                h_out, w_out = 1, 1
                if isinstance(m.output_size, tuple) and len(m.output_size) >= 2:
                    h_out, w_out = m.output_size[0], m.output_size[1]
                elif hasattr(m, "out_channels"):
                    kernel_ops = m.kernel_size[0] * m.kernel_size[1]
                    macs += batch_size * m.in_channels * m.out_channels * kernel_ops * h_out * w_out
        if macs > 0:
            return macs
        return None

    def get_state_dict_mb(self) -> float:
        return _state_dict_mb(self._module)

    def task_name(self) -> str:
        if self._task_name:
            return self._task_name
        info = self.get_model_info()
        return info.get("task", "unknown")


# Backwards-compat alias
NNModuleAdapter = ModelAdapter

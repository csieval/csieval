"""EvalContext: shared, mutable state passed to every Metric.compute().

The context carries the ModelAdapter, DataAdapter, and a small KV cache so
that expensive intermediate results (e.g. per-sample NMSE, compressed codes)
are computed at most once across all metrics in a category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .config import EvalConfig
from .protocols import DataAdapter, ModelAdapter


@dataclass
class EvalContext:
    """Per-run shared state for Metric.compute() calls.

    Use ``ctx.get_or_compute(key, fn)`` to memoize expensive operations.
    """

    model: ModelAdapter
    data: DataAdapter
    device: torch.device
    config: EvalConfig
    # Model metadata from model_meta.json (declared by user, not from model code).
    # Provides nt, n_subbands, compressed_dim, total_dim, layout_hint etc.
    model_meta: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    splits: Tuple[str, ...] = ("test",)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    # Optional: a secondary DataAdapter for the OOD target dataset
    # (e.g. Part1_NEW or Part2). Set by the Evaluator when
    # ``config.ood_dataset`` is provided. The generalization / cross-scenario
    # metrics read this field to evaluate on the OOD target instead of the
    # in-distribution test set.
    ood_data: Optional[Any] = None
    # Optional: the active task adapter (e.g. EigenvectorFeedbackTask).
    # Some metrics need to call back into the task to build a secondary
    # data adapter on demand.
    task: Optional[Any] = None

    # ---- Capability probing (cheap) ----
    def has(self, capability: str) -> bool:
        if capability in self.capabilities:
            return self.capabilities[capability]
        val = self._probe(capability)
        self.capabilities[capability] = val
        return val

    def _probe(self, capability: str) -> bool:
        if capability == "model.encode":
            return callable(getattr(self.model, "encode", None))
        if capability == "model.compression_ratio":
            return callable(getattr(self.model, "get_compression_ratio", None))
        if capability == "model.quant_bits":
            return callable(getattr(self.model, "get_quant_bits", None))
        if capability == "model.macs":
            return callable(getattr(self.model, "estimate_macs", None))
        if capability == "model.info":
            return callable(getattr(self.model, "get_model_info", None))
        if capability == "data.complex_raw":
            return callable(getattr(self.data, "get_complex_raw", None))
        if capability == "data.env_index":
            return callable(getattr(self.data, "env_index", None))
        if capability == "data.env_index_array":
            return hasattr(self.data, "env_index_array")
        if capability == "training.history":
            return self.config.training_history_path is not None
        return False

    # ---- Cache ----
    def get_or_compute(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.cache.get(key, default)

    # ---- Model metadata (from model_meta.json, not from model code) ----
    def meta(self, key: str, default: Any = None) -> Any:
        """Read a value from model_meta.json (set at bundle-load time).

        This is the primary way metrics should obtain architectural parameters
        (nt, n_subbands, compressed_dim, etc.) instead of querying the model.
        """
        return self.model_meta.get(key, default)

    def add_sub(self, key: str, value: Any) -> None:
        """Append a value to a list-typed cache key (used by sub_results)."""
        cur = self.cache.get(key)
        if isinstance(cur, list):
            cur.append(value)
        elif cur is None:
            self.cache[key] = [value]
        else:
            self.cache[key] = [cur, value]

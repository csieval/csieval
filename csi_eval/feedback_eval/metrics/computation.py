"""Computation metrics: latency, FLOPs, MACs."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from csi_eval.progress import progress

from ..core.context import EvalContext
from ..core.registries import MetricRegistry


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
@MetricRegistry.register("computation", requires=frozenset())
class Latency:
    """Average inference latency in ms (over ``latency_runs`` runs)."""

    name = "latency"
    category = "computation"
    higher_is_better = False
    requires = frozenset()
    unit = "ms"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        runs = int(ctx.config.latency_runs)
        input_shape = self._resolve_shape(ctx)
        if input_shape is None:
            return {"value": None, "note": "no input_shape available (set input_shape in model_meta.json)"}

        results: Dict[str, float] = {}
        for bs in (1, max(1, min(16, runs))):
            results[f"bs{bs}"] = self._measure(ctx, input_shape, batch_size=bs, runs=runs)

        return {"value": results["bs1"], "by_batch_size": results, "unit": "ms"}

    @staticmethod
    def _resolve_shape(ctx: EvalContext):
        try:
            return tuple(ctx.model.get_input_shape())
        except Exception:
            pass
        shape = ctx.meta("input_shape")
        if shape:
            return tuple(shape)
        return None

    @torch.no_grad()
    def _measure(self, ctx: EvalContext, input_shape, batch_size: int, runs: int) -> float:
        ctx.model.eval().to(ctx.device)
        x = torch.randn((batch_size, *input_shape), device=ctx.device)
        # Warmup
        warmup_runs = max(1, runs // 10)
        for _ in progress(range(warmup_runs), total=warmup_runs,
                          desc=f"feedback latency bs={batch_size}: warmup",
                          unit="run", leave=False):
            _ = ctx.model.forward(x)
        if str(ctx.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in progress(range(runs), total=runs,
                          desc=f"feedback latency bs={batch_size}: benchmark",
                          unit="run", leave=False):
            _ = ctx.model.forward(x)
        if str(ctx.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / runs


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------
@MetricRegistry.register("computation", requires=frozenset())
class FLOPs:
    """FLOPs (floating point operations) per inference.

    Strategy:
      1. If the model exposes estimate_macs, use 2*macs.
      2. Otherwise, try thop.profile (if installed).
      3. Otherwise, return None.
    """

    name = "flops"
    category = "computation"
    higher_is_better = False
    requires = frozenset()
    unit = "FLOPs"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        macs = None
        # Strategy 1: model's estimate_macs
        if callable(getattr(ctx.model, "estimate_macs", None)):
            try:
                macs = int(ctx.model.estimate_macs(batch_size=1))
            except Exception:
                macs = None
        # Strategy 2: thop fallback
        if macs is None:
            try:
                import thop  # type: ignore
                shape = self._resolve_shape(ctx)
                if shape:
                    x = torch.randn((1, *shape), device=ctx.device)
                    macs, _ = thop.profile(ctx.model, inputs=(x,), verbose=False)
            except Exception:
                macs = None
        if macs is None:
            return {"value": None, "note": "FLOPs profiler not available (model lacks estimate_macs and thop is not installed)"}
        return {"value": int(2 * macs), "macs": int(macs), "unit": "FLOPs"}

    @staticmethod
    def _resolve_shape(ctx: EvalContext):
        try:
            return tuple(ctx.model.get_input_shape())
        except Exception:
            pass
        shape = ctx.meta("input_shape")
        if shape:
            return tuple(shape)
        return None


# ---------------------------------------------------------------------------
# MAC
# ---------------------------------------------------------------------------
@MetricRegistry.register("computation", requires=frozenset())
class MAC:
    """Multiply-accumulate operations per inference.

    Strategy:
      1. If the model exposes estimate_macs, use it.
      2. Otherwise, try thop.profile (if installed).
      3. Otherwise, return None.
    """

    name = "macs"
    category = "computation"
    higher_is_better = False
    requires = frozenset()
    unit = "MACs"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        macs = None
        note = ""
        # Strategy 1: model's estimate_macs
        if callable(getattr(ctx.model, "estimate_macs", None)):
            try:
                macs = int(ctx.model.estimate_macs(batch_size=1))
            except Exception as e:
                note = f"estimate_macs failed: {e}"
        # Strategy 2: thop fallback
        if macs is None:
            try:
                import thop  # type: ignore
                input_shape = self._resolve_shape(ctx)
                if input_shape:
                    x = torch.randn((1, *input_shape), device=ctx.device)
                    macs, _ = thop.profile(ctx.model, inputs=(x,), verbose=False)
            except Exception:
                pass
        if macs is None:
            return {"value": None, "note": note or "MACs profiler not available"}
        return {"value": int(macs), "unit": "MACs"}

    @staticmethod
    def _resolve_shape(ctx: EvalContext):
        try:
            return tuple(ctx.model.get_input_shape())
        except Exception:
            pass
        # Fall back to model_meta input_shape
        shape = ctx.meta("input_shape")
        if shape:
            return tuple(shape)
        return None

"""Task performance metrics: NMSE, MSE, correlation (rho), SGCS, EVM.

All metrics here share a single forward pass: the Runner runs the model on
the test loader once, caches y_true / y_pred tensors in EvalContext, and
all five metrics read from the cache.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from csi_eval.progress import progress

from ..core.context import EvalContext
from ..core.registries import MetricRegistry
from ._math import correlation_ri, evm_ri, mse_ri, nmse_db_ri, sgcs_ri


def _ensure_predictions(ctx: EvalContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model once on every requested split and cache (y_true, y_pred)."""
    if ctx.get("yhat") is not None and ctx.get("ytrue") is not None:
        return ctx.get("ytrue"), ctx.get("yhat")

    ys, yhats = [], []
    ctx.model.eval()
    for split in progress(ctx.splits, desc="feedback task: splits", unit="split", leave=False):
        loader = ctx.data.get_loader(split, batch_size=128, num_workers=0, shuffle=False)
        with torch.no_grad():
            for batch in progress(
                loader, total=len(loader) if hasattr(loader, '__len__') else None,
                desc=f"feedback task: {split} inference", unit="batch", leave=False
            ):
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(ctx.device, non_blocking=True).float()
                yhat = ctx.model.forward(x)
                ys.append(x.detach().cpu())
                yhats.append(yhat.detach().cpu())
    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(yhats, dim=0)
    ctx.set("ytrue", y_true)
    ctx.set("yhat", y_pred)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# NMSE
# ---------------------------------------------------------------------------
@MetricRegistry.register("task_performance", higher_is_better=False)
class NMSE:
    """Normalized Mean Squared Error in dB (lower is better)."""

    name = "nmse"
    category = "task_performance"
    higher_is_better = False
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        y_true, y_pred = _ensure_predictions(ctx)
        stats = nmse_db_ri(y_true, y_pred)
        return {"value": stats["mean"], "stats": stats, "unit": "dB"}


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------
@MetricRegistry.register("task_performance", higher_is_better=False)
class MSE:
    """Per-sample Mean Squared Error (lower is better)."""

    name = "mse"
    category = "task_performance"
    higher_is_better = False
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        y_true, y_pred = _ensure_predictions(ctx)
        stats = mse_ri(y_true, y_pred)
        return {"value": stats["mean"], "stats": stats}


# ---------------------------------------------------------------------------
# Correlation (rho)
# ---------------------------------------------------------------------------
@MetricRegistry.register("task_performance", higher_is_better=True)
class CosineRho:
    """Non-squared complex cosine similarity (higher is better, [0, 1])."""

    name = "rho"
    category = "task_performance"
    higher_is_better = True
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        y_true, y_pred = _ensure_predictions(ctx)
        stats = correlation_ri(y_true, y_pred)
        return {"value": stats["mean"], "stats": stats}


# ---------------------------------------------------------------------------
# SGCS
# ---------------------------------------------------------------------------
@MetricRegistry.register("task_performance", higher_is_better=True)
class SGCS:
    """Per-subband Squared Generalized Cosine Similarity (higher is better, [0, 1])."""

    name = "sgcs"
    category = "task_performance"
    higher_is_better = True
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        y_true, y_pred = _ensure_predictions(ctx)
        stats = sgcs_ri(y_true, y_pred)
        return {"value": stats["mean"], "stats": stats}


# ---------------------------------------------------------------------------
# EVM
# ---------------------------------------------------------------------------
@MetricRegistry.register("task_performance", higher_is_better=False)
class EVM:
    """Error Vector Magnitude in percent (lower is better)."""

    name = "evm"
    category = "task_performance"
    higher_is_better = False
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        y_true, y_pred = _ensure_predictions(ctx)
        stats = evm_ri(y_true, y_pred)
        return {"value": stats["mean_pct"], "stats": stats, "unit": "%"}

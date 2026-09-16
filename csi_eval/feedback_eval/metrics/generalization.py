"""Generalization metrics: 3GPP Rel-19 Cases 1 / 2 / 3.

3GPP TR 38.843 V19.0.0 (2025-09) Section 6.2 defines:

  Baseline (Case 1):      Train on Part1 0-800 → test on Part1 900-1000.
  Cross-scenario Zero-Shot (Case 2):
                           Train on Part1 0-800 →
                           zero-shot test on Part1_NEW/Part2.
  Cross-scenario Fine-Tune (Case 3):
                           Train on Part1 0-800 →
                           fine-tune on Part1_NEW/Part2 →
                           test on Part1_NEW/Part2.

Aggregation: per-map mean (average of per-environment means).
The Cross-Scenario Evaluation uses 2-bit quantization (enforced by Evaluator).

Metrics per target:
  id_baseline — Case 1 in-distribution baseline (per-map mean NMSE, dB)
  zero_shot   — Case 2 cross-scenario zero-shot (per-map mean NMSE, dB)
  fine_tune   — Case 3 cross-scenario fine-tune (gap recovery rate, %)
  gap_nmse    — Case 2 − Case 1 gap in dB (per-map mean)
  sgcs_decay_rate — (SGCS(Zero-Shot) − SGCS(Baseline)) / SGCS(Baseline)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from csi_eval.progress import progress

from ..core.context import EvalContext
from ..core.registries import MetricRegistry


# ---------------------------------------------------------------------------
# Helper loaders
# ---------------------------------------------------------------------------

def _make_loader(data, split, batch_size=128, num_workers=4):
    return data.get_loader(split, batch_size=batch_size,
                          num_workers=num_workers, shuffle=False)


def _cache_key(name: str, ctx: EvalContext) -> str:
    ood = getattr(ctx, "ood_data", None)
    if ood is not None:
        return f"{name}::ood::{id(ood)}"
    return f"{name}::id::{id(ctx.data)}"


# ---------------------------------------------------------------------------
# Per-map evaluator (robust generalization aggregation)
# ---------------------------------------------------------------------------

def _per_map_eval(
    loader: DataLoader,
    model: torch.nn.Module,
    device: str,
    env_index_array_array: Any,
) -> Dict[str, Any]:
    """Evaluate model on a loader and compute per-map mean NMSE/SGCS.

    1. Collect per-sample NMSE and SGCS.
    2. Group samples by their environment (map) index.
    3. Average per environment → per-map mean NMSE/SGCS.
    4. Also compute the pooled (global) values for reference.

    env_index_array_array: np.ndarray of shape [n_samples], one environment ID per sample.
    """
    model.eval()
    err_list: List[float] = []
    pwr_list: List[float] = []
    sgcs_list: List[float] = []

    with torch.no_grad():
        for batch in progress(
            loader, total=len(loader) if hasattr(loader, '__len__') else None,
            desc="feedback generalization: evaluate", unit="batch", leave=False
        ):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True).float()
            yhat = model.forward(x)
            bs = x.shape[0]

            # SGCS per sample
            if x.shape[2] > x.shape[3]:
                wc = torch.complex(x[:, 0], x[:, 1])
                wp = torch.complex(yhat[:, 0], yhat[:, 1])
            else:
                wc = torch.complex(x[:, :, 0], x[:, :, 1])
                wp = torch.complex(yhat[:, :, 0], yhat[:, :, 1])
            inner = torch.sum(torch.conj(wc) * wp, dim=1)
            n_t = torch.sum(torch.abs(wc) ** 2, dim=1)
            n_p = torch.sum(torch.abs(wp) ** 2, dim=1)
            rho2 = (torch.abs(inner) ** 2 / (n_t * n_p + 1e-12)).clamp(0, 1).mean(dim=1)

            err = ((x - yhat) ** 2).flatten(1).sum(dim=1)
            pwr = (x ** 2).flatten(1).sum(dim=1).clamp_min(1e-12)

            err_list.extend(err.cpu().tolist())
            pwr_list.extend(pwr.cpu().tolist())
            sgcs_list.extend(rho2.cpu().tolist())

    err_arr = np.array(err_list, dtype=np.float64)
    pwr_arr = np.array(pwr_list, dtype=np.float64)
    sgcs_arr = np.array(sgcs_list, dtype=np.float64)

    # Resolve environment indices
    if env_index_array_array is not None:
        try:
            ei = np.asarray(env_index_array_array, dtype=np.int32)
            # Trim/pad to match actual sample count
            if len(ei) >= len(err_arr):
                ei = ei[:len(err_arr)]
            else:
                # Fall back: all same env
                ei = np.zeros(len(err_arr), dtype=np.int32)
        except Exception:
            ei = np.zeros(len(err_arr), dtype=np.int32)
    else:
        ei = np.zeros(len(err_arr), dtype=np.int32)

    unique_envs = sorted(set(ei.tolist()))
    per_map_nmse_db: List[float] = []
    per_map_sgcs: List[float] = []
    for e in unique_envs:
        mask = [i for i, eid in enumerate(ei) if eid == e]
        e_err = sum(err_arr[mask])
        e_pwr = sum(pwr_arr[mask])
        e_nmse_db = float(10.0 * np.log10(max(e_err / (e_pwr + 1e-12), 1e-12)))
        e_sgcs = float(np.mean(sgcs_arr[mask]))
        per_map_nmse_db.append(e_nmse_db)
        per_map_sgcs.append(e_sgcs)

    # Per-map mean (primary metric for generalization)
    mean_nmse_db = float(np.mean(per_map_nmse_db))
    std_nmse_db = float(np.std(per_map_nmse_db, ddof=0))
    mean_sgcs = float(np.mean(per_map_sgcs))
    std_sgcs = float(np.std(per_map_sgcs, ddof=0))

    # Pooled (global) for reference
    total_err = float(np.sum(err_arr))
    total_pwr = float(np.sum(pwr_arr))
    pooled_nmse_db = float(10.0 * np.log10(max(total_err / (total_pwr + 1e-12), 1e-12)))
    pooled_sgcs = float(np.mean(sgcs_arr))

    return {
        "per_map_nmse_db": per_map_nmse_db,
        "per_map_sgcs": per_map_sgcs,
        "mean_nmse_db": mean_nmse_db,
        "std_nmse_db": std_nmse_db,
        "mean_sgcs": mean_sgcs,
        "std_sgcs": std_sgcs,
        "pooled_nmse_db": pooled_nmse_db,
        "pooled_sgcs": pooled_sgcs,
        "n_maps": len(unique_envs),
        "n_samples": len(err_arr),
    }


# ---------------------------------------------------------------------------
# Case 1 — In-Distribution Baseline (per-map mean)
# ---------------------------------------------------------------------------

@MetricRegistry.register("robustness", requires=frozenset(), higher_is_better=False)
class IDBaseline:
    """3GPP Case 1: in-distribution baseline (per-map mean aggregation).

    Train on Part1 0-800 → test on Part1 900-1000.
    Aggregation: per-map mean NMSE/SGCS (average of per-environment means).
    This is the reference value for Case 2 and Case 3 comparisons.
    """

    name = "id_baseline"
    category = "robustness"
    higher_is_better = False
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        # Evaluate on ID test data with per-map aggregation
        data = ctx.data
        loader = _make_loader(data, "test", batch_size=ctx.config.latency_runs)
        env_idx = getattr(data, "env_index_array", None)
        result = ctx.get_or_compute(
            _cache_key("case1", ctx),
            lambda: _per_map_eval(loader, ctx.model, ctx.device, env_idx),
        )
        return {
            "value": result["mean_nmse_db"],
            "unit": "dB",
            "nmse_db": result["mean_nmse_db"],
            "sgcs": result["mean_sgcs"],
            "pooled_nmse_db": result["pooled_nmse_db"],
            "pooled_sgcs": result["pooled_sgcs"],
            "std_nmse_db": result["std_nmse_db"],
            "std_sgcs": result["std_sgcs"],
            "n_maps": result["n_maps"],
            "n_samples": result["n_samples"],
        }


# ---------------------------------------------------------------------------
# Case 2 — Cross-Scenario Zero-Shot (per-map mean)
# ---------------------------------------------------------------------------

@MetricRegistry.register("robustness",
                         requires=frozenset(),
                         higher_is_better=False)
class ZeroShot:
    """3GPP Case 2: cross-scenario zero-shot (per-map mean aggregation).

    Train on Part1 0-800 → zero-shot test on Part1_NEW/Part2.
    No fine-tuning; the model never sees the OOD target during training.
    Aggregation: per-map mean NMSE/SGCS.
    """

    name = "zero_shot"
    category = "robustness"
    higher_is_better = False
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        ood = getattr(ctx, "ood_data", None)
        if ood is None:
            return {"value": None, "skipped": "no OOD target"}

        split = getattr(ood, "_split", "ood")
        loader = _make_loader(ood, split, batch_size=ctx.config.latency_runs)
        env_idx = getattr(ood, "env_index_array", None)
        result = ctx.get_or_compute(
            _cache_key("case2", ctx),
            lambda: _per_map_eval(loader, ctx.model, ctx.device, env_idx),
        )
        return {
            "value": result["mean_nmse_db"],
            "unit": "dB",
            "nmse_db": result["mean_nmse_db"],
            "sgcs": result["mean_sgcs"],
            "pooled_nmse_db": result["pooled_nmse_db"],
            "pooled_sgcs": result["pooled_sgcs"],
            "std_nmse_db": result["std_nmse_db"],
            "std_sgcs": result["std_sgcs"],
            "n_maps": result["n_maps"],
            "n_samples": result["n_samples"],
            "per_map_nmse_db": result["per_map_nmse_db"],
            "per_map_sgcs": result["per_map_sgcs"],
            "split": split,
        }


# ---------------------------------------------------------------------------
# Case 3 — Cross-Scenario Fine-Tune (per-map mean)
# ---------------------------------------------------------------------------

@MetricRegistry.register("robustness", requires=frozenset(), higher_is_better=True)
class FineTune:
    """3GPP Case 3: cross-scenario fine-tune (per-map mean aggregation).

    Train on Scenario#A/Config#A → fine-tune on n_support samples from
    Scenario#B/Config#B → evaluate on FULL Scenario#B/Config#B split
    (per-map mean NMSE/SGCS).

    For each n_support in [0, 50, 100, 300, 500] (0 = zero-shot baseline):
      - Fine-tune model on n_support samples (AdamW, MSE loss)
      - Evaluate on FULL target split (per-map mean)
      - Report per-map NMSE, SGCS, and closed_gap_pct

    closed_gap_pct = -100 * (NMSE_fs - NMSE_c2) / (NMSE_c2 - NMSE_c1)
    (positive = recovering gap toward Case 1)
    """

    name = "fine_tune"
    category = "robustness"
    higher_is_better = True
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        ood = getattr(ctx, "ood_data", None)
        if ood is None:
            return {"value": None, "skipped": "no OOD target"}

        result = ctx.get_or_compute(
            _cache_key("case3", ctx),
            lambda: self._do(ctx),
        )
        best_closed = result.get("best_closed_gap_pct", 0.0)
        return {
            "value": best_closed,
            "unit": "%",
            **result,
        }

    @staticmethod
    def _do(ctx: EvalContext) -> Dict[str, Any]:
        ood = getattr(ctx, "ood_data", None)
        device = ctx.device
        model = ctx.model
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        ood_split = getattr(ood, "_split", "ood")
        samples = ctx.config.fewshot_samples or [0, 5, 10, 20, 50, 100, 300]
        fewshot_epochs = ctx.config.fewshot_epochs or 30
        fewshot_lr = ctx.config.fewshot_lr or 1e-4

        try:
            # Case 1 (ID baseline — per-map mean)
            data = ctx.data
            id_loader = _make_loader(data, "test", batch_size=128)
            id_env_idx = getattr(data, "env_index_array", None)
            id_result = ctx.get_or_compute(
                _cache_key("case1", ctx),
                lambda: _per_map_eval(id_loader, model, device, id_env_idx),
            )
            c1_nmse = id_result["mean_nmse_db"]
            c1_sgcs = id_result["mean_sgcs"]

            # Case 2 (OOD zero-shot — per-map mean)
            ood_loader = _make_loader(ood, ood_split, batch_size=128)
            ood_env_idx = getattr(ood, "env_index_array", None)
            c2_result = ctx.get_or_compute(
                _cache_key("case2", ctx),
                lambda: _per_map_eval(ood_loader, model, device, ood_env_idx),
            )
            c2_nmse = c2_result["mean_nmse_db"]
            c2_sgcs = c2_result["mean_sgcs"]

            gap_total = c2_nmse - c1_nmse  # positive → OOD is harder

            per_n: List[Dict[str, Any]] = []

            # Include n_support=0 as the zero-shot baseline point
            per_n.append({
                "n_support": 0,
                "nmse_db": c2_nmse,
                "sgcs": c2_sgcs,
                "closed_gap_pct": 0.0,
                "is_zeroshot": True,
            })

            for n_support in [s for s in samples if s > 0]:
                # Fine-tune
                optim = torch.optim.Adam(model.parameters(), lr=fewshot_lr)
                model.train()
                n_seen = 0
                max_steps = fewshot_epochs * 10

                for _step in range(max_steps):
                    for batch in _make_loader(ood, ood_split, batch_size=128, num_workers=0):
                        if n_seen >= n_support:
                            break
                        x = batch[0] if isinstance(batch, (tuple, list)) else batch
                        x = x.to(device, non_blocking=True).float()
                        optim.zero_grad()
                        loss = torch.nn.functional.mse_loss(model(x), x)
                        loss.backward()
                        optim.step()
                        n_seen += x.shape[0]
                    if n_seen >= n_support:
                        break

                # Evaluate on FULL target split (per-map mean)
                eval_result = _per_map_eval(ood_loader, model, device, ood_env_idx)
                fs_nmse = eval_result["mean_nmse_db"]
                fs_sgcs = eval_result["mean_sgcs"]

                closed = 0.0
                if gap_total != 0.0 and (gap_total == gap_total):  # not NaN
                    closed = max(0.0, min(100.0,
                        -100.0 * (c2_nmse - fs_nmse) / gap_total))

                per_n.append({
                    "n_support": n_support,
                    "nmse_db": fs_nmse,
                    "sgcs": fs_sgcs,
                    "closed_gap_pct": closed,
                    "is_zeroshot": False,
                })

                # Restore for next iteration
                for k, v in original_state.items():
                    model.state_dict()[k].copy_(v)

            best = max(per_n, key=lambda r: r["closed_gap_pct"])

            return {
                "per_n": per_n,
                "case1_nmse_db": c1_nmse,
                "case1_sgcs": c1_sgcs,
                "case2_nmse_db": c2_nmse,
                "case2_sgcs": c2_sgcs,
                "best_nmse_db": best["nmse_db"],
                "best_sgcs": best["sgcs"],
                "best_closed_gap_pct": best["closed_gap_pct"],
                "unit": "%",
            }
        finally:
            model.load_state_dict(original_state)


# ---------------------------------------------------------------------------
# Generalization Gap (Case 2 − Case 1, per-map mean)
# ---------------------------------------------------------------------------

@MetricRegistry.register("robustness", requires=frozenset(), higher_is_better=False)
class GapNMSE:
    """3GPP Case 2 − Case 1 gap in dB (per-map mean NMSE).

    gap = Case2_per_map_mean_NMSE(dB) − Case1_per_map_mean_NMSE(dB)

    Positive gap  → Case 2 (OOD) is harder than Case 1 (ID).
    Negative gap  → Case 2 (OOD) is easier than Case 1 (ID).
    Zero gap     → Perfect cross-scenario generalization.
    """

    name = "gap_nmse"
    category = "robustness"
    higher_is_better = False
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        ood = getattr(ctx, "ood_data", None)
        if ood is None:
            return {"value": None, "skipped": "no OOD target"}

        device = ctx.device
        model = ctx.model

        # Case 1 (ID baseline — per-map mean)
        data = ctx.data
        id_loader = _make_loader(data, "test", batch_size=128)
        id_env_idx = getattr(data, "env_index_array", None)
        id_result = ctx.get_or_compute(
            _cache_key("case1", ctx),
            lambda: _per_map_eval(id_loader, model, device, id_env_idx),
        )
        c1_nmse = id_result["mean_nmse_db"]
        c1_sgcs = id_result["mean_sgcs"]

        # Case 2 (OOD zero-shot — per-map mean)
        ood_split = getattr(ood, "_split", "ood")
        ood_loader = _make_loader(ood, ood_split, batch_size=ctx.config.latency_runs)
        ood_env_idx = getattr(ood, "env_index_array", None)
        c2_result = ctx.get_or_compute(
            _cache_key("case2", ctx),
            lambda: _per_map_eval(ood_loader, model, device, ood_env_idx),
        )
        c2_nmse = c2_result["mean_nmse_db"]
        c2_sgcs = c2_result["mean_sgcs"]

        gap_db = float(c2_nmse - c1_nmse)
        return {
            "value": gap_db,
            "unit": "dB",
            "case1_nmse_db": c1_nmse,
            "case1_sgcs": c1_sgcs,
            "case2_nmse_db": c2_nmse,
            "case2_sgcs": c2_sgcs,
        }


# ---------------------------------------------------------------------------
# SGCS Decay Rate (Case 2 − Case 1 relative SGCS drop)
# ---------------------------------------------------------------------------

@MetricRegistry.register("robustness", requires=frozenset(), higher_is_better=False)
class SGCSDecayRate:
    """SGCS decay rate: (SGCS(Zero-Shot) - SGCS(Baseline)) / SGCS(Baseline).

    Measures how much the SGCS metric degrades under cross-scenario zero-shot
    evaluation relative to the in-distribution baseline.

    Lower is better (small drop = good generalization).
    Positive value means SGCS increased (unlikely but not clamped).
    """

    name = "sgcs_decay_rate"
    category = "robustness"
    higher_is_better = False
    requires = frozenset()
    unit = ""

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        ood = getattr(ctx, "ood_data", None)
        if ood is None:
            return {"value": None, "skipped": "no OOD target"}

        device = ctx.device
        model = ctx.model

        # Case 1 (ID baseline — per-map mean)
        data = ctx.data
        id_loader = _make_loader(data, "test", batch_size=128)
        id_env_idx = getattr(data, "env_index_array", None)
        id_result = ctx.get_or_compute(
            _cache_key("case1", ctx),
            lambda: _per_map_eval(id_loader, model, device, id_env_idx),
        )
        c1_sgcs = id_result["mean_sgcs"]

        # Case 2 (OOD zero-shot — per-map mean)
        ood_split = getattr(ood, "_split", "ood")
        ood_loader = _make_loader(ood, ood_split, batch_size=ctx.config.latency_runs)
        ood_env_idx = getattr(ood, "env_index_array", None)
        c2_result = ctx.get_or_compute(
            _cache_key("case2", ctx),
            lambda: _per_map_eval(ood_loader, model, device, ood_env_idx),
        )
        c2_sgcs = c2_result["mean_sgcs"]

        if c1_sgcs > 0:
            decay = float((c2_sgcs - c1_sgcs) / c1_sgcs)
        else:
            decay = float("nan")

        return {
            "value": decay,
            "case1_sgcs": c1_sgcs,
            "case2_sgcs": c2_sgcs,
            "unit": "",
        }

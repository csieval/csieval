"""Runners: glue between Metrics and EvalContext.

A Runner walks through all metrics in one category, probes capabilities,
skips what the model cannot provide, and feeds the result into the
EvalReport.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from csi_eval.progress import progress

from ..core.config import EvalConfig
from ..core.context import EvalContext
from ..core.registries import MetricRegistry
from ..core.report import EvalReport, MetricRecord


class Runner:
    """Base class: walks one metric category and fills in records."""

    category: str = "unknown"

    def __init__(self, config: EvalConfig, ctx: EvalContext):
        self.config = config
        self.ctx = ctx

    def _metric_names(self) -> List[str]:
        return [m.name for m in MetricRegistry.list_by_category(self.category)]

    def run(
        self,
        report: Optional[EvalReport] = None,
        only: Optional[set] = None,
        into: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Walk every metric in the category and feed results.

        Parameters
        ----------
        report:
            Where MetricRecord entries are appended. Ignored if
            ``into`` is supplied.
        only:
            Optional set of metric names to keep; metrics not in this
            set are reported as "skipped (filtered)".
        into:
            Optional dict to populate with ``{metric_name: value}``
            instead of (or in addition to) a full EvalReport. Used by
            ``Evaluator._run_ood_target`` to build per-target sub-reports.
        """
        if report is None and into is None:
            report = EvalReport(config=self.config)
        metric_names = self._metric_names()
        for name in progress(
            metric_names, total=len(metric_names),
            desc=f"feedback {self.category}: metrics", unit="metric", leave=False
        ):
            metric = MetricRegistry.get(name)
            if only is not None and metric.name not in only:
                if report is not None:
                    report.add_skipped(metric.name, "filtered out by user")
                continue
            if not self._has_all_capabilities(metric):
                missing = {cap for cap in metric.requires if not self.ctx.has(cap)}
                msg = self._skip_reason(metric, missing)
                if report is not None:
                    report.add_skipped(metric.name, msg)
                continue
            try:
                result = metric.compute(self.ctx)
            except Exception as e:  # pragma: no cover
                import traceback
                traceback.print_exc()
                if report is not None:
                    report.add_skipped(metric.name, f"compute failed: {e}")
                if into is not None:
                    into[metric.name] = {"value": None, "error": str(e)}
                continue
            value = result.get("value", result)
            unit = result.get("unit", "")
            note = result.get("note", "")
            if into is not None:
                into[metric.name] = result if isinstance(result, dict) else {"value": result}
                continue
            if report is not None:
                report.add(
                    MetricRecord(
                        name=metric.name,
                        category=metric.category,
                        value=value,
                        higher_is_better=metric.higher_is_better,
                        unit=unit,
                        note=note,
                    )
                )
                extras = {k: v for k, v in result.items()
                          if k not in ("value", "unit", "note")}
                for ek, ev in extras.items():
                    report.add_sub(f"{self.category}.{metric.name}.{ek}", ev)

    def _has_all_capabilities(self, metric) -> bool:
        for cap in metric.requires:
            if not self.ctx.has(cap):
                return False
        return True

    def _skip_reason(self, metric, missing_caps: set) -> str:
        return f"missing required capabilities: {sorted(missing_caps)}"


# ---------------------------------------------------------------------------
# TaskRunner: 任务性能
# ---------------------------------------------------------------------------

class TaskRunner(Runner):
    category = "task_performance"


# ---------------------------------------------------------------------------
# StorageRunner: 部署存储
# ---------------------------------------------------------------------------

class StorageRunner(Runner):
    category = "storage"


# ---------------------------------------------------------------------------
# ComputationRunner: 计算效能
# ---------------------------------------------------------------------------

class ComputationRunner(Runner):
    category = "computation"


# ---------------------------------------------------------------------------
# RobustnessRunner: 鲁棒泛化（合并了 robustness 和 generalization）
#
# The "robustness" category now includes both noise/quantization robustness
# metrics and cross-scenario generalization metrics.
#
# Generalization metrics that need an OOD target return None when no OOD
# target is active; the OOD pass recomputes them with actual values.
# ---------------------------------------------------------------------------

_OOD_METRICS = {
    "zero_shot",
    "fine_tune",
    "gap_nmse",
    "sgcs_decay_rate",
}


class RobustnessRunner(Runner):
    category = "robustness"

    def _has_ood(self) -> bool:
        return getattr(self.ctx, "ood_data", None) is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_default_runners(config: EvalConfig, ctx: EvalContext) -> List[Runner]:
    return [
        TaskRunner(config, ctx),
        StorageRunner(config, ctx),
        ComputationRunner(config, ctx),
        RobustnessRunner(config, ctx),
    ]


__all__ = [
    "Runner",
    "TaskRunner",
    "StorageRunner",
    "ComputationRunner",
    "RobustnessRunner",
    "build_default_runners",
]

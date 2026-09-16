"""EvalReport: structured result of an evaluation run.

Accumulates MetricRecord entries from each Runner. Supports:
- save("json" | "html" | "markdown")
- print_summary()
- compare(other_reports, fmt) for leaderboards
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import EvalConfig


@dataclass
class MetricRecord:
    """One computed metric value."""

    name: str
    category: str
    value: Any
    higher_is_better: bool = True
    unit: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
            "unit": self.unit,
            "note": self.note,
        }


@dataclass
class EvalReport:
    """Aggregated evaluation report.

    Attributes:
        config: The EvalConfig used.
        records: List of MetricRecord produced by all Runners.
        meta: Free-form metadata (timestamp, model size, device, ...).
        skipped: Metrics that were skipped (with reasons).
        sub_results: Runner-specific nested results (e.g. per-SNR dicts).
    """

    config: EvalConfig
    records: List[MetricRecord] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    sub_results: Dict[str, Any] = field(default_factory=dict)

    # ---- Aggregation helpers ----
    def add(self, record: MetricRecord) -> None:
        self.records.append(record)

    def add_skipped(self, name: str, reason: str) -> None:
        self.skipped.append({"name": name, "reason": reason})

    def add_sub(self, key: str, value: Any) -> None:
        self.sub_results[key] = value

    def add_sub_dict(self, key: str) -> Dict[str, Any]:
        """Return a sub-dict stored under ``key``, creating it on demand."""
        cur = self.sub_results.get(key)
        if not isinstance(cur, dict):
            cur = {}
            self.sub_results[key] = cur
        return cur

    def get(self, name: str, default: Any = None) -> Any:
        for r in self.records:
            if r.name == name:
                return r.value
        return default

    def __getitem__(self, key: str) -> Any:
        """Dict-like access.

        * Plain metric name: returns the metric value
          (``report["nmse"] == report.get("nmse")``).
        * "category/metric": returns the metric from that category.
        * "ood/<name>::<metric>" or "ood/<name>/<metric>": returns the
          per-OOD-target metric recorded under ``sub_results["ood"]``.
        """
        if "::" in key:
            prefix, name = key.split("::", 1)
            if prefix == "ood":
                ood = self.sub_results.get("ood", {})
                for tgt_name, sub in ood.items():
                    if tgt_name == name or sub.get("target", {}).get("name") == name:
                        return sub
                return None
        if "/" in key:
            head, tail = key.split("/", 1)
            if head == "ood":
                # ood/<name>/<metric> -> sub_results["ood"][<name>]["metrics"][<metric>]
                parts = tail.split("/", 1)
                if len(parts) == 2:
                    tgt_name, metric = parts
                    return self.sub_results.get("ood", {}).get(tgt_name, {}).get("metrics", {}).get(metric)
            if head in {"task_performance", "storage", "computation", "robustness"}:
                for r in self.records:
                    if r.category == head and r.name == tail:
                        return r.value
                return None
        return self.get(key, default=None)

    def filter(self, category: Optional[str] = None) -> "EvalReport":
        """Return a shallow-copied report restricted to one category."""
        if category is None:
            return self
        new = EvalReport(
            config=self.config,
            meta=dict(self.meta),
            skipped=list(self.skipped),
            sub_results=dict(self.sub_results),
        )
        for r in self.records:
            if r.category == category:
                new.records.append(r)
        return new

    def to_pandas(self):
        """Return a ``pandas.DataFrame`` with one row per metric record.

        Falls back to printing a tabular representation if pandas is
        not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            print("pandas not installed; printing as table instead.")
            self.print_summary()
            return None
        rows = []
        for r in self.records:
            rows.append({
                "name": r.name,
                "category": r.category,
                "value": r.value,
                "unit": r.unit,
                "higher_is_better": r.higher_is_better,
                "note": r.note,
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "meta": self.meta,
            "metrics": [r.to_dict() for r in self.records],
            "skipped": self.skipped,
            "sub_results": self.sub_results,
        }

    # ---- Output ----
    def save(self, fmt: str = "json", output_dir: Optional[str] = None) -> List[Path]:
        if output_dir is None:
            output_dir = self.config.output_dir
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        from ..reports import json_report, html_report, markdown_report  # local import

        if fmt == "json":
            paths.append(json_report.save(self, out_dir))
        elif fmt == "html":
            paths.append(html_report.save(self, out_dir))
        elif fmt == "markdown" or fmt == "md":
            paths.append(markdown_report.save(self, out_dir))
        else:
            raise ValueError(f"Unknown report format: {fmt}")
        return paths

    def print_summary(
        self,
        categories: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Pretty-print the report.

        Parameters
        ----------
        categories:
            If given, only print records from these categories.
        metrics:
            If given, only print records whose name is in this list.
        Both filters are AND-combined.
        """
        print("=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        for cat in ("task_performance", "storage", "computation", "robustness"):
            if categories and cat not in categories:
                continue
            cat_records = [
                r for r in self.records
                if r.category == cat
                and (metrics is None or r.name in metrics)
            ]
            if not cat_records:
                continue
            print(f"\n--- {cat} ---")
            for r in cat_records:
                arrow = "↑" if r.higher_is_better else "↓"
                print(f"  [{arrow}] {r.name}: {r.value} {r.unit}".rstrip())

        if self.skipped and (not metrics or any(s["name"] in metrics for s in self.skipped)):
            print(f"\n--- Skipped ({len(self.skipped)}) ---")
            for s in self.skipped:
                print(f"  - {s['name']}: {s['reason']}")

        ood = self.sub_results.get("ood")
        if ood and (categories is None or "generalization" in categories):
            print("\n--- Cross-scenario (OOD) ---")
            for tgt_name, sub in ood.items():
                tgt_meta = sub.get("target", {})
                desc = tgt_meta.get("description") or f"path={tgt_meta.get('path')}"
                print(f"\n  {tgt_name}  ({desc})")
                m = sub.get("metrics", {})
                if not m:
                    print(f"    (no metrics; error={sub.get('error')})")
                    continue
                for name, val in m.items():
                    if metrics is not None and name not in metrics:
                        continue
                    print(f"    {name}: {val}")

    # ---- Comparison ----
    @classmethod
    def compare(
        cls,
        reports: List["EvalReport"],
        metric_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        title: str = "Model Comparison",
    ) -> "EvalReport":
        """Build a leaderboard EvalReport from multiple reports.

        Each row is one report (one model). Each column is one metric.
        """
        if not reports:
            raise ValueError("compare() needs at least one report")

        # Pick the union of metric names if not specified
        if metric_names is None:
            names: List[str] = []
            for r in reports:
                for rec in r.records:
                    if rec.name not in names:
                        names.append(rec.name)
            metric_names = names

        leaderboard = cls(
            config=reports[0].config,
            records=[],
            meta={
                "title": title,
                "n_reports": len(reports),
                "config_tags": [r.meta.get("tag", r.meta.get("model_name", f"report_{i}"))
                                for i, r in enumerate(reports)],
            },
        )
        # Each report's model name becomes one MetricRecord per metric
        for i, r in enumerate(reports):
            tag = r.meta.get("tag", r.meta.get("model_name", f"report_{i}"))
            for name in metric_names:
                value = r.get(name, default=None)
                if value is None:
                    continue
                leaderboard.records.append(
                    MetricRecord(
                        name=f"{tag}::{name}",
                        category="comparison",
                        value=value,
                        higher_is_better=any(
                            rec.higher_is_better for rec in r.records if rec.name == name
                        ),
                        unit=next(
                            (rec.unit for rec in r.records if rec.name == name), ""
                        ),
                        note=f"source=report_{i}",
                    )
                )

        if output_dir:
            for fmt in ("json", "html"):
                leaderboard.save(fmt, output_dir=output_dir)
        return leaderboard

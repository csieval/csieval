"""JSON report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..core.report import EvalReport


def save(report: EvalReport, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=_default)
    return out_path


def _default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return list(o)
    return str(o)

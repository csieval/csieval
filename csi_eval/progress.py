"""Lightweight progress-bar utilities used by both CSI evaluation subpackages.

Progress bars are enabled by default and can be disabled by setting
``CSI_EVAL_PROGRESS=0`` (also accepts false/no/off).  ``tqdm`` is optional at
runtime: if it is unavailable the evaluation still works, only without bars.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")

try:  # pragma: no cover - fallback is intentionally dependency-safe
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def progress_enabled() -> bool:
    """Return whether CSI evaluation progress bars should be displayed."""
    value = os.environ.get("CSI_EVAL_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: str = "it",
    leave: bool = False,
    enabled: Optional[bool] = None,
    **kwargs,
):
    """Wrap *iterable* in a ``tqdm`` progress bar when available.

    The wrapper centralizes progress behavior so long-running loops do not need
    to depend directly on tqdm.  Existing callers can therefore run in minimal
    environments without changing evaluation semantics.
    """
    show = progress_enabled() if enabled is None else bool(enabled)
    if _tqdm is None or not show:
        return iterable
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        **kwargs,
    )

"""Core abstractions: protocols, evaluator, config, registries."""

from .config import EvalConfig
from .context import EvalContext
from .report import EvalReport, MetricRecord
from .evaluator import Evaluator

__all__ = [
    "EvalConfig",
    "EvalContext",
    "EvalReport",
    "MetricRecord",
    "Evaluator",
]

"""Metrics package: pluggable metric implementations.

All concrete metrics live in this package. Each metric is registered
via the @MetricRegistry.register decorator. The framework auto-discovers
all registered metrics and dispatches them at run time.

Categories:
  - task_performance
  - efficiency
  - computation
  - robustness
  - generalization
"""

from ..core.registries import MetricRegistry

# Trigger registration of all built-in metrics via side-effect imports.
from . import task_performance        # noqa: F401
from . import efficiency              # noqa: F401
from . import computation             # noqa: F401
from . import robustness              # noqa: F401
from . import generalization          # noqa: F401

__all__ = ["MetricRegistry"]

"""Registries: TaskRegistry and MetricRegistry.

Users register new tasks/metrics via decorators; the framework auto-discovers
all registered items and dispatches accordingly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Optional, Type

from .protocols import Metric, TaskAdapter


# ---------------------------------------------------------------------------
# TaskRegistry
# ---------------------------------------------------------------------------

class TaskRegistry:
    """Registry of TaskAdapter implementations, keyed by task name."""

    _registry: Dict[str, Type[TaskAdapter]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[TaskAdapter]], Type[TaskAdapter]]:
        def decorator(task_cls: Type[TaskAdapter]) -> Type[TaskAdapter]:
            if not hasattr(task_cls, "name"):
                task_cls.name = name  # type: ignore[attr-defined]
            cls._registry[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TaskAdapter:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown task '{name}'. Available: {sorted(cls._registry)}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_tasks(cls) -> List[str]:
        return sorted(cls._registry)


# ---------------------------------------------------------------------------
# MetricRegistry
# ---------------------------------------------------------------------------

class MetricRegistry:
    """Registry of Metric implementations, keyed by metric class name.

    The framework supports two ways to add a metric:
    1. Decorator: @MetricRegistry.register("category", requires=..., higher_is_better=...)
    2. Subclass and call MetricRegistry.register_class(cls, "category", ...)
    """

    _registry: Dict[str, Metric] = {}

    @classmethod
    def register(
        cls,
        category: str,
        requires: FrozenSet[str] = frozenset(),
        higher_is_better: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> Callable[[Any], Any]:
        """Decorator for metric classes.

        If ``higher_is_better`` is omitted, the class's own
        ``higher_is_better`` attribute (if defined) is respected.
        """
        def deco(metric_cls: Type[Metric]) -> Type[Metric]:
            # Only use the explicit arg; class attribute takes priority.
            cls_attr = getattr(metric_cls, "higher_is_better", None)
            if higher_is_better is None:
                hib = bool(cls_attr) if cls_attr is not None else True
            else:
                hib = bool(higher_is_better)
            # The class attribute ``name = "..."`` is the canonical user-
            # facing name (matches the keys in Evaluator.metrics() and
            # the only-string callers should pass to ``run(metrics=...)``).
            cls_attr_name = getattr(metric_cls, "name", None)
            resolved_name = name or cls_attr_name or metric_cls.__name__
            return cls.register_class(
                metric_cls,
                category=category,
                requires=requires,
                higher_is_better=hib,
                name=resolved_name,
            )
        return deco

    @classmethod
    def register_class(
        cls,
        metric_cls: Type[Metric],
        category: str,
        requires: FrozenSet[str] = frozenset(),
        higher_is_better: bool = True,
        name: Optional[str] = None,
    ) -> Type[Metric]:
        # Prefer the explicit ``name`` argument, then the class attribute
        # ``name = "..."`` (if defined), and finally fall back to the
        # class name. This keeps the user-facing name consistent with
        # the @register decorator shorthand.
        cls_attr_name = getattr(metric_cls, "name", None)
        key = name or cls_attr_name or metric_cls.__name__
        instance = metric_cls()
        instance.category = category
        instance.requires = frozenset(requires)
        instance.higher_is_better = higher_is_better
        instance.name = key
        cls._registry[key] = instance
        return metric_cls

    @classmethod
    def get(cls, name: str) -> Metric:
        if name not in cls._registry:
            raise KeyError(
                f"Metric '{name}' not registered. Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def list_by_category(cls, category: str) -> List[Metric]:
        return [m for m in cls._registry.values() if m.category == category]

    @classmethod
    def list_all(cls) -> List[Metric]:
        return list(cls._registry.values())

    @classmethod
    def clear(cls) -> None:
        """For tests: clear all registered metrics."""
        cls._registry.clear()

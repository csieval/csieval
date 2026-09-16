"""Evaluator: the framework's main facade.

Glues together:
  - TaskAdapter (data side)
  - ModelAdapter (model side, including duck-typed loaders)
  - Runners (4 of them: task, storage, computation, robustness)
  - EvalReport (output)

Construction is intentionally minimal. The minimum required is
``task``, ``model_bundle``, and ``data``; everything else has a sensible default.

Examples
--------
Minimal with bundle::

    from csi_eval.feedback_eval import Evaluator
    report = Evaluator(
        task="eigenvector_feedback",
        model_bundle="bundles/my_model/",
        data="data/Dataset/wair_d_output/2_6GHz",
    ).run()
    report.print_summary()
    print(report["sgcs"])  # access a single metric value

Minimal with legacy checkpoint::

    report = Evaluator(
        task="eigenvector_feedback",
        checkpoint="runs/best.pt",
        model="model/ev_csinet.py",
        data="data/Dataset/wair_d_output/2_6GHz",
    ).run()

Cross-scenario (auto-generates Part 1 NEW + Part 2 if missing)::

    cfg = EvalConfig(
        task="eigenvector_feedback",
        model_bundle="bundles/my_model/",
        data="data/Dataset/wair_d_output/2_6GHz",
    )
    cfg.add_ood_target("part1_new", ".../2_6GHz_part1_new",
                       auto_generate={"kind": "part1",
                                      "scenario_root": ".../scenario_1",
                                      "map_start": 1000, "map_count": 100})
    cfg.add_ood_target("part2", ".../2_6GHz_part2", scenario="II", split="all",
                       auto_generate={"kind": "part2",
                                      "scenario_root": ".../scenario_2",
                                      "sample_per_map": 500})
    report = Evaluator(cfg).run()
    report.print_summary()
    report["robustness/part1_new::gap_nmse"]  # per-OOD subresult
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import torch

from csi_eval.progress import progress

from .config import EvalConfig
from .context import EvalContext
from .registries import MetricRegistry, TaskRegistry
from .report import EvalReport
from .protocols import DataAdapter, ModelAdapter, TaskAdapter


MetricFilter = Union[None, str, List[str]]


class Evaluator:
    """The main entry point for the evaluation framework.

    Two construction styles are supported:

      1. Direct construction with model_bundle (recommended)::

            ev = Evaluator(
                task="eigenvector_feedback",
                model_bundle="bundles/my_model/",
                data="data/2_6GHz",
                output_dir="results/my_model",
            )

      2. From an existing ``EvalConfig`` (advanced usage)::

            cfg = EvalConfig(task=..., model_bundle=..., data=..., ...)
            ev = Evaluator(cfg)

    Then call ``ev.run(categories=None, metrics=None)`` to compute all
    requested metrics and get an ``EvalReport``.

    ``run()`` accepts two filters:

    - ``categories``: subset of the 5 high-level categories
      ("task_performance", "efficiency", "computation", "robustness",
      "robustness"). ``None`` or empty list = all categories.
    - ``metrics``: explicit list of metric *names* (e.g. ``["nmse",
      "sgcs", "case_gap"]``). When supplied, takes priority over
      ``categories`` — only these named metrics are computed, all others
      are skipped.

    If ``config.ood_targets`` is non-empty (set via
    ``EvalConfig.add_ood_target`` or the legacy ``ood_dataset=`` kwarg),
    each target is evaluated *after* the in-distribution run, and the
    per-OOD records are stored under
    ``report.sub_results["ood"][<name>]``. Missing datasets are
    auto-generated when ``auto_generate`` is configured.
    """

    def __init__(
        self,
        config: Optional[EvalConfig] = None,
        *,
        task: str = "eigenvector_feedback",
        model_bundle: Optional[str] = None,
        checkpoint: Optional[str] = None,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        task_layout: str = "paper",
        **overrides: Any,
    ):
        if config is not None:
            self.config = config
        else:
            self._ensure_task_imported(task)
            overrides = dict(overrides)
            data_val = overrides.pop("data", None) or data_path
            output_val = overrides.pop("output_dir", None) or output_dir

            # Build kwargs for EvalConfig
            config_kwargs = {
                "task": task,
                "data": data_val,
                "task_layout": task_layout,
                **overrides,
            }

            # Add model_bundle or checkpoint
            if model_bundle is not None:
                config_kwargs["model_bundle"] = model_bundle
            elif checkpoint is not None:
                config_kwargs["checkpoint"] = checkpoint

            if output_val is not None:
                config_kwargs["output_dir"] = output_val

            self.config = EvalConfig(**config_kwargs)

        self._task: Optional[TaskAdapter] = None
        self._data: Optional[DataAdapter] = None
        self._model: Optional[ModelAdapter] = None
        self._ctx: Optional[EvalContext] = None
        self._runners: List[Any] = []
        self._ensure_all_tasks_imported()

    def _ensure_all_tasks_imported(self) -> None:
        """Best-effort: import every task module shipped with the framework
        so its metrics are registered before the user calls
        ``.metrics()`` or ``.run()``.

        We tolerate failures (e.g. optional tasks not installed).
        """
        import importlib
        try:
            importlib.import_module("csi_eval.feedback_eval.tasks.eigenvector_feedback")
        except Exception:
            pass

    @staticmethod
    def _ensure_task_imported(task_name: str) -> None:
        """Make sure the requested task's module has been imported.

        Tasks live in their own submodules and are not auto-imported.
        Importing the module triggers its @TaskRegistry.register decorator.
        """
        import importlib
        candidates = {
            "eigenvector_feedback": "csi_eval.feedback_eval.tasks.eigenvector_feedback",
        }
        if task_name in candidates:
            try:
                importlib.import_module(candidates[task_name])
            except Exception as e:
                raise ImportError(
                    f"Failed to import task module for {task_name!r}: {e}"
                ) from e

    # ------------------------------------------------------------------
    # Discovery: surface what's available without running
    # ------------------------------------------------------------------
    @staticmethod
    def tasks() -> List[Dict[str, Any]]:
        """List every registered task.

        Returns
        -------
        list of dict
            ``[{"name": "eigenvector_feedback", "class": "EigenvectorFeedbackTask", ...}, ...]``
        """
        out = []
        for name in TaskRegistry.list_tasks():
            cls = TaskRegistry._registry[name]
            t = cls()
            out.append({
                "name": name,
                "class": cls.__name__,
                "input_layout": getattr(t, "input_layout", None),
                "output_layout": getattr(t, "output_layout", None),
                "primary_metric": getattr(t, "primary_metric", None),
            })
        return out

    @staticmethod
    def metrics(category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List every registered metric (optionally filtered by category).

        Returns
        -------
        list of dict
            ``[{"name": "nmse", "category": "task_performance", "unit": "dB",
                "higher_is_better": False, "requires": [...]}, ...]``
        """
        if category is None:
            metrics = MetricRegistry.list_all()
        else:
            metrics = MetricRegistry.list_by_category(category)
        out = []
        for m in metrics:
            out.append({
                "name": m.name,
                "category": m.category,
                "unit": getattr(m, "unit", ""),
                "higher_is_better": m.higher_is_better,
                "requires": sorted(m.requires) if m.requires else [],
            })
        return out

    def info(self) -> Dict[str, Any]:
        """Return a snapshot of this Evaluator's effective configuration.

        Useful for printing or saving alongside the report.
        """
        self._ensure_all_tasks_imported()
        return {
            "config": self.config.to_dict(),
            "available_tasks": [t["name"] for t in Evaluator.tasks()],
            "available_metrics": Evaluator.metrics(),
            "n_ood_targets": len(self.config.ood_targets),
            "ood_targets": [t.get("name", t.get("path", "?")) for t in self.config.ood_targets],
        }

    # ------------------------------------------------------------------
    # Lazy property accessors (build on first access)
    # ------------------------------------------------------------------
    @property
    def task(self) -> TaskAdapter:
        if self._task is None:
            self._task = TaskRegistry.create(self.config.task)
        return self._task

    @property
    def data(self) -> DataAdapter:
        if self._data is None:
            cfg = dict(self.config.dataset)
            cfg["path"] = self.config.data_path
            cfg["cache_dir"] = self.config.eig_cache_dir
            self._data = self.task.build_data(cfg, self.config.splits)
        return self._data

    @property
    def model(self) -> ModelAdapter:
        if self._model is None:
            from ..loaders import load_model_adapter
            self._model = load_model_adapter(self.config, device=self._resolve_device())
        return self._model

    @property
    def ctx(self) -> EvalContext:
        if self._ctx is None:
            device = self._resolve_device()
            # Load model_meta from bundle so metrics can read declarative
            # parameters (nt, compressed_dim, input_shape, etc.) without
            # querying the model.
            model_meta = self._load_bundle_meta()
            # Merge bundle's model_kwargs into config so they appear in
            # config.model_kwargs for any metric that accesses config.
            if model_meta:
                existing = dict(self.config.model_kwargs or {})
                existing.update(model_meta)
                self.config.model_kwargs = existing
            self._ctx = EvalContext(
                model=self.model,
                data=self.data,
                device=device,
                config=self.config,
                splits=self.config.splits,
                task=self.task,
                model_meta=model_meta,
            )
        # NOTE: do NOT pre-populate ``ctx.ood_data`` here.
        # In the previous version the property eagerly built the
        # first OOD adapter, which had a subtle side-effect: by the
        # time the in-distribution pass ran, ``ctx.ood_data`` was
        # already set, so ``RobustnessRunner._has_ood()``
        # returned True and the runner's robustness metrics
        # all reported the OOD (part1_new) numbers instead of the
        # true in-distribution Case 1 baseline.
        #
        # The OOD adapter is built on demand by ``_run_ood_target``
        # right before each OOD pass and explicitly swapped into
        # ``self._ctx.ood_data`` for the duration of that pass;
        # it is reset to ``None`` again afterwards. So the ID
        # pass is guaranteed to see ``ctx.ood_data is None`` and
        # produce genuine in-distribution numbers.
        return self._ctx

    def _load_bundle_meta(self) -> Dict[str, Any]:
        """Load model_meta.json from the bundle directory.

        Flattens ``model_kwargs`` into the top-level dict so that metrics can
        use ``ctx.meta("nt")`` etc. without needing to know the nested structure.
        Also includes ``layout_hint`` for the LayoutAdapter.
        """
        if self.config.model_bundle is None:
            return {}
        try:
            from ..bundle import ModelBundle
            bundle = ModelBundle(self.config.model_bundle)
            meta = dict(bundle.meta or {})
            # Flatten model_kwargs so ctx.meta("nt") works directly
            model_kwargs = meta.pop("model_kwargs", {})
            if model_kwargs:
                for k, v in model_kwargs.items():
                    if k not in meta:
                        meta[k] = v
            # layout_hint used by LayoutAdapter
            layout_hint = meta.pop("layout_hint", {})
            if layout_hint:
                meta["layout_hint"] = layout_hint
            return meta
        except Exception:
            return {}

    def _resolve_device(self) -> torch.device:
        dev = self.config.device
        if dev == "cuda" and not torch.cuda.is_available():
            print("[Evaluator] CUDA not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device(dev)

    # ------------------------------------------------------------------
    # OOD auto-generate (cache-aware)
    # ------------------------------------------------------------------
    def _ensure_ood_dataset(self, target: Dict[str, Any]) -> bool:
        """Make sure the .npy file for an OOD target exists. If a cache
        file is missing and ``target["auto_generate"]`` is configured,
        call the corresponding generation script.

        Returns True when the dataset is ready, False on failure.
        """
        data_path = target.get("data", target.get("path", ""))
        path = Path(data_path)
        scenario = target.get("scenario", "I")
        split = target.get("split", "ood")
        # The file naming follows the same convention as
        # ``WAIREigenDataset._candidate_paths``: split "ood" -> "ood",
        # split "all" -> "all", else the literal split name.
        prefix = {"ood": "ood", "all": "all"}.get(split, split)
        npy = path / f"DATA_H{prefix}{scenario}.npy"
        if npy.exists():
            return True
        ag = target.get("auto_generate")
        if not ag:
            print(f"[Evaluator] OOD target {target.get('name', path)}: "
                  f"{npy.name} not found. "
                  f"To generate it, run: python scripts/generate_part1_subset_2_6GHz.py "
                  f"--output {path}  (or generate_part2_2_6GHz.py for Part 2)")
            return False
        kind = ag.get("kind", "part1")
        script_name = "generate_part1_subset_2_6GHz.py" if kind == "part1" \
            else "generate_part2_2_6GHz.py" if kind == "part2" else None
        if script_name is None:
            print(f"[Evaluator] auto_generate.kind={kind!r} not supported "
                  f"(expected 'part1' or 'part2').")
            return False
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "scripts" / script_name
        if not script.exists():
            print(f"[Evaluator] Auto-generate script not found: {script}")
            return False
        cmd = [os.sys.executable, str(script)]
        for arg in ("scenario_root", "map_start", "map_count",
                    "sample_per_map", "workers", "seed",
                    "bs_list", "ue_list"):
            if arg in ag:
                cmd += [f"--{arg.replace('_', '-')}", str(ag[arg])]
        cmd += ["--output", str(path)]
        if "sample_per_map" in ag and kind == "part2":
            pass  # already added
        if kind == "part1" and "map_start" not in ag:
            print("[Evaluator] auto_generate for part1 requires map_start.")
            return False
        print(f"[Evaluator] Auto-generating OOD target "
              f"{target.get('name', path.name)} via {script_name} ...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Evaluator] Auto-generate failed: {e}")
            return False
        return npy.exists()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        categories: Optional[List[str]] = None,
        metrics: MetricFilter = None,
        id_only: bool = False,
    ) -> EvalReport:
        """Run the full evaluation pipeline.

        Parameters
        ----------
        categories:
            Subset of metric categories to run. ``None`` (default) means
            "all 4 categories". Ignored when ``metrics`` is provided.
        metrics:
            Optional list of metric *names* to run (e.g. ``["nmse",
            "sgcs", "recovery"]``). When supplied, takes priority
            over ``categories``: only the named metrics are computed.
            This is the convenient way to ask for a single number::

                Evaluator(...).run(metrics=["sgcs"])
        id_only:
            When True, skip every OOD target entirely (no cross-scenario pass).
            The Robustness category then contains only the in-distribution
            Case 1 baseline; the zero_shot/fine_tune/gap_nmse metrics are
            recorded as ``skipped (no OOD target)`` so you still get a
            single ``EvalReport`` back. Equivalent to constructing the
            ``EvalConfig`` with ``include_default_ood=False``.

        Returns
        -------
        EvalReport
            Contains all in-distribution records plus, for every entry
            in ``config.ood_targets``, an extra sub-result under
            ``report.sub_results["ood"][<target_name>]``.
        """
        from ..runners import build_default_runners

        # 0. If id_only was requested, drop the configured OOD targets so
        # the rest of the run() flow naturally skips them.
        if id_only:
            self.config.ood_targets = []
            self.config.include_default_ood = False

        _ = self.ctx
        meta: Dict[str, Any] = {
            "task": self.config.task,
            "device": str(self.ctx.device),
            "model_bundle": self.config.model_bundle,
            "checkpoint": self.config.checkpoint,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.config.seed,
            "ood_targets": [t.get("name", t.get("data", "?"))
                            for t in self.config.ood_targets],
        }
        try:
            meta["model_name"] = self.model.task_name()
        except Exception:
            pass
        try:
            meta["model_size_mb"] = self.model.get_state_dict_mb()
        except Exception:
            pass

        report = EvalReport(config=self.config, meta=meta)

        # ---- In-distribution pass ----
        requested_names = self._normalise_metric_filter(metrics)
        runners = build_default_runners(self.config, self.ctx)
        for r in progress(runners, total=len(runners),
                          desc="feedback: evaluation sections", unit="section", leave=True):
            cat = getattr(r, "category", "unknown")
            if not self._category_allowed(cat, categories):
                continue
            print(f"[Evaluator] Running {cat} ...")
            try:
                r.run(report, only=requested_names)
            except TypeError:
                # Backwards compat: runner.run(report) without filter
                r.run(report)
            except Exception as e:
                import traceback
                traceback.print_exc()
                report.add_skipped(f"{cat}_runner", f"runner failed: {e}")

        # ---- OOD targets ----
        # OOD cross-scenario is only meaningful when the user actually
        # wants the robustness category (which now includes generalization metrics).
        # When ``metrics`` is given and excludes every robustness metric, skip OOD
        # entirely (we cannot synthesise a cross-scenario view).
        robustness_metric_names = {
            m.name for m in MetricRegistry.list_by_category("robustness")
        }
        ood_requested = (
            self.config.ood_targets
            and self._category_allowed("robustness", categories)
            and (
                requested_names is None
                or any(n in robustness_metric_names for n in requested_names)
            )
        )
        if ood_requested:
            ood_sub = report.add_sub_dict("ood")
            for tgt in progress(
                self.config.ood_targets, total=len(self.config.ood_targets),
                desc="feedback: OOD targets", unit="target", leave=True
            ):
                name = tgt.get("name") or Path(tgt.get("data", "?")).name
                if not self._ensure_ood_dataset(tgt):
                    report.add_skipped(
                        f"ood::{name}",
                        "dataset not found",
                    )
                    continue
                print(f"[Evaluator] Cross-scenario: {name} ({tgt.get('data')}) ...")
                sub = self._run_ood_target(tgt, requested_names)
                ood_sub[name] = sub
                # Reset the OOD adapter so the next OOD target (or
                # the cleanup path) doesn't see stale data.
                self._ctx.ood_data = None
        elif self.config.ood_targets:
            print("[Evaluator] OOD targets configured but skipped: "
                  "no robustness/generalization metric requested.")

        # ---- Save reports ----
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in self.config.report_formats:
            try:
                report.save(fmt, output_dir=str(out_dir))
            except Exception as e:
                print(f"[Evaluator] report save({fmt}) failed: {e}")

        return report

    def _normalise_metric_filter(self, metrics: MetricFilter) -> Optional[set]:
        if metrics is None:
            return None
        if isinstance(metrics, str):
            metrics = [m]
        return {m.strip() for m in metrics if m and m.strip()}

    def _category_allowed(self, cat: str, categories: Optional[List[str]]) -> bool:
        if not categories:
            return True
        return cat in categories

    def _run_ood_target(
        self,
        target: Dict[str, Any],
        only: Optional[set],
    ) -> Dict[str, Any]:
        """Run the cross-scenario / generalization metrics on a single
        OOD target. The result is a dict of metric-name -> value (or
        a more detailed sub-dict for the ones that produce per-map /
        per-fine-tune breakdowns).
        """
        from ..runners import build_default_runners

        # Make sure the .npy exists (auto-generate if needed) before
        # we ask the task to build a data adapter, otherwise the
        # adapter will report "not found" and we lose the OOD target
        # even when the cache could have been created on the fly.
        if not self._ensure_ood_dataset(target):
            return {"error": "dataset not found and auto_generate failed"}
        # Swap the active OOD adapter in the context.
        target_split = target.get("split", "ood")
        try:
            # Inject eig_cache path into the OOD target config
            ood_data_path = target.get("data", target.get("path", ""))
            ood_cfg = dict(target)
            ood_cfg["path"] = ood_data_path
            ood_cfg["cache_dir"] = str(Path(ood_data_path) / "eig_cache")
            self._ctx.ood_data = self.task.build_ood_adapter(
                ood_cfg, target_split=target_split
            )
        except Exception as e:
            print(f"[Evaluator] OOD adapter build for {target.get('name')}: {e}")
            return {"error": str(e)}
        if self._ctx.ood_data is None:
            return {"error": "ood_data adapter not built"}

        sub: Dict[str, Any] = {
            "target": target,
            "metrics": {},
        }
        try:
            # Apply 2-bit quantization for Cross-Scenario Evaluation to avoid
            # excessive generalization from floating-point precision.
            supports_quant = hasattr(self._ctx.model, "quant_bits")
            original_bits = None
            if supports_quant:
                original_bits = int(getattr(self._ctx.model, "quant_bits", 0))
                setattr(self._ctx.model, "quant_bits", 2)

            for r in build_default_runners(self.config, self._ctx):
                cat = getattr(r, "category", "unknown")
                if cat != "robustness":
                    continue
                try:
                    # When ``into`` is supplied, ``report`` must be None
                    # (otherwise the runner would call ``report.add_skipped``,
                    # which EvalContext does not implement).
                    r.run(None, only=only, into=sub["metrics"])
                except TypeError:
                    # Backwards compat: older runner signature
                    tmp = EvalReport(config=self.config)
                    try:
                        r.run(tmp, only=only)
                    except TypeError:
                        r.run(tmp)
                    for rec in tmp.records:
                        sub["metrics"][rec.name] = rec.value
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    sub["error"] = str(e)
        finally:
            # Restore original quantization bit-width
            if supports_quant and original_bits is not None:
                setattr(self._ctx.model, "quant_bits", original_bits)
            # Always reset the OOD adapter so the next iteration /
            # downstream consumers do not see a stale OOD view
            # masquerading as the in-distribution data.
            self._ctx.ood_data = None
        return sub

    # ------------------------------------------------------------------
    # Convenience: compare several evaluators
    # ------------------------------------------------------------------
    @classmethod
    def compare(
        cls,
        evaluators: List["Evaluator"],
        metric_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> EvalReport:
        reports = [ev.run() for ev in evaluators]
        for i, ev in enumerate(evaluators):
            tag = ev.config.checkpoint or f"model_{i}"
            reports[i].meta["tag"] = tag
        return EvalReport.compare(reports, metric_names=metric_names, output_dir=output_dir)

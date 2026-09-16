"""Command-line interface: ``csifb-eval``.

Usage::

  # Full form
  csifb-eval run --task eigenvector_feedback \\
                   --checkpoint runs/best.pt \\
                   --data-root /path/to/data \\
                   --output results/eval

  # From YAML config
  csifb-eval run --config experiment.yaml

  # Compare
  csifb-eval compare --checkpoints run1.pt run2.pt --names A B \\
                     --output results/compare

  # Discovery
  csifb-eval list --kind all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .version import __version__


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="csifb-eval",
        description=(
            "Self-contained CSI feedback compression model evaluation framework. "
            "Minimum required: --task and (--checkpoint or --model-class or --config)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal
  csifb-eval run --task eigenvector_feedback --checkpoint run.pt --data-root ./data

  # Custom model class
  csifb-eval run --task eigenvector_feedback --checkpoint run.pt \\
                 --model-class models.CsiNet --model-kwargs '{"nt":32,"n_subbands":13}'

  # From YAML config
  csifb-eval run --config eval_config.yaml

  # Compare two models
  csifb-eval compare --checkpoints run_a.pt run_b.pt --names A B --output results/compare
        """,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- run ----
    run_p = sub.add_parser(
        "run",
        help="Evaluate one model and emit reports",
        description="Run the full evaluation pipeline for a single model.",
    )
    run_p.add_argument(
        "--config", "-c",
        metavar="YAML",
        help=(
            "Path to a YAML config file. When provided, all other arguments are ignored "
            "except --output, --device, --categories, and --report (which override the YAML values)."
        ),
    )
    run_p.add_argument(
        "--task", "-t",
        default="eigenvector_feedback",
        help="Task name. Built-in: 'eigenvector_feedback'. Default: eigenvector_feedback",
    )
    run_p.add_argument(
        "--checkpoint",
        help="Path to model checkpoint (.pt). Required unless --model-class is given.",
    )
    run_p.add_argument(
        "--model-class",
        help="Fully qualified class name or importable module.class (e.g. mymodule.MyModel).",
    )
    run_p.add_argument(
        "--model-kwargs",
        help="JSON dict of model constructor kwargs, e.g. '{\"nt\":32,\"compression_dim\":104}'.",
    )
    run_p.add_argument(
        "--data-root", "-d",
        default=None,
        help=(
            "Root directory for data. All relative paths are resolved from here. "
            "Default: '.' (current working directory). "
            "YAML equivalent: data_root."
        ),
    )
    run_p.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory for generated .npy caches. "
            "Default: ~/.cache/csifb-eval/ (Linux/macOS). "
            "YAML equivalent: cache_dir."
        ),
    )
    run_p.add_argument(
        "--dataset",
        default=None,
        help=(
            "Path to the dataset, relative to --data-root. "
            "Default: 'data/wair_d_output/2_6GHz'. "
            "YAML equivalent: dataset.path."
        ),
    )
    run_p.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for reports. Default: results/eval",
    )
    run_p.add_argument(
        "--device",
        default=None,
        help="Compute device (e.g. 'cuda', 'cuda:0', 'cpu'). Default: cuda",
    )
    run_p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Metric categories to run. Default: all five categories.",
    )
    run_p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Specific metric names to run. Overrides --categories.",
    )
    run_p.add_argument(
        "--report",
        nargs="+",
        default=None,
        choices=["json", "html", "markdown"],
        help="Report formats to write. Default: json,html",
    )
    run_p.add_argument(
        "--latency-runs",
        type=int,
        default=None,
        help="Number of inference runs for latency timing. Default: 100",
    )
    run_p.add_argument(
        "--snr",
        nargs="+",
        type=float,
        default=None,
        help="SNR levels (dB) for noise robustness sweep. Default: 5 10 15 20 25 30 40",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Default: 42",
    )
    run_p.add_argument(
        "--history",
        default=None,
        help="Path to train_history.json (enables training_time metric).",
    )
    run_p.add_argument(
        "--preprocessing",
        default=None,
        choices=["eigenvector", "delay_angle"],
        help="CSI preprocessing mode. Default: eigenvector",
    )
    run_p.add_argument(
        "--no-default-ood",
        action="store_true",
        help="Skip auto-registered Part1_NEW and Part2 OOD targets.",
    )
    run_p.add_argument(
        "--validate-only",
        action="store_true",
        help="Load the config and validate paths, then exit without running.",
    )

    # ---- compare ----
    cmp_p = sub.add_parser(
        "compare",
        help="Evaluate several models and emit a leaderboard",
        description="Run the same evaluation for multiple checkpoints and compare.",
    )
    cmp_p.add_argument(
        "--config",
        help="Base YAML config. All models share this config unless overridden.",
    )
    cmp_p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="PT",
        help="Checkpoint paths (one per model).",
    )
    cmp_p.add_argument(
        "--names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Display names for each model (same order as --checkpoints).",
    )
    cmp_p.add_argument(
        "--task",
        default="eigenvector_feedback",
        help="Task name. Default: eigenvector_feedback",
    )
    cmp_p.add_argument(
        "--data-root", "-d",
        default=None,
        help="Root directory for data.",
    )
    cmp_p.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for the leaderboard.",
    )
    cmp_p.add_argument(
        "--device",
        default=None,
        help="Compute device.",
    )
    cmp_p.add_argument(
        "--report",
        nargs="+",
        default=["html", "json"],
        choices=["json", "html", "markdown"],
        help="Report formats. Default: html,json",
    )
    cmp_p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric names to compare. Default: all.",
    )

    # ---- list ----
    list_p = sub.add_parser(
        "list",
        help="List registered tasks and metrics",
        description="Show what tasks and metrics are available without running anything.",
    )
    list_p.add_argument(
        "--kind",
        choices=["tasks", "metrics", "all"],
        default="all",
        help="What to list. Default: all",
    )
    list_p.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (machine-readable).",
    )

    # ---- info ----
    info_p = sub.add_parser(
        "info",
        help="Show effective configuration for a YAML config",
        description="Load a YAML config and print the resolved configuration (paths, defaults, etc.).",
    )
    info_p.add_argument(
        "config",
        help="Path to YAML config file.",
    )
    info_p.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON.",
    )

    return p


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _load_config(args: argparse.Namespace) -> Any:
    """Build an EvalConfig from CLI arguments (with YAML merging support)."""
    from . import EvalConfig

    if args.config:
        # Load from YAML, then override with CLI args
        cfg = EvalConfig.from_yaml(args.config)
        # CLI args override YAML values
        if args.data_root is not None:
            cfg.data_root = args.data_root
        if args.cache_dir is not None:
            cfg.cache_dir = args.cache_dir
        if args.output is not None:
            cfg.output_dir = args.output
        if args.device is not None:
            cfg.device = args.device
        if args.categories is not None:
            cfg.categories = args.categories
        if args.report is not None:
            cfg.report_formats = tuple(args.report)
        if args.no_default_ood:
            cfg.include_default_ood = False
    else:
        # Build from scratch
        model_class = None
        model_kwargs = None

        if args.model_class:
            model_class = _import_class(args.model_class)
            if args.model_kwargs:
                try:
                    model_kwargs = json.loads(args.model_kwargs)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"--model-kwargs must be valid JSON: {e}")

        dataset_cfg: Dict[str, Any] = {}
        if args.cache_dir:
            dataset_cfg["cache_dir"] = args.cache_dir
        if args.dataset:
            dataset_cfg["path"] = args.dataset

        cfg = EvalConfig(
            task=args.task,
            checkpoint=args.checkpoint,
            model_class=model_class,
            model_kwargs=model_kwargs,
            data=args.data_root or ".",
            dataset=dataset_cfg,
            device=args.device or "cuda",
            seed=args.seed or 42,
            output_dir=args.output or "results/eval",
            categories=args.categories,
            latency_runs=args.latency_runs,
            snr_levels_db=tuple(args.snr) if args.snr else None,
            training_history_path=args.history,
            report_formats=tuple(args.report) if args.report else None,
            include_default_ood=not args.no_default_ood,
        )

    return cfg


def _cmd_run(args: argparse.Namespace) -> int:
    from . import Evaluator, EvalConfig

    cfg = _load_config(args)

    if args.validate_only:
        warnings = cfg.validate()
        print(f"[csifb-eval] Config validated.")
        print(f"  data_path          : {cfg.data_path}")
        print(f"  effective_cache_dir: {cfg.eig_cache_dir}")
        print(f"  checkpoint         : {cfg.checkpoint}")
        print(f"  output_dir        : {cfg.output_dir}")
        print(f"  device             : {cfg.device}")
        print(f"  ood_targets        : {len(cfg.ood_targets)}")
        if warnings:
            print(f"\n  Warnings:")
            for w in warnings:
                print(f"    - {w}")
        else:
            print("  No warnings.")
        return 0

    ev = Evaluator(cfg)

    # Show effective config
    info = ev.info()
    print(f"[csifb-eval] task={info['config']['task']}  "
          f"data_path={info['config']['_derived']['data_path']}  "
          f"ood_targets={info['n_ood_targets']}")

    report = ev.run(categories=args.categories, metrics=args.metrics)
    report.print_summary()
    print(f"\nReports written to: {cfg.output_dir}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from . import Evaluator, EvalConfig
    from .core.report import EvalReport

    n = len(args.checkpoints)
    if args.names and len(args.names) != n:
        raise SystemExit("Error: --names and --checkpoints must have the same length.")

    # Base config
    if args.config:
        base_cfg = EvalConfig.from_yaml(args.config)
        if args.data_root:
            base_cfg.data_root = args.data_root
        if args.device:
            base_cfg.device = args.device
    else:
        base_cfg = EvalConfig(
            task=args.task,
            data_root=args.data_root or ".",
            device=args.device or "cuda",
        )

    evaluators: List[Any] = []
    for i, ckpt in enumerate(args.checkpoints):
        cfg = EvalConfig(
            task=base_cfg.task,
            checkpoint=ckpt,
            data_root=base_cfg.data_root,
            device=base_cfg.device,
        )
        tag = args.names[i] if args.names else Path(ckpt).stem
        ev = Evaluator(cfg)
        ev._tag = tag  # type: ignore[attr-defined]
        evaluators.append(ev)

    reports: List[Any] = []
    for ev in evaluators:
        tag = getattr(ev, "_tag", ev.config.checkpoint or "model")
        r = ev.run(metrics=args.metrics)
        r.meta["tag"] = tag
        reports.append(r)

    leaderboard = EvalReport.compare(
        reports,
        metric_names=args.metrics,
        output_dir=args.output,
        title="Model Comparison",
    )

    print("\n" + "=" * 70)
    print("LEADERBOARD")
    print("=" * 70)
    by_model: Dict[str, Dict[str, Any]] = {}
    for rec in leaderboard.records:
        if "::" not in rec.name:
            continue
        model, metric = rec.name.split("::", 1)
        by_model.setdefault(model, {})[metric] = rec.value

    models = list(by_model.keys())
    all_metrics = sorted({m for v in by_model.values() for m in v})
    if not models:
        print("(no metrics recorded)")
    else:
        col_w = max(20, max((len(m) for m in all_metrics), default=20) + 2)
        header = f"{'model':<20} " + " ".join(f"{m:<{col_w}}" for m in all_metrics)
        print(header)
        for model in models:
            row = f"{model:<20} "
            for met in all_metrics:
                v = by_model[model].get(met)
                if v is None:
                    row += f"{'-':<{col_w}} "
                elif isinstance(v, float):
                    row += f"{v:<{col_w}.4f} "
                else:
                    row += f"{str(v):<{col_w}} "
            print(row)

    print(f"\nLeaderboard written to: {args.output}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from .core.registries import MetricRegistry, TaskRegistry

    if args.kind in ("tasks", "all"):
        try:
            from .tasks import eigenvector_feedback  # noqa: F401
        except Exception:
            pass
        tasks = TaskRegistry.list_tasks()
        if args.json:
            print(json.dumps({"tasks": [{"name": t} for t in tasks]}, indent=2))
        else:
            print("Registered tasks:")
            for t in tasks:
                print(f"  - {t}")

    if args.kind in ("metrics", "all"):
        from . import metrics as _m  # noqa: F401
        by_cat: Dict[str, List[Any]] = {}
        for m in MetricRegistry.list_all():
            by_cat.setdefault(m.category, []).append(m)

        if args.json:
            out = {}
            for cat in sorted(by_cat):
                out[cat] = [
                    {
                        "name": m.name,
                        "higher_is_better": m.higher_is_better,
                        "requires": sorted(m.requires) if m.requires else [],
                    }
                    for m in by_cat[cat]
                ]
            print(json.dumps(out, indent=2))
        else:
            print("\nRegistered metrics (by category):")
            for cat in sorted(by_cat):
                print(f"  [{cat}]")
                for m in by_cat[cat]:
                    hi = "↑" if m.higher_is_better else "↓"
                    req = ", ".join(sorted(m.requires)) if m.requires else "none"
                    print(f"    {hi} {m.name:<30}  requires=[{req}]")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from . import EvalConfig

    cfg = EvalConfig.from_yaml(args.config)

    info = {
        "task": cfg.task,
        "data_root": cfg.data_root,
        "data_path": cfg.data_path,
        "cache_dir": cfg.cache_dir,
        "effective_cache_dir": cfg.effective_cache_dir,
        "checkpoint": cfg.checkpoint,
        "device": cfg.device,
        "output_dir": cfg.output_dir,
        "seed": cfg.seed,
        "categories": cfg.categories,
        "fewshot_samples": cfg.fewshot_samples,
        "quant_bits_sweep": cfg.quant_bits_sweep,
        "snr_levels_db": cfg.snr_levels_db,
        "n_ood_targets": len(cfg.ood_targets),
        "ood_targets": [
            {"name": t.get("name"), "path": t.get("path"), "scenario": t.get("scenario")}
            for t in cfg.ood_targets
        ],
        "warnings": cfg.validate(),
    }

    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print("Effective configuration:")
        print(f"  task                  : {info['task']}")
        print(f"  data_root            : {info['data_root']}")
        print(f"  data_path            : {info['data_path']}")
        print(f"  cache_dir            : {info['cache_dir']}")
        print(f"  effective_cache_dir  : {info['effective_cache_dir']}")
        print(f"  checkpoint           : {info['checkpoint']}")
        print(f"  device               : {info['device']}")
        print(f"  output_dir           : {info['output_dir']}")
        print(f"  seed                 : {info['seed']}")
        print(f"  fewshot_samples      : {info['fewshot_samples']}")
        print(f"  quant_bits_sweep     : {info['quant_bits_sweep']}")
        print(f"  snr_levels_db        : {info['snr_levels_db']}")
        print(f"  ood_targets          : {info['n_ood_targets']} target(s)")
        for tgt in info["ood_targets"]:
            print(f"    - {tgt['name']}  scenario={tgt['scenario']}  path={tgt['path']}")
        if info["warnings"]:
            print("\n  Warnings:")
            for w in info["warnings"]:
                print(f"    - {w}")
        else:
            print("\n  No warnings.")
    return 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _import_class(name: str) -> type:
    """Import a class from a fully qualified name like 'mymodule.MyClass'."""
    parts = name.rsplit(".", 1)
    if len(parts) != 2:
        raise SystemExit(
            f"--model-class must be 'module.ClassName', got {name!r}"
        )
    module_name, class_name = parts
    try:
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise AttributeError(class_name)
        return cls
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"Cannot import {name}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "run": _cmd_run,
        "compare": _cmd_compare,
        "list": _cmd_list,
        "info": _cmd_info,
    }

    try:
        return dispatch[args.cmd](args)
    except KeyboardInterrupt:
        print("\n[csifb-eval] Interrupted by user.")
        return 130
    except Exception as e:
        print(f"[csifb-eval] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

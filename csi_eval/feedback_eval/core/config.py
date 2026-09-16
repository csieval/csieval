"""EvalConfig: normalized run configuration for the CSI feedback evaluation framework.

Minimal usage::

    from csi_eval.feedback_eval import Evaluator, EvalConfig

    report = Evaluator(
        task="eigenvector_feedback",
        checkpoint="runs/my_model.pt",
        data="data/Dataset/wair_d_output/2_6GHz",
    ).run()

Field guide
-----------
  REQUIRED:  task,  checkpoint OR model,  data
  RUNTIME:  device, seed, report_formats
  METRICS:  latency_runs, snr_levels_db,
            fewshot_samples, fewshot_epochs, fewshot_lr,
            quant_bits_sweep
  OOD:      ood_targets, include_default_ood
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch.nn as nn
import yaml

from ..bundle import BundleError


# ---------------------------------------------------------------------------
# Default dataset presets
# ---------------------------------------------------------------------------

DEFAULT_DATASET_2_6GHZ: Dict[str, Any] = {
    "config": "2_6GHz",
    "scenario": "I",
    "subband_size": 8,
    "use_cache": True,
    "phase_fix": True,
    "normalize_input_power": False,
    "eig_batch_size": 64,
    "num_workers": 4,
    "n_bs_per_map": 5,
    "n_ue_per_map": 30,
}

DEFAULT_DATASET_7GHZ: Dict[str, Any] = {
    "config": "7GHz",
    "scenario": "I",
    "subband_size": 4,
    "use_cache": True,
    "phase_fix": True,
    "normalize_input_power": False,
    "eig_batch_size": 64,
    "num_workers": 4,
    "n_bs_per_map": 5,
    "n_ue_per_map": 30,
}


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Normalized configuration for one evaluation run.

    Parameters
    ----------
    task : str
        Task name. Must be registered in ``TaskRegistry``.
        Built-in: ``"eigenvector_feedback"``.
    model_bundle : str, optional
        Path to a model bundle directory. A bundle contains:
        - model.py: model class definition
        - *.pt/*.pth: checkpoint file(s)
        - model_meta.json: model metadata (input/output shapes, etc.)
        Either this *or* ``checkpoint`` (with optional ``model``) is required.
    checkpoint : str, optional
        Path to the model checkpoint (.pt file). Used together with ``model``
        (path to model class file) or ``model_class`` + ``model_kwargs``.
        Either this *or* ``model_bundle`` is required.
    data : str
        Path to the preprocessed dataset directory (the directory that contains
        ``DATA_Htest.npy`` etc.). This is the output of the
        ``scripts/generate_part1_subset_2_6GHz.py`` and similar scripts.
        The framework automatically manages the ``eig_cache`` subdirectory
        inside this directory.
    model : Union[nn.Module, str], optional
        An already-instantiated model **or** a path to a model class file
        (e.g. ``"model/ev_csinet.py"``). When a path is given, the framework
        dynamically imports the file, discovers the ``nn.Module`` class inside it,
        infers ``model_kwargs`` from the checkpoint's ``model_info`` (if available),
        instantiates the model, and loads the checkpoint weights into it.
        Mutually exclusive with ``model_bundle``.
    model_class : type, optional
        Explicit ``nn.Module`` subclass to instantiate. Used together with
        ``model_kwargs``. The framework calls ``model_class(**model_kwargs)``,
        then loads ``checkpoint`` into it.
    model_kwargs : dict, optional
        Keyword arguments for ``model_class``.
    model_name : str, optional
        Registry name of a known model (e.g. ``"ev_csinet"``). When set, the
        framework tries to instantiate the real model class from its own
        registry.
    device : str
        Compute device. Default: ``"cuda"`` (auto-downgrades to ``"cpu"`` if
        no GPU is available).
    seed : int
        Random seed for reproducibility. Default: ``42``.
    output_dir : str
        Where to write report files. Default: ``"results/eval"``.
    splits : tuple of str
        Data splits to use for in-distribution evaluation.
        Default: ``("test",)``.
    latency_runs : int
        Number of forward passes for latency timing. Default: ``100``.
    snr_levels_db : tuple of float
        SNR levels (dB) for the noise-robustness sweep. Default:
        ``(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)``.
    fewshot_samples : tuple of int
        Support-set sizes for the few-shot fine-tuning curve.
        Default: ``(0, 5, 10, 20, 50, 100, 300)``.
    fewshot_epochs : int
        Maximum fine-tuning epochs per few-shot point. Default: ``30``.
    fewshot_lr : float
        Fine-tuning learning rate. Default: ``1e-4``.
    n_bs_per_map / n_ue_per_map : int
        Number of BSs / UEs per map for per-map analysis. Defaults: ``5`` / ``30``.
    quant_bits_sweep : tuple of int
        Quantization bit-widths for the robustness sweep.
        Default: ``(0, 2, 4, 8)``.
    report_formats : tuple of str
        Report formats to write. Options: ``"json"``, ``"html"``, ``"markdown"``.
        Default: ``("json", "html")``.
    ood_targets : list of dict
        OOD / cross-scenario evaluation targets. Each dict must contain at
        minimum ``{"name": "...", "data": "..."}``.
    include_default_ood : bool
        Auto-register Part 1 NEW and Part 2 OOD targets for the
        ``eigenvector_feedback`` task when no explicit ``ood_targets`` are given.
        Default: ``True``. Set to ``False`` for a pure in-distribution run.
    """

    # ---- Identity ----
    task: str = "eigenvector_feedback"

    # ---- Model Bundle (preferred way) ----
    model_bundle: Optional[str] = None
    # model_bundle_meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- Tensor Layout ----
    task_layout: str = "paper"  # layout the task expects: "paper", "image", or "flat"

    # ---- Model (mutually exclusive: provide one) ----
    checkpoint: Optional[str] = None
    model: Optional[Union[Any, str]] = field(default=None, repr=False)
    model_class: Optional[Type] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    try_real_model_registry: bool = True

    # ---- Data paths ----
    data: str = "data/Dataset/wair_d_output/2_6GHz"
    _resolved_data: Optional[str] = field(default=None, repr=False)
    dataset: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_DATASET_2_6GHZ))

    # ---- Runtime ----
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "results/eval"
    splits: Tuple[str, ...] = ("test",)

    # ---- Metrics ----
    categories: Optional[List[str]] = None
    latency_runs: int = 100
    snr_levels_db: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)
    fewshot_samples: Tuple[int, ...] = (0, 5, 10, 20, 50, 100, 300)
    fewshot_epochs: int = 30
    fewshot_lr: float = 1e-4
    n_bs_per_map: int = 5
    n_ue_per_map: int = 30
    quant_bits_sweep: Tuple[int, ...] = (0, 2, 4, 8)
    training_history_path: Optional[str] = None
    report_formats: Tuple[str, ...] = ("json", "html")

    # ---- OOD / cross-scenario ----
    ood_targets: List[Dict[str, Any]] = field(default_factory=list)
    include_default_ood: bool = True
    ood_dataset: Optional[Dict[str, Any]] = None  # legacy compat

    _skip_model_validation: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Derived / read-only properties
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> str:
        """Absolute path to the dataset directory."""
        return self._resolved_data or self.data

    @property
    def eig_cache_dir(self) -> str:
        """Absolute path to the eig_cache subdirectory inside data_path."""
        return str(Path(self.data_path) / "eig_cache")

    @property
    def primary_metric(self) -> str:
        """Primary metric name for this task (for quick access)."""
        return {
            "eigenvector_feedback": "sgcs",
            "csi_feedback": "nmse",
            "delay_angle_csi": "nmse",
        }.get(self.task, "sgcs")

    # ------------------------------------------------------------------
    # Preset helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default_dataset(preset: str = "2_6GHz") -> Dict[str, Any]:
        """Return the built-in dataset defaults for a known preset."""
        if preset == "2_6GHz":
            return dict(DEFAULT_DATASET_2_6GHZ)
        if preset == "7GHz":
            return dict(DEFAULT_DATASET_7GHZ)
        raise ValueError(
            f"Unknown preset {preset!r}. Available: '2_6GHz', '7GHz'"
        )

    # ------------------------------------------------------------------
    # Construction / post-init
    # ------------------------------------------------------------------

    def __post_init__(self):
        # 0. Validate exclusive model inputs (skip when loading from YAML)
        if not getattr(self, "_skip_model_validation", False):
            has_bundle = bool(self.model_bundle)
            has_checkpoint = bool(self.checkpoint)
            has_model_instance = self.model is not None and isinstance(self.model, nn.Module)
            has_model_path = self.model is not None and isinstance(self.model, str)
            has_model_class = self.model_class is not None

            if not has_bundle and not has_checkpoint and not has_model_instance and not has_model_class:
                raise ValueError(
                    "EvalConfig requires one of:\n"
                    "  model_bundle='path/to/bundle/'     — bundle directory (recommended)\n"
                    "  checkpoint='path/to/model.pt'      — checkpoint file + model.py\n"
                    "  model=my_nn_module_instance         — pass a pre-loaded nn.Module\n"
                    "  model_class=MyClass, model_kwargs={...}  — instantiate + load"
                )

            # Validate mutual exclusivity
            if has_bundle and (has_checkpoint or has_model_instance or has_model_path or has_model_class):
                raise ValueError(
                    "model_bundle is mutually exclusive with checkpoint/model/model_class.\n"
                    "Provide either model_bundle OR the legacy model loading options."
                )

        # 1. Validate bundle if provided
        if self.model_bundle is not None and not getattr(self, "_skip_model_validation", False):
            from ..bundle import ModelBundle
            try:
                bundle = ModelBundle(self.model_bundle)
                # Use task from bundle if not explicitly specified
                if self.task == "eigenvector_feedback" and bundle.task != "unknown":
                    pass  # Keep user's explicit task choice
            except Exception as e:
                raise BundleError(f"Invalid model_bundle: {e}") from e

        # 2. Resolve data path
        data = Path(self.data).expanduser().resolve()
        self._resolved_data = str(data)
        self.data = str(data)

        # 3. Auto-infer model_name from task
        if self.model_name is None and self.try_real_model_registry and not self.model_bundle:
            self.model_name = self._infer_model_name()

        # 4. Legacy compat: fold ``ood_dataset`` into ``ood_targets``
        if self.ood_dataset is not None and not self.ood_targets:
            self.ood_targets = [dict(self.ood_dataset)]

        # 5. Auto-register default OOD targets
        if self.include_default_ood and not self.ood_targets:
            self.ood_targets = self._default_ood_targets()

    def _infer_model_name(self) -> Optional[str]:
        t = (self.task or "").lower()
        if t in ("eigenvector_feedback", "delay_angle_csi"):
            return "ev_csinet"
        if t in ("csi_feedback",):
            return "csinet"
        return None

    def _default_ood_targets(self) -> List[Dict[str, Any]]:
        """Return default OOD targets for the active task."""
        if self.task != "eigenvector_feedback":
            return []
        parent = str(Path(self.data_path).resolve().parent)
        config_name = self.dataset.get("config", "2_6GHz")
        return [
            {
                "name": "part1_new",
                "data": str(Path(parent) / f"{config_name}_part1_new"),
                "config": config_name,
                "scenario": "I",
                "split": "ood",
                "description": "Part 1 NEW: maps 1000+ (unseen deployment envs)",
            },
            {
                "name": "part2",
                "data": str(Path(parent) / f"{config_name}_part2"),
                "config": config_name,
                "scenario": "II",
                "split": "all",
                "description": "Part 2: 1 BS x 10000 UE (dense deployment)",
            },
        ]

    # ------------------------------------------------------------------
    # OOD target helpers
    # ------------------------------------------------------------------

    def add_ood_target(
        self,
        name: str,
        data: str,
        config: Optional[str] = None,
        scenario: str = "I",
        split: str = "ood",
        description: str = "",
    ) -> "EvalConfig":
        """Append one OOD target and return ``self`` (chainable).

        Example::

            cfg = EvalConfig(task="eigenvector_feedback", checkpoint="run.pt")
            cfg.add_ood_target("part1_new", data="data/part1_new", scenario="I")
            cfg.add_ood_target("part2", data="data/part2", scenario="II", split="all")
        """
        if config is None:
            config = self.dataset.get("config", "2_6GHz")
        resolved = str(Path(data).expanduser().resolve())
        entry: Dict[str, Any] = {
            "name": name,
            "data": resolved,
            "config": config,
            "scenario": scenario,
            "split": split,
        }
        if description:
            entry["description"] = description
        self.ood_targets.append(entry)
        return self

    def remove_ood_target(self, name: str) -> bool:
        for i, tgt in enumerate(self.ood_targets):
            if tgt.get("name") == name:
                self.ood_targets.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Serialization (YAML)
    # ------------------------------------------------------------------

    _PUBLIC_FIELDS: Tuple[str, ...] = (
        "task", "model_bundle", "checkpoint", "data",
        "device", "seed", "output_dir", "splits", "task_layout",
        "latency_runs", "snr_levels_db",
        "fewshot_samples", "fewshot_epochs", "fewshot_lr",
        "n_bs_per_map", "n_ue_per_map",
        "quant_bits_sweep", "training_history_path",
        "report_formats",
        "ood_targets", "include_default_ood",
        "model_name",
    )

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for name in self._PUBLIC_FIELDS:
            v = getattr(self, name, None)
            if v is None:
                continue
            if isinstance(v, tuple):
                v = list(v)
            d[name] = v
        d["_derived"] = {
            "eig_cache_dir": self.eig_cache_dir,
            "primary_metric": self.primary_metric,
        }
        return d

    def to_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
        print(f"[EvalConfig] Saved to {path.resolve()}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EvalConfig":
        path = Path(path).expanduser()
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}

        # Resolve data path relative to the YAML file's parent
        if "data" in d and d["data"]:
            pv = str(d["data"])
            if not os.path.isabs(pv):
                d["data"] = str((path.parent / pv).resolve())

        # Resolve ood_targets paths
        if "ood_targets" in d:
            for tgt in d["ood_targets"]:
                tp = tgt.get("data", "")
                if tp and not os.path.isabs(tp):
                    tgt["data"] = str((path.parent / tp).resolve())

        d.pop("_derived", None)
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        cleaned = {k: v for k, v in d.items() if k in valid}

        # Bypass model validation — YAML files cannot store nn.Module instances.
        # The user must supply the model separately after loading.
        return cls(_skip_model_validation=True, **cleaned)

    def validate(self) -> List[str]:
        warnings: List[str] = []

        if not getattr(self, "_skip_model_validation", False):
            if not self.model_bundle and not self.checkpoint and self.model is None and self.model_class is None:
                warnings.append("Neither model_bundle, checkpoint, model instance, nor model_class provided.")

        if self.model_bundle:
            from ..bundle import ModelBundle
            if not ModelBundle.is_bundle(self.model_bundle):
                warnings.append(f"model_bundle path does not appear to be a valid bundle: {self.model_bundle}")

        dp = Path(self.data_path)
        if not dp.exists():
            warnings.append(f"data path does not exist: {dp}")

        if self.task not in ("eigenvector_feedback", "csi_feedback", "delay_angle_csi"):
            warnings.append(f"Task {self.task!r} is not a known built-in task.")

        return warnings

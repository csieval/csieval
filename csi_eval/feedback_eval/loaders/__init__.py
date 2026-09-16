"""Model loading: four strategies to bring any model into the framework.

The framework is **self-contained**: it does NOT depend on the original project's
``models/`` package. The following strategies are tried in order:

  0. ``model_bundle="path/to/bundle/"`` (Strategy 5, recommended) — A bundle
     directory containing model.py, checkpoint(s), and model_meta.json.
     Best for: sharing models with standardized interface.

  1. ``model=nn.Module`` (Strategy 4) — Pass an already-instantiated model.
     The framework uses it directly. No ``.pt`` required for loading.
     Best for: quick experiments, training pipelines that already have a model.

  2. ``model="path/to/model.py"`` (Strategy 1b) — A path to a model class file.
     The framework dynamically imports the file, discovers the model class inside,
     infers ``model_kwargs`` from the checkpoint's ``model_info`` (if available),
     instantiates the model, and loads the checkpoint weights into it.
     Best for: using a model class file alongside a checkpoint file.

  3. ``model_class=`` + ``model_kwargs=`` — The framework instantiates the
     class and loads the checkpoint. Your class does not need to be in any
     registry; it just needs to accept ``**model_kwargs``.
     Best for: cross-project comparison where the model class is available.

  4. ``checkpoint=`` with full pickled ``nn.Module`` — The framework
     unpickles the module directly. Works when the .pt contains the full
     model object, not just a state_dict.
     Best for: checkpoints saved with ``torch.save(model, path)``.

  5. ``checkpoint=`` with ``state_dict`` — The framework builds a
     ``PlaceholderEVCsiNet`` whose parameter shapes match the standard
     EVCsiNet, then ``load_state_dict(strict=False)``.
     Best for: pure state_dict checkpoints from known architectures.
     Note: meaningful NMSE/SGCS metrics require Strategy 0, 1, 2, or 3.

If none of the above succeeds, a ``RuntimeError`` is raised.

---

Quick reference::

    # Strategy 0: model bundle (recommended)
    from .. import Evaluator
    report = Evaluator(
        task="eigenvector_feedback",
        model_bundle="bundles/my_model/",
        data="data/2_6GHz",
    ).run()

    # Strategy 1b: model class file path + checkpoint
    report = Evaluator(
        checkpoint="runs/best.pt",
        model="model/ev_csinet.py",
        data="data/Dataset/wair_d_output/2_6GHz",
    ).run()

    # Strategy 2: explicit class
    report = Evaluator(
        checkpoint="runs/best.pt",
        model_class=MyCsiNet,
        model_kwargs={"nt": 32, "n_subbands": 13, "compression_dim": 104},
    ).run()

    # Strategy 4: state_dict only (shape-compatible)
    report = Evaluator(checkpoint="runs/state_dict.pt").run()
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

from ..bundle import BundleError, ModelBundle
from ..core.config import EvalConfig
from ..models import PlaceholderEVCsiNet
from .adapter import ModelAdapter, NNModuleAdapter
from .layout import LayoutAdapter


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _try_load_full_module(ckpt_path: str, device: torch.device) -> Optional[nn.Module]:
    """Try to load a full pickled nn.Module from a .pt file."""
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception:
        return None
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict):
        for key in ("model", "module", "net", "model_ema"):
            v = obj.get(key)
            if isinstance(v, nn.Module):
                return v
    return None


def _try_load_state_dict(
    ckpt_path: str,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Try to extract a state_dict from a .pt file."""
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception:
        try:
            obj = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            return None
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        # Raw key→tensor dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model_adapter(
    config: EvalConfig,
    device: Optional[torch.device] = None,
) -> ModelAdapter:
    """Build a ``ModelAdapter`` from an ``EvalConfig``.

    Parameters
    ----------
    config : EvalConfig
        The configuration. Must provide one of:
        - ``model_bundle``: a bundle directory (recommended)
        - ``model``: a pre-loaded ``nn.Module`` instance
        - ``model_class`` + ``model_kwargs``: a class to instantiate
        - ``checkpoint``: a path to a .pt file (Strategies 2 → 3)
    device : torch.device, optional
        Target device. If omitted, resolved from ``config.device``.

    Returns
    -------
    ModelAdapter
        Wraps the underlying ``nn.Module`` and exposes the framework's
        capability interface. Automatically wraps with LayoutAdapter if the
        model's tensor layout differs from the task's expected layout.

    Raises
    ------
    RuntimeError
        When none of the loading strategies succeed.
    """
    if device is None:
        device = torch.device(
            config.device
            if config.device != "cuda" or torch.cuda.is_available()
            else "cpu"
        )

    # Per-call state used by ``_build_and_wrap`` to know which meta to use
    _bundle_meta: Dict[str, Any] = {}

    def _build_and_wrap(net: nn.Module, model_meta: Optional[Dict[str, Any]] = None) -> ModelAdapter:
        """Wrap net with LayoutAdapter (if needed) then ModelAdapter."""
        meta = model_meta or _bundle_meta
        layout_hint = meta.get("layout_hint")
        task_layout = getattr(config, "task_layout", "paper")
        try:
            wrapped = LayoutAdapter(
                net,
                task_layout=task_layout,
                layout_hint=layout_hint,
                model_meta=meta,
            )
            print(f"[loaders] Applied layout adapter: {wrapped}")
        except Exception as e:
            print(f"[loaders] Layout conversion skipped: {e}")
            wrapped = net
        return _wrap(wrapped, device)

    ckpt_path = config.checkpoint

    # ── Strategy 5: Model Bundle (recommended) ──────────────────────────────
    if config.model_bundle is not None:
        try:
            bundle = ModelBundle(config.model_bundle)
            print(f"[loaders] Loading model from bundle: {config.model_bundle}")
            print(f"[loaders] Bundle meta: {bundle.meta}")
            net = bundle.create_model(device=device, validate=True)
            # Flatten model_kwargs so ctx.meta("nt") works at metrics level
            # and LayoutAdapter can read directly.
            _bundle_meta.clear()
            _bundle_meta.update(bundle.meta or {})
            model_kwargs = _bundle_meta.pop("model_kwargs", {})
            for k, v in model_kwargs.items():
                if k not in _bundle_meta:
                    _bundle_meta[k] = v
            layout_hint = _bundle_meta.pop("layout_hint", {})
            if layout_hint:
                _bundle_meta["layout_hint"] = layout_hint
            return _build_and_wrap(net, _bundle_meta)
        except BundleError as e:
            raise RuntimeError(f"Failed to load model_bundle: {e}") from e

    # ── Strategy 1b: model class file path string ─────────────────────────────
    if isinstance(config.model, str):
        net = _try_load_model_file(config.model, config, device)
        if net is not None:
            return _build_and_wrap(net)
        raise RuntimeError(
            f"Failed to load model from file: {config.model!r}\n"
            f"Make sure the file defines an nn.Module subclass."
        )

    # ── Strategy 0b: pre-loaded nn.Module instance ──────────────────────────
    if config.model is not None:
        if isinstance(config.model, nn.Module):
            _warn_if_checkpoint_ignored(
                ckpt_path,
                reason="model instance provided (checkpoint path kept for metadata only)"
            )
            return _build_and_wrap(config.model)
        raise TypeError(
            f"`config.model` must be an nn.Module instance or a .py file path string; "
            f"got {type(config.model).__name__}"
        )
    if config.model_class is not None:
        net = _instantiate_class(config, device)
        return _build_and_wrap(net)

    # ── Strategies 2 & 3: load from checkpoint ─────────────────────────────
    if ckpt_path and os.path.exists(ckpt_path):
        # Strategy 2: full pickled module
        net = _try_load_full_module(ckpt_path, device)
        if net is not None:
            print(f"[loaders] checkpoint contains full nn.Module: {type(net).__name__}")
            return _wrap(net, device)

        # Strategy 3: state_dict → placeholder
        sd = _try_load_state_dict(ckpt_path, device)
        if sd is not None:
            print(
                f"[loaders] checkpoint contains state_dict "
                f"({len(sd)} keys); using placeholder architecture"
            )
            net = _build_placeholder(config, sd)
            try:
                net.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[loaders] state_dict shape mismatch (continuing): {e}")
            return _wrap(net, device)

    raise RuntimeError(
        "load_model_adapter: no model_bundle, model instance, model class, "
        "and checkpoint not found or could not be loaded.\n"
        "Provide one of:\n"
        "  model_bundle='path/to/bundle/'   — bundle directory (recommended)\n"
        "  model=my_nn_module_instance      — pass a pre-loaded nn.Module\n"
        "  model_class=MyClass, model_kwargs={...}  — instantiate + load\n"
        "  checkpoint='path/to/model.pt'"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wrap(net: nn.Module, device: torch.device) -> ModelAdapter:
    """Move net to device and wrap it."""
    try:
        net = net.to(device)
    except Exception as e:
        print(f"[loaders] .to({device}) failed: {e}")
    return NNModuleAdapter(net)


def _instantiate_class(config: EvalConfig, device: torch.device) -> nn.Module:
    """Instantiate ``config.model_class`` and load the checkpoint into it."""
    cls = config.model_class
    kwargs = dict(config.model_kwargs or {})

    # Fill missing dimensions from dataset metadata
    meta = _safe_dataset_metadata(config)
    for key, default in (("nt", 32), ("n_subbands", 13)):
        if key not in kwargs:
            v = meta.get(key) or default
            kwargs.setdefault(key, v)

    net = cls(**kwargs)
    print(f"[loaders] instantiated {cls.__name__} ({_count_params(net):.3f}M params)")

    ckpt_path = config.checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        sd = _try_load_state_dict(ckpt_path, device)
        if sd is not None:
            try:
                net.load_state_dict(sd, strict=True)
                print(f"[loaders] checkpoint loaded (strict=True)")
            except Exception as e:
                print(
                    f"[loaders] strict load failed, falling back to strict=False: {e}"
                )
                try:
                    net.load_state_dict(sd, strict=False)
                except Exception as e2:
                    print(f"[loaders] strict=False load also failed: {e2}")
    return net


def _build_placeholder(
    config: EvalConfig,
    sd: Dict[str, torch.Tensor],
) -> nn.Module:
    """Build a PlaceholderEVCsiNet with shapes inferred from the state_dict or config."""
    # Try to infer Nt / K from encoder key shapes in the state_dict
    nt_hint = config.model_kwargs.get("nt") if config.model_kwargs else None
    k_hint = config.model_kwargs.get("n_subbands") if config.model_kwargs else None
    reduction_hint = config.model_kwargs.get("reduction") if config.model_kwargs else 8

    for key, tensor in sorted(sd.items()):
        if "enc_fc1.weight" in key and tensor.ndim == 2:
            total_dim = tensor.shape[1]
            if nt_hint and k_hint:
                pass  # use explicit values
            else:
                # infer from weight shape: total_dim = 2 * nt * n_subbands
                # Try common (32, 13) first
                for _nt, _k in [(32, 13), (64, 13), (256, 13), (32, 7)]:
                    if 2 * _nt * _k == total_dim:
                        if not nt_hint:
                            nt_hint = _nt
                        if not k_hint:
                            k_hint = _k
                        break

    nt = int(nt_hint or 32)
    n_subbands = int(k_hint or 13)
    reduction = int(reduction_hint or 8)

    return PlaceholderEVCsiNet(
        nt=nt,
        n_subbands=n_subbands,
        reduction=reduction,
    )


def _count_params(net: nn.Module) -> float:
    return sum(p.numel() for p in net.parameters()) / 1e6


def _safe_dataset_metadata(config: EvalConfig) -> Dict[str, Any]:
    """Best-effort dataset metadata from the data adapter."""
    try:
        from ..tasks import get_task
        task = get_task(config.task)
        data = task.build_data(config.dataset, config.splits)
        return data.get_metadata() or {}
    except Exception:
        return {}


def _warn_if_checkpoint_ignored(ckpt_path: Optional[str], reason: str) -> None:
    if ckpt_path:
        print(
            f"[loaders] checkpoint path provided but ignored: {reason}"
        )


def _try_load_model_file(
    model_file: str,
    config: EvalConfig,
    device: torch.device,
) -> Optional[nn.Module]:
    """Dynamically import a .py model file and instantiate the model.

    Discovers the model class by:
      1. Looking for ``EVCsiNet`` (common name in eigenvector feedback tasks).
      2. Looking for any class inheriting from ``nn.Module`` with ``forward``.

    ``model_kwargs`` are inferred from the checkpoint's ``model_info`` if available,
    otherwise from ``config.model_kwargs``.
    """
    path = Path(model_file).expanduser().resolve()
    if not path.exists():
        print(f"[loaders] model file not found: {path}")
        return None

    model_root = path.parent
    model_package = model_root.name  # e.g. "model"

    # Ensure sibling .py files (base.py, registry.py) in the same directory
    # are loaded into sys.modules so that "from .base import ..." inside the
    # model file resolves correctly without requiring a real package install.
    if model_package not in sys.modules:
        parent_mod = type(sys)(model_package, "")
        sys.modules[model_package] = parent_mod

    # Load sibling modules that are not yet populated.
    for sibling in model_root.glob("*.py"):
        sibling_name = sibling.stem  # e.g. "base", "registry", "ev_csinet"
        full_name = f"{model_package}.{sibling_name}"
        if full_name in sys.modules and hasattr(sys.modules[full_name], "__loader__"):
            continue  # already fully loaded
        spec = importlib.util.spec_from_file_location(full_name, sibling)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = model_package
            sys.modules[full_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"[loaders] warning: could not load sibling {full_name}: {e}")

    try:
        target_name = f"{model_package}.{path.stem}"
        spec = importlib.util.spec_from_file_location(target_name, path)
        if spec is None or spec.loader is None:
            print(f"[loaders] could not load spec for: {path}")
            return None

        module = importlib.util.module_from_spec(spec)
        module.__package__ = model_package
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[loaders] error importing {path}: {e}")
        return None

    # --- Discover the model class ---
    model_cls = None
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, nn.Module)
            and attr is not nn.Module
            and attr is not nn.Sequential
            and not getattr(attr, "__abstractmethods__", None)
        ):
            if attr_name in ("EVCsiNet", "CsiNet", "EVCsiNetCNN"):
                model_cls = attr
                break
            if model_cls is None:
                model_cls = attr  # first candidate

    if model_cls is None:
        print(f"[loaders] no nn.Module subclass found in {path}")
        return None

    print(f"[loaders] discovered model class: {model_cls.__name__} from {path.name}")

    # --- Build model_kwargs ---
    kwargs: Dict[str, Any] = dict(config.model_kwargs or {})
    ckpt_path = config.checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            info = ckpt.get("model_info", {})
            if info:
                # Use model_info from checkpoint to auto-fill kwargs
                known_keys = {"nt", "n_subbands", "reduction", "compressed_dim",
                             "embed_dim", "nhead", "num_layers", "inner_ffn",
                             "dropout", "quant_bits", "use_positional_encoding",
                             "learnable_pe", "embedding_scale"}
                for k in known_keys:
                    if k not in kwargs and k in info:
                        kwargs[k] = info[k]
                print(f"[loaders] inferred model_kwargs from checkpoint model_info: {list(kwargs.keys())}")
        except Exception:
            pass

    # Fill dataset-derived defaults if still missing
    meta = _safe_dataset_metadata(config)
    for key, default in (("nt", 32), ("n_subbands", 13)):
        if key not in kwargs:
            kwargs.setdefault(key, meta.get(key) or default)

    # Instantiate
    try:
        net = model_cls(**kwargs)
        print(f"[loaders] instantiated {model_cls.__name__} ({_count_params(net):.3f}M params)")
    except Exception as e:
        print(f"[loaders] failed to instantiate {model_cls.__name__} with kwargs={kwargs}: {e}")
        return None

    # Load checkpoint weights
    if ckpt_path and os.path.exists(ckpt_path):
        sd = _try_load_state_dict(ckpt_path, device)
        if sd is not None:
            try:
                net.load_state_dict(sd, strict=True)
                print(f"[loaders] checkpoint loaded (strict=True)")
            except Exception as e:
                print(f"[loaders] strict load failed, trying strict=False: {e}")
                try:
                    net.load_state_dict(sd, strict=False)
                except Exception as e2:
                    print(f"[loaders] strict=False also failed: {e2}")
    return net

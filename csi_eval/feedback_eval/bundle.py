"""Model bundle: self-contained package for model evaluation.

A bundle directory contains everything needed to evaluate a model:
- model.py: the model class definition (can be any nn.Module subclass)
- *.pt/*.pth: checkpoint/weight files (one is designated as default)
- model_meta.json: metadata about the model's interface (input/output shapes, etc.)

This design allows the evaluator to work with any model without knowing
its internal structure. Only the input/output shapes are constrained.

Directory structure:
    bundle/
    ├── model.py           # Required: model class definition
    ├── best.pt            # Default checkpoint (can be named anything)
    ├── model_meta.json    # Required: model metadata
    └── README.md          # Optional: documentation

model_meta.json schema:
{
    "model_name": "MyModel",
    "task": "eigenvector_feedback",
    "input_shape": [2, 32, 13],
    "output_shape": [2, 32, 13],
    "default_checkpoint": "best.pt",  // optional, defaults to first .pt file
    "description": "Eigenvector-based CSI feedback model",
    "requirements": ["torch", "numpy"],  // optional Python packages
    "model_kwargs": {}  // optional: default kwargs for model instantiation
}
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


class BundleError(Exception):
    """Raised when a bundle is invalid or cannot be loaded."""
    pass


class ModelBundle:
    """A self-contained model package for evaluation.

    Attributes
    ----------
    bundle_path : Path
        Root directory of the bundle.
    meta : Dict[str, Any]
        Parsed contents of model_meta.json.
    model_class : Type[nn.Module]
        The loaded model class.
    checkpoint_path : Path
        Path to the default checkpoint file.
    """

    def __init__(self, bundle_path: str):
        self.bundle_path = Path(bundle_path).expanduser().resolve()
        if not self.bundle_path.is_dir():
            raise BundleError(f"Bundle path is not a directory: {self.bundle_path}")

        self.meta = self._load_meta()
        self.model_class = self._load_model_class()
        self.checkpoint_path = self._find_checkpoint()
        # Merge checkpoint's model_info into meta so all declarative parameters
        # (compressed_dim, total_dim, reduction, etc.) are available to metrics.
        self._merge_checkpoint_info()

    # Subset of ``model_info`` keys that are real constructor parameters.
    # This is used as a hint; the final filter happens after we have access
    # to the model class (see ``_filter_to_constructor_args``).
    _CONSTRUCTOR_KEYS = frozenset({
        "nt", "n_subbands", "nr",
        "reduction", "compressed_dim", "compression_ratio",
        "embed_dim", "nhead", "num_layers", "inner_ffn",
        "dropout", "quant_bits",
        "use_positional_encoding", "learnable_pe", "embedding_scale",
        "patch_dim", "seq_len",
    })

    def _merge_checkpoint_info(self) -> None:
        """Load model_info from checkpoint and merge into self.meta.

        Only fields that are actual constructor parameters of the loaded
        model class are placed into ``model_kwargs``. The full info dict
        is also copied to top-level ``self.meta`` (without overriding
        ``model_meta.json`` values) so metrics can read any of them via
        ``ctx.meta(...)``.
        """
        try:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            return
        info = {}
        if isinstance(ckpt, dict):
            info = ckpt.get("model_info", {})
        if not isinstance(info, dict):
            return

        # Top-level: keep all info keys, never override model_meta.json values.
        for k, v in info.items():
            if k not in self.meta:
                self.meta[k] = v

        # model_kwargs: filter down to constructor parameters of the loaded
        # class. This handles the common case where checkpoints include
        # extra metadata (name, params, *_estimate, ...) that the model
        # class constructor does not accept.
        mk = self.meta.get("model_kwargs", {})
        if not isinstance(mk, dict):
            mk = {}
        accepted = self._filter_to_constructor_args(info.keys())
        for k in accepted:
            if k not in mk:
                mk[k] = info[k]
        if mk:
            self.meta["model_kwargs"] = mk
        print(
            f"[Bundle] Merged model_info from checkpoint: "
            f"{len(info)} fields, {len(accepted)} used as constructor args"
        )

    def _filter_to_constructor_args(self, keys) -> List[str]:
        """Return the subset of ``keys`` that the model class constructor accepts.

        Uses the ``_CONSTRUCTOR_KEYS`` whitelist as a hint, but ALSO verifies
        against the loaded model class's signature when possible. Any key not
        matching the model's signature is dropped.
        """
        try:
            import inspect
            sig = inspect.signature(self.model_class.__init__)
            params = set(sig.parameters.keys())
            # Always allow **kwargs-style constructors to receive everything.
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
        except Exception:
            params = set()
            accepts_var_keyword = True

        result: List[str] = []
        for k in keys:
            if k in self._CONSTRUCTOR_KEYS:
                if accepts_var_keyword or k in params or not params:
                    result.append(k)
        return result

    def _load_meta(self) -> Dict[str, Any]:
        """Load and validate model_meta.json."""
        meta_path = self.bundle_path / "model_meta.json"
        if not meta_path.exists():
            raise BundleError(
                f"model_meta.json not found in bundle: {self.bundle_path}\n"
                f"Bundle must contain a model_meta.json file."
            )

        import json
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except json.JSONDecodeError as e:
            raise BundleError(f"Invalid JSON in model_meta.json: {e}") from e

        # Validate required fields
        required = ["input_shape", "output_shape"]
        for field in required:
            if field not in meta:
                raise BundleError(f"model_meta.json missing required field: {field}")

        return meta

    def _load_model_class(self) -> Type[nn.Module]:
        """Dynamically import the model class from model.py."""
        model_file = self.bundle_path / "model.py"
        if not model_file.exists():
            raise BundleError(
                f"model.py not found in bundle: {self.bundle_path}\n"
                f"Bundle must contain a model.py file defining the model class."
            )

        # Set up module loading environment
        bundle_name = self.bundle_path.name
        if bundle_name not in sys.modules:
            parent_mod = type(sys)(bundle_name, "")
            sys.modules[bundle_name] = parent_mod

        # Load sibling modules if they exist
        for sibling in self.bundle_path.glob("*.py"):
            sibling_name = f"{bundle_name}.{sibling.stem}"
            if sibling_name in sys.modules and hasattr(sys.modules[sibling_name], "__loader__"):
                continue
            spec = importlib.util.spec_from_file_location(sibling_name, sibling)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = bundle_name
                sys.modules[sibling_name] = mod
                try:
                    spec.loader.exec_module(mod)
                except Exception as e:
                    print(f"[Bundle] Warning: could not load sibling {sibling_name}: {e}")

        # Load the main model.py
        model_module_name = f"{bundle_name}.model"
        spec = importlib.util.spec_from_file_location(model_module_name, model_file)
        if spec is None or spec.loader is None:
            raise BundleError(f"Could not load model.py: {model_file}")

        model_module = importlib.util.module_from_spec(spec)
        model_module.__package__ = bundle_name
        try:
            spec.loader.exec_module(model_module)
        except Exception as e:
            raise BundleError(f"Failed to import model.py: {e}") from e
        sys.modules[model_module_name] = model_module

        # Discover the model class
        model_cls = None
        for attr_name in dir(model_module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(model_module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, nn.Module)
                and attr is not nn.Module
                and attr is not nn.Sequential
                and not getattr(attr, "__abstractmethods__", None)
            ):
                model_cls = attr
                break

        if model_cls is None:
            raise BundleError(
                f"No nn.Module subclass found in model.py: {model_file}\n"
                f"The file must define a class that inherits from nn.Module."
            )

        print(f"[Bundle] Loaded model class: {model_cls.__name__}")
        return model_cls

    def _find_checkpoint(self) -> Path:
        """Find the default checkpoint file."""
        # Check if meta specifies a default
        default_name = self.meta.get("default_checkpoint")
        if default_name:
            ckpt_path = self.bundle_path / default_name
            if not ckpt_path.exists():
                raise BundleError(
                    f"Specified checkpoint not found: {ckpt_path}\n"
                    f"Please check 'default_checkpoint' in model_meta.json"
                )
            return ckpt_path

        # Otherwise, find the first .pt or .pth file
        for pattern in ["*.pt", "*.pth"]:
            checkpoints = list(self.bundle_path.glob(pattern))
            if checkpoints:
                return checkpoints[0]

        raise BundleError(
            f"No checkpoint file found in bundle: {self.bundle_path}\n"
            f"Please add a .pt or .pth file, or specify 'default_checkpoint' in model_meta.json"
        )

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Expected input tensor shape (excluding batch dimension)."""
        shape = self.meta["input_shape"]
        return tuple(shape)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Expected output tensor shape (excluding batch dimension)."""
        shape = self.meta["output_shape"]
        return tuple(shape)

    @property
    def layout_hint(self) -> Dict[str, Any]:
        """Return layout configuration from model_meta.json."""
        return self.meta.get("layout_hint", {})

    @property
    def task(self) -> str:
        """The task this model is designed for."""
        return self.meta.get("task", "unknown")

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return self.meta.get("model_name", self.model_class.__name__)

    def instantiate(self, **kwargs) -> nn.Module:
        """Create a model instance with the given kwargs.

        Parameter resolution order (highest priority first):
        1. Explicit kwargs passed to this method
        2. ``model_info`` from the checkpoint file (if available)
        3. ``model_kwargs`` from ``model_meta.json``
        4. Defaults inferred from ``input_shape``

        Parameters
        ----------
        **kwargs : Any
            Override parameters for model instantiation.

        Returns
        -------
        nn.Module
            The instantiated model.
        """
        # Start with defaults from model_meta.json
        default_kwargs = dict(self.meta.get("model_kwargs", {}))

        # Override with kwargs from checkpoint's model_info (if checkpoint exists)
        ckpt_info = self._peek_ckpt_model_info()
        if ckpt_info:
            # Only fill in keys the user has NOT explicitly set via meta or kwargs.
            known_keys = (
                "nt", "n_subbands", "reduction", "compressed_dim",
                "embed_dim", "nhead", "num_layers", "inner_ffn",
                "dropout", "quant_bits", "use_positional_encoding",
                "learnable_pe", "embedding_scale",
            )
            for k in known_keys:
                if k not in default_kwargs and k not in kwargs and k in ckpt_info:
                    default_kwargs[k] = ckpt_info[k]
            print(
                f"[Bundle] Using model_info from checkpoint "
                f"(embed_dim={ckpt_info.get('embed_dim')}, "
                f"num_layers={ckpt_info.get('num_layers')}, "
                f"reduction={ckpt_info.get('reduction')})"
            )

        # Finally, override with explicit kwargs
        default_kwargs.update(kwargs)

        # Use input_shape from meta to infer missing dimensions
        if "nt" not in default_kwargs or "n_subbands" not in default_kwargs:
            input_shape = self.meta.get("input_shape", [])
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3 and input_shape[0] == 2:
                dim1, dim2 = input_shape[1], input_shape[2]
                for nt, n_subbands in [(32, 13), (256, 13), (64, 13), (32, 7)]:
                    if (dim1 == nt and dim2 == n_subbands) or (dim2 == nt and dim1 == n_subbands):
                        if "nt" not in default_kwargs:
                            default_kwargs["nt"] = nt
                        if "n_subbands" not in default_kwargs:
                            default_kwargs["n_subbands"] = n_subbands
                        break

        try:
            model = self.model_class(**default_kwargs)
            print(
                f"[Bundle] Instantiated {self.model_class.__name__} "
                f"with kwargs: {list(default_kwargs.keys())}"
            )
            return model
        except Exception as e:
            raise BundleError(
                f"Failed to instantiate {self.model_class.__name__} "
                f"with kwargs={default_kwargs}: {e}"
            ) from e

    def _peek_ckpt_model_info(self) -> Dict[str, Any]:
        """Safely read ``model_info`` from the default checkpoint.

        Returns an empty dict if the checkpoint cannot be opened or
        contains no model_info. Used to drive model_kwargs auto-fill.
        """
        try:
            ckpt = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception:
            return {}
        if isinstance(ckpt, dict):
            info = ckpt.get("model_info", {})
            return dict(info) if isinstance(info, dict) else {}
        return {}

    def load_weights(self, model: nn.Module, device: torch.device = None) -> None:
        """Load the default checkpoint weights into a model instance.

        Parameters
        ----------
        model : nn.Module
            The model to load weights into.
        device : torch.device, optional
            Target device for loading.
        """
        if device is None:
            device = torch.device("cpu")

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            raise BundleError(
                f"Failed to load checkpoint {self.checkpoint_path}: {e}\n"
                f"The checkpoint may be corrupted or saved in an incompatible format."
            ) from e

        # Extract state_dict if needed
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
        else:
            state_dict = None

        # Try strict load first, then fallback to non-strict
        try:
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=True)
            else:
                # ckpt itself is a state_dict
                model.load_state_dict(ckpt, strict=True)
            print(f"[Bundle] Loaded checkpoint (strict=True): {self.checkpoint_path.name}")
        except Exception as e:
            print(f"[Bundle] Strict load failed, trying strict=False: {e}")
            try:
                if state_dict is not None:
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
                print(f"[Bundle] Loaded checkpoint (strict=False): {self.checkpoint_path.name}")
            except Exception as e2:
                raise BundleError(
                    f"Failed to load weights into {self.model_class.__name__}.\n"
                    f"This usually means the model architecture does not match the checkpoint.\n"
                    f"Check that model_meta.json (or model_kwargs) describe the same architecture "
                    f"as the one used during training.\n"
                    f"Underlying error: {e2}"
                ) from e2

    def validate_io_shapes(self, model: nn.Module) -> bool:
        """Validate that the model produces the expected shapes.

        Parameters
        ----------
        model : nn.Module
            The model to validate.

        Returns
        -------
        bool
            True if shapes match, False otherwise.
        """
        expected_in = self.input_shape
        expected_out = self.output_shape

        if hasattr(model, "get_input_shape"):
            actual_in = tuple(model.get_input_shape())
        elif hasattr(model, "input_shape"):
            actual_in = tuple(model.input_shape)
        else:
            print(f"[Bundle] Warning: model has no get_input_shape() method, skipping shape validation")
            return True  # Cannot validate, assume OK

        if hasattr(model, "get_output_shape"):
            actual_out = tuple(model.get_output_shape())
        elif hasattr(model, "output_shape"):
            actual_out = tuple(model.output_shape)
        else:
            actual_out = actual_in  # Assume same as input for autoencoders

        if expected_in != actual_in:
            print(f"[Bundle] Warning: input shape mismatch")
            print(f"  Expected: {expected_in}")
            print(f"  Actual:   {actual_in}")
            return False

        if expected_out != actual_out:
            print(f"[Bundle] Warning: output shape mismatch")
            print(f"  Expected: {expected_out}")
            print(f"  Actual:   {actual_out}")
            return False

        print(f"[Bundle] Shape validation passed: {expected_in} -> {expected_out}")
        return True

    def create_model(self, device: torch.device = None, validate: bool = True) -> nn.Module:
        """Fully create and load a model from the bundle.

        This is a convenience method that:
        1. Instantiates the model class
        2. Loads the checkpoint weights
        3. Optionally validates input/output shapes

        Parameters
        ----------
        device : torch.device, optional
            Target device. Defaults to CPU.
        validate : bool
            Whether to validate input/output shapes. Default True.

        Returns
        -------
        nn.Module
            The fully loaded model.
        """
        if device is None:
            device = torch.device("cpu")

        model = self.instantiate()
        self.load_weights(model, device)

        if validate:
            self.validate_io_shapes(model)

        model.to(device)
        return model

    @staticmethod
    def is_bundle(path: str) -> bool:
        """Check if a path contains a valid bundle structure.

        Parameters
        ----------
        path : str
            Path to check.

        Returns
        -------
        bool
            True if the path appears to be a valid bundle.
        """
        p = Path(path)
        if not p.is_dir():
            return False
        # A bundle must have model.py and model_meta.json
        has_model_py = (p / "model.py").exists()
        has_meta = (p / "model_meta.json").exists()
        has_checkpoint = any(p.glob("*.pt")) or any(p.glob("*.pth"))
        return has_model_py and has_meta

    def __repr__(self) -> str:
        return (
            f"ModelBundle(\n"
            f"  path={self.bundle_path},\n"
            f"  model={self.model_name},\n"
            f"  task={self.task},\n"
            f"  input_shape={self.input_shape},\n"
            f"  output_shape={self.output_shape},\n"
            f"  checkpoint={self.checkpoint_path.name}\n"
            f")"
        )

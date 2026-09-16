"""Layout adapter: automatic input/output tensor layout conversion.

This module bridges the gap between the dataset layout (fixed per task) and
the model's native layout (fixed by the model definition). When a user's model
uses a different tensor layout than what the evaluation framework expects,
this adapter transparently permutes the data so neither side needs to change.

Supported layouts for eigenvector_feedback task
---------------------------------------------
The task always produces / expects tensors in "paper" layout:

    paper   : [B, 2, Nt, K]   (polarisation, Nt antennas, K subbands)
    image   : [B, 2, K, Nt]   (transpose of paper)
    flat    : [B, 2*Nt*K]     (flattened)

Common mismatches for user's models:
    user_layout="flat"  <-> expected="paper"  : reshape [B,2,Nt,K]<->[B,2*Nt*K]
    user_layout="image" <-> expected="paper"  : transpose [B,2,K,Nt]<->[B,2,Nt,K]

The adapter is transparent: all Protocol methods (forward, get_input_shape,
get_output_shape, encode, ...) are forwarded. Only the tensor layout changes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..core.protocols import DataAdapter, ModelAdapter


class LayoutMismatchError(Exception):
    """Raised when layout conversion is impossible."""
    pass


# ---------------------------------------------------------------------------
# Layout definitions
# ---------------------------------------------------------------------------

class Layout:
    """Describes a tensor layout for eigenvector feedback.

    Attributes
    ----------
    name : str
        Layout identifier: "paper", "image", or "flat".
    ndim : int
        Number of dimensions (4 for paper/image, 1 for flat).
    shape : Tuple[int, ...]
        Shape signature with symbolic names: (B, 2, Nt, K) etc.
    """

    PAPER = "paper"    # [B, 2, Nt, K]
    IMAGE = "image"    # [B, 2, K, Nt]
    FLAT = "flat"     # [B, 2*Nt*K]

    KNOWN = {PAPER, IMAGE, FLAT}

    def __init__(self, name: str, ndim: int, dim_order: Tuple[str, ...]):
        if name not in self.KNOWN:
            raise ValueError(f"Unknown layout {name!r}. Known: {self.KNOWN}")
        if len(dim_order) != ndim:
            raise ValueError(f"dim_order {dim_order} must have {ndim} entries for layout {name}")
        self.name = name
        self.ndim = ndim
        self.dim_order = dim_order  # e.g. ("B", "C", "H", "W")

    def __repr__(self) -> str:
        return f"Layout({self.name}, {'×'.join(self.dim_order)})"

    def dims_of(self, key: str) -> Tuple[int, ...]:
        """Return the axis indices for a named dimension (e.g. "Nt")."""
        return tuple(i for i, d in enumerate(self.dim_order) if d == key)

    @classmethod
    def from_signature(cls, name: str) -> "Layout":
        """Create a layout from its canonical name."""
        if name == cls.PAPER:
            return cls(cls.PAPER, 4, ("B", "C", "H", "W"))
        if name == cls.IMAGE:
            return cls(cls.IMAGE, 4, ("B", "C", "W", "H"))
        if name == cls.FLAT:
            return cls(cls.FLAT, 2, ("B", "F"))
        raise ValueError(f"Unknown layout {name!r}")

    @classmethod
    def infer_from_shape(cls, shape: Tuple[int, ...]) -> "Layout":
        """Infer the most likely layout from a shape tuple.

        Heuristics:
          - 2 dims + prod(dim[1:]) = 2*Nt*K -> flat
          - dim[1]=2, dim[2]*dim[3] = Nt*K -> paper or image
          - If dim[2] <= dim[3] -> image (transpose of paper)
          - Otherwise -> paper
        """
        if len(shape) == 2:
            return cls(cls.FLAT, 2, ("B", "F"))
        # 3D: shape == (C, Nt, K) – heuristics still apply
        if len(shape) == 3 and shape[0] == 2:
            if shape[1] <= shape[2]:
                return cls(cls.IMAGE, 4, ("B", "C", "W", "H"))
            return cls(cls.PAPER, 4, ("B", "C", "H", "W"))
        if len(shape) == 4 and shape[1] == 2:
            if shape[2] <= shape[3]:
                return cls(cls.IMAGE, 4, ("B", "C", "W", "H"))
            return cls(cls.PAPER, 4, ("B", "C", "H", "W"))
        # Fallback: treat as flat with inferred F
        return cls(cls.FLAT, 2, ("B", "F"))


# ---------------------------------------------------------------------------
# Layout conversion helpers
# ---------------------------------------------------------------------------

def _to_paper(x: torch.Tensor, src_layout: Layout) -> torch.Tensor:
    """Convert a tensor from src_layout to paper layout [B, 2, Nt, K]."""
    if src_layout.name == Layout.PAPER:
        return x
    if src_layout.name == Layout.FLAT:
        # x shape: [B, 2*Nt*K] -> need Nt and K from context
        # Caller must have provided them
        raise LayoutMismatchError(
            "Flat layout requires Nt and K to convert. "
            "Set 'layout_hint' in model_meta.json or implement get_input_shape()."
        )
    if src_layout.name == Layout.IMAGE:
        # [B, 2, K, Nt] -> [B, 2, Nt, K]
        return x.transpose(2, 3)
    raise LayoutMismatchError(f"Unknown source layout: {src_layout}")


def _from_paper(x: torch.Tensor, dst_layout: Layout, *, Nt: int, K: int) -> torch.Tensor:
    """Convert a tensor from paper layout [B, 2, Nt, K] to dst_layout."""
    if dst_layout.name == Layout.PAPER:
        return x
    if dst_layout.name == Layout.FLAT:
        # [B, 2, Nt, K] -> [B, 2*Nt*K]
        return x.reshape(x.shape[0], 2 * Nt * K)
    if dst_layout.name == Layout.IMAGE:
        # [B, 2, Nt, K] -> [B, 2, K, Nt]
        return x.transpose(2, 3)
    raise LayoutMismatchError(f"Unknown destination layout: {dst_layout}")


def _make_permute_order(src_layout: Layout, dst_layout: Layout) -> Tuple[int, ...]:
    """Return a tuple of dimension indices to permute src -> dst.

    Only valid when both layouts have the same symbolic dimension names.
    """
    try:
        src_idx = {d: i for i, d in enumerate(src_layout.dim_order)}
        dst_idx = {d: i for i, d in enumerate(dst_layout.dim_order)}
        if set(src_idx.keys()) != set(dst_idx.keys()):
            raise LayoutMismatchError(
                f"Layouts have incompatible dimensions: "
                f"{src_layout.dim_order} vs {dst_layout.dim_order}"
            )
        return tuple(src_idx[d] for d in dst_layout.dim_order)
    except Exception as e:
        raise LayoutMismatchError(
            f"Cannot convert {src_layout.name} -> {dst_layout.name}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# LayoutAdapter
# ---------------------------------------------------------------------------

class LayoutAdapter(nn.Module):
    """Wraps any model and transparently converts input/output layouts.

    This adapter sits between the framework (which always uses the task's
    canonical layout) and the model (which may use a different layout).

    Parameters
    ----------
    model : nn.Module
        The underlying model (already wrapped by ModelAdapter).
    task_layout : str
        The layout expected by the task / dataset. Default: "paper".
    model_layout : str
        The model's native layout. Can be "paper", "image", "flat", or
        "auto" (infer from get_input_shape). Default: "auto".
    layout_hint : dict, optional
        Explicit layout specification from model_meta.json.
        Keys: "input_layout", "output_layout", "Nt", "K".

    Example model_meta.json entry::

        {
            "layout_hint": {
                "input_layout": "flat",
                "output_layout": "flat",
                "Nt": 32,
                "K": 13
            }
        }
    """

    def __init__(
        self,
        model: nn.Module,
        task_layout: str = Layout.PAPER,
        model_layout: str = "auto",
        layout_hint: Optional[Dict[str, Any]] = None,
        model_meta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._inner = model
        self._task_layout = Layout.from_signature(task_layout)
        self._layout_hint = dict(layout_hint) if layout_hint else {}
        # Declarative metadata from model_meta.json (not from model code).
        self._model_meta: Dict[str, Any] = dict(model_meta) if model_meta else {}

        # Resolve model's native layout
        if model_layout != "auto":
            self._model_layout = Layout.from_signature(model_layout)
        elif self._layout_hint.get("input_layout"):
            self._model_layout = Layout.from_signature(
                self._layout_hint["input_layout"]
            )
        else:
            self._model_layout = self._infer_model_layout()

        # Resolve model dimensions: layout_hint > model_meta > inference
        meta_nt = self._model_meta.get("nt") or self._model_meta.get("Nt")
        meta_k = self._model_meta.get("n_subbands") or self._model_meta.get("K")
        hint_nt = self._layout_hint.get("Nt")
        hint_k = self._layout_hint.get("K")
        self._Nt = hint_nt or meta_nt or self._infer_nt()
        self._K = hint_k or meta_k or self._infer_k()
        self._total_dim = 2 * self._Nt * self._K if self._Nt and self._K else None

        # Validate
        if self._model_layout.name == Layout.FLAT and not self._total_dim:
            raise LayoutMismatchError(
                "Flat layout detected but Nt/K are unknown. "
                "Set layout_hint.Nt and layout_hint.K in model_meta.json."
            )

        if self._task_layout.name == Layout.FLAT or self._model_layout.name == Layout.FLAT:
            if not self._total_dim:
                raise LayoutMismatchError(
                    "Flat layout requires Nt and K. "
                    "Set layout_hint.Nt and layout_hint.K in model_meta.json."
                )

        # Build permutation if needed
        if self._task_layout.ndim == self._model_layout.ndim:
            try:
                self._to_model = _make_permute_order(self._task_layout, self._model_layout)
                self._from_model = _make_permute_order(self._model_layout, self._task_layout)
            except LayoutMismatchError:
                self._to_model = None
                self._from_model = None
        else:
            self._to_model = None
            self._from_model = None

        self._needs_reshape = (
            self._task_layout.name == Layout.FLAT
            or self._model_layout.name == Layout.FLAT
        )

    def __setattr__(self, name: str, value: Any) -> None:
        # Always set our own state via direct __setattr__ to bypass nn.Module's
        # sub-module registration machinery for non-inner attributes.
        if name in (
            "_inner", "_task_layout", "_model_layout",
            "_Nt", "_K", "_total_dim", "_to_model", "_from_model",
            "_needs_reshape", "_layout_hint", "_model_meta",
        ):
            object.__setattr__(self, name, value)
            return
        # For inner attributes that nn.Module manages (parameters, buffers,
        # sub-modules), register them via nn.Module.__setattr__ instead of
        # forwarding to the inner model. This preserves the standard PyTorch
        # contract for ``model.quant_bits = 2`` etc.
        if hasattr(self, "_inner") and self._inner is not None:
            # Forward to inner model — this lets ``model.quant_bits = 2``
            # propagate when called via the LayoutAdapter wrapper.
            try:
                setattr(self._inner, name, value)
                return
            except AttributeError:
                pass
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Forward reads from inner model (e.g. quant_bits, parameters(), etc.)
        if name.startswith("_"):
            raise AttributeError(name)
        if hasattr(self, "_inner") and self._inner is not None:
            return getattr(self._inner, name)
        raise AttributeError(name)

    def _infer_model_layout(self) -> Layout:
        """Infer the model's native layout from its shape."""
        try:
            shape = self._inner.get_input_shape()
            layout = Layout.infer_from_shape(tuple(shape))
            print(f"[LayoutAdapter] Inferred model layout: {layout.name} from shape {shape}")
            return layout
        except Exception as e:
            import traceback
            print(f"[LayoutAdapter] Could not infer layout: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"[LayoutAdapter] Defaulting to paper layout")
            return Layout.from_signature(Layout.PAPER)

    def _infer_nt(self) -> Optional[int]:
        """Try to infer Nt from model's get_input_shape."""
        try:
            shape = self._inner.get_input_shape()
            shape = tuple(shape)
            if len(shape) == 3 and shape[0] == 2:
                # [2, Nt, K]
                return max(shape[1], shape[2])
            if len(shape) == 4 and shape[1] == 2:
                # [B, 2, H, W] -> H and W are Nt/K
                return max(shape[2], shape[3])
            if len(shape) == 2:
                total = shape[1]
                for nt, k in [(32, 13), (256, 13), (64, 13), (32, 7)]:
                    if 2 * nt * k == total:
                        return nt
            return None
        except Exception:
            return None

    def _infer_k(self) -> Optional[int]:
        """Try to infer K from model's get_input_shape."""
        try:
            shape = self._inner.get_input_shape()
            shape = tuple(shape)
            if len(shape) == 3 and shape[0] == 2:
                # [2, Nt, K]
                return min(shape[1], shape[2])
            if len(shape) == 4 and shape[1] == 2:
                return min(shape[2], shape[3])
            if len(shape) == 2:
                total = shape[1]
                for nt, k in [(32, 13), (256, 13), (64, 13), (32, 7)]:
                    if 2 * nt * k == total:
                        return k
            return None
        except Exception:
            return None

    def _convert_to_model(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from task layout to model's native layout."""
        # Flat <-> 4D requires reshape
        if self._needs_reshape:
            if self._model_layout.name == Layout.FLAT and self._task_layout.name != Layout.FLAT:
                # [B, 2, Nt, K] -> [B, 2*Nt*K]
                return x.reshape(x.shape[0], self._total_dim)
            if self._task_layout.name == Layout.FLAT and self._model_layout.name != Layout.FLAT:
                # [B, 2*Nt*K] -> [B, 2, Nt, K]
                return x.reshape(x.shape[0], 2, self._Nt, self._K)

        # Permute if needed
        if self._to_model is not None:
            return x.permute(*self._to_model)
        return x

    def _convert_from_model(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from model's native layout back to task layout."""
        # Flat <-> 4D requires reshape
        if self._needs_reshape:
            if self._task_layout.name == Layout.FLAT and self._model_layout.name != Layout.FLAT:
                # [B, 2, Nt, K] -> [B, 2*Nt*K]
                return x.reshape(x.shape[0], self._total_dim)
            if self._model_layout.name == Layout.FLAT and self._task_layout.name != Layout.FLAT:
                # [B, 2*Nt*K] -> [B, 2, Nt, K]
                return x.reshape(x.shape[0], 2, self._Nt, self._K)

        # Permute if needed
        if self._from_model is not None:
            return x.permute(*self._from_model)
        return x

    # ---- nn.Module passthrough ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._convert_to_model(x)
        out = self._inner.forward(x)
        return self._convert_from_model(out)

    # ---- Protocol methods ----
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if not callable(getattr(self._inner, "encode", None)):
            raise NotImplementedError("Inner model does not implement encode()")
        x = self._convert_to_model(x)
        return self._inner.encode(x)

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the shape the task expects (task layout).

        Fully derived from model_meta.json values, NOT from the model.
        Falls back to the inner model's get_input_shape() only when
        model_meta does not provide Nt/K.
        """
        if self._task_layout.name == Layout.PAPER:
            return (2, self._Nt or 0, self._K or 0)
        if self._task_layout.name == Layout.IMAGE:
            return (2, self._K or 0, self._Nt or 0)
        if self._task_layout.name == Layout.FLAT:
            return (self._total_dim or 0,)
        return (0,)

    def get_output_shape(self) -> Tuple[int, ...]:
        return self.get_input_shape()

    def get_compression_ratio(self) -> Optional[float]:
        """Compute compression ratio from model_meta.json if available.

        Does NOT call the model — the ratio is a declarative property.
        """
        total = self._model_meta.get("total_dim", 0) or 0
        comp = self._model_meta.get("compressed_dim", 0) or 0
        if total > 0 and comp > 0:
            return float(comp) / float(total)
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata from model_meta.json.

        Does NOT call the model — all values are declarative.
        """
        return dict(self._model_meta)

    def get_quant_bits(self) -> int:
        """Return quantization bits from model_meta.json (default 0 = float32)."""
        return int(self._model_meta.get("quant_bits", 0))

    def estimate_macs(self, batch_size: int = 1) -> Optional[int]:
        if callable(getattr(self._inner, "estimate_macs", None)):
            return self._inner.estimate_macs(batch_size=batch_size)
        return None

    def get_state_dict_mb(self) -> float:
        if callable(getattr(self._inner, "get_state_dict_mb", None)):
            return self._inner.get_state_dict_mb()
        total = 0.0
        for p in self._inner.parameters():
            total += p.numel() * p.element_size()
        return total / (1024 ** 2)

    def task_name(self) -> str:
        if callable(getattr(self._inner, "task_name", None)):
            return self._inner.task_name()
        return "unknown"

    def parameters(self, recurse: bool = True):
        return self._inner.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self._inner.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._inner.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self._inner.to(*args, **kwargs)
        return self

    def eval(self):
        self._inner.eval()
        return self

    def train(self, mode: bool = True):
        self._inner.train(mode)
        return self

    def __repr__(self) -> str:
        return (
            f"LayoutAdapter(\n"
            f"  task_layout={self._task_layout.name},\n"
            f"  model_layout={self._model_layout.name},\n"
            f"  Nt={self._Nt}, K={self._K},\n"
            f"  inner={self._inner.__class__.__name__}\n"
            f")"
        )

"""Model plugin system: register and instantiate user-supplied CSI compression models.

The framework ships a minimal placeholder (``PlaceholderEVCsiNet``) for shape
inference, but the real model logic lives in user code. This module provides
two ways to bring your model into the framework:

---

## Way 1 — Register a plugin (recommended for published models)

.. code-block:: python

    from ..models import CSICompressionModel, ModelPlugin

    @ModelPlugin.register("my_csinett")
    class MyCsiNet(nn.Module, CSICompressionModel):
        def __init__(self, nt=32, n_subbands=13, compression_dim=104):
            super().__init__()
            self._input_shape = (2, nt, n_subbands)
            self.compression_dim = compression_dim
            self.quant_bits = 0
            # ... your layers ...

        def forward(self, x):
            # x: [B, 2, Nt, K]
            return x  # or your actual model output

        def get_input_shape(self):
            return self._input_shape

    # Now use it:
    cfg = EvalConfig(
        task="eigenvector_feedback",
        checkpoint="runs/my_model.pt",
        model_name="my_csinett",     # matches the registry name
        model_kwargs={"nt": 32, "n_subbands": 13, "compression_dim": 104},
    )

---

## Way 2 — Pass a pre-loaded model instance directly (best for quick experiments)

.. code-block:: python

    from .. import Evaluator, EvalConfig

    my_model = MyCsiNet(nt=32, n_subbands=13)
    my_model.load_state_dict(torch.load("runs/my_model.pt"))

    cfg = EvalConfig(
        task="eigenvector_feedback",
        model=my_model,   # nn.Module instance — no .pt needed here
        checkpoint="runs/my_model.pt",  # for metadata only
    )

---

## Protocol: CSICompressionModel

Your model class does **not** need to inherit from this Protocol. It is
documented here so you know exactly which methods the framework calls and
what they are expected to return.

Required
~~~~~~~~
- ``forward(x: torch.Tensor) -> torch.Tensor``
  Input: ``[B, 2, Nt, K]`` (complex-valued eigenvector, real-imag stacked).
  Output: reconstructed ``[B, 2, Nt, K]`` tensor. The framework handles
  complex-conversion internally.

- ``get_input_shape() -> Tuple[int, int, int]``
  Returns ``(C, H, W)`` = ``(2, Nt, K)``.

Optional
~~~~~~~~
- ``encode(x) -> Tensor`` — returns compressed representation.
  Required for compression_ratio and CSI reduction rate metrics.

- ``get_compression_ratio() -> float`` — return ``compressed_dim / total_dim``.
  If absent, the framework calls ``encode(x)`` once and infers the ratio.

- ``get_quant_bits() -> int`` — effective quantization bit-width (0 = float32).
  Defaults to 0 if absent.

- ``get_model_info() -> Dict[str, Any]`` — metadata dict. Keys used by the
  framework: ``nt``, ``n_subbands``, ``compressed_dim``, ``total_dim``,
  ``compression_ratio``, ``reduction``, ``quant_bits``.

- ``estimate_macs(batch_size=1) -> int`` — MAC count for the computation metrics.

- ``get_state_dict_mb() -> float`` — on-disk checkpoint size in MB.
  If absent, the framework measures ``sum(p.numel() * p.element_size())``.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

__all__ = [
    "CSICompressionModel",
    "ModelPlugin",
    "PlaceholderEVCsiNet",
]


# ---------------------------------------------------------------------------
# Protocol (structural typing — models do NOT need to inherit this)
# ---------------------------------------------------------------------------

class CSICompressionModel:
    """Marker mixin / Protocol for CSI compression models.

    Models do not need to inherit this class. The framework uses duck typing:
    if the model has ``forward`` and ``get_input_shape`` it is usable.
    This class exists solely for documentation and IDE autocompletion.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct CSI feedback from compressed representation."""
        raise NotImplementedError

    def get_input_shape(self) -> Tuple[int, int, int]:
        """Return (C, H, W) = (2, Nt, K)."""
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return compressed code. Return None if not available."""
        return None

    def get_compression_ratio(self) -> Optional[float]:
        """Return compressed_dim / total_dim. Return None if unknown."""
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata dict. Keys used: nt, n_subbands, compressed_dim, total_dim."""
        return {}

    def get_quant_bits(self) -> int:
        """Return effective quantization bit-width. 0 = float32."""
        return 0


# ---------------------------------------------------------------------------
# Model Plugin Registry
# ---------------------------------------------------------------------------

class _ModelRegistry:
    """Simple plugin registry for model classes."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[nn.Module]] = {}

    def register(self, name: str, cls: Optional[Type[nn.Module]] = None):
        """Register a model class.

        Can be used as a decorator::

            @ModelPlugin.register("my_model")
            class MyModel(nn.Module):
                ...
        """
        def _inner(c: Type[nn.Module]) -> Type[nn.Module]:
            self._registry[name] = c
            return c
        if cls is None:
            return _inner
        self._registry[name] = cls
        return cls

    def get(self, name: str) -> Optional[Type[nn.Module]]:
        return self._registry.get(name)

    def list_models(self) -> List[str]:
        return sorted(self._registry.keys())


ModelPlugin: _ModelRegistry = _ModelRegistry()
"""Global model plugin registry. Use ``ModelPlugin.register(name)`` to add models."""


# ---------------------------------------------------------------------------
# Built-in placeholder model (for state_dict-only checkpoints)
# ---------------------------------------------------------------------------

class PlaceholderEVCsiNet(nn.Module):
    """Minimal CNN autoencoder used as a shape-compatible state_dict carrier.

    This model is used **only** when you supply a ``.pt`` checkpoint that
    contains only a ``state_dict`` (no full module), and the framework
    cannot locate the original model class. Its parameter shapes match the
    real ``EVCsiNet`` architecture so ``load_state_dict`` succeeds.

    The placeholder is **not** intended for real metric evaluation — its
    forward pass produces a non-meaningful reconstruction. For real metrics,
    use ``model_class=`` or ``model=`` to supply the actual model.

    Parameter shapes match EVCsiNet for 2.6 GHz (Nt=32, K=13, reduction=8):
      encoder: 2*Nt*K=832 → 4*compressed=416 → compressed=104
      decoder:  compressed=104 → 4*compressed=416 → 2*Nt*K=832
    """

    def __init__(
        self,
        nt: int = 32,
        n_subbands: int = 13,
        embed_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 6,
        reduction: int = 8,
    ):
        super().__init__()
        self.nt = int(nt)
        self.n_subbands = int(n_subbands)
        self.embed_dim = int(embed_dim)
        self.reduction = int(reduction)
        in_dim = 2 * nt * n_subbands
        compressed = max(1, in_dim // reduction)
        self.compressed_dim = compressed
        self.quant_bits = 0
        self._input_shape = (2, nt, n_subbands)

        self.enc_fc1 = nn.Linear(in_dim, 4 * compressed)
        self.enc_fc2 = nn.Linear(4 * compressed, compressed)
        self.dec_fc1 = nn.Linear(compressed, 4 * compressed)
        self.dec_fc2 = nn.Linear(4 * compressed, in_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = torch.relu(self.enc_fc1(x.reshape(b, -1)))
        return torch.relu(self.enc_fc2(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = torch.relu(self.enc_fc1(x.reshape(b, -1)))
        h = torch.relu(self.enc_fc2(h))
        h = torch.relu(self.dec_fc1(h))
        out = torch.tanh(self.dec_fc2(h))
        return out.reshape(*x.shape)

    def get_input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    def get_compression_ratio(self) -> float:
        total = 2 * self.nt * self.n_subbands
        return float(self.compressed_dim) / float(total)

    def get_model_info(self) -> Dict[str, Any]:
        total = 2 * self.nt * self.n_subbands
        return {
            "type": "placeholder",
            "params": int(sum(p.numel() for p in self.parameters())),
            "nt": self.nt,
            "n_subbands": self.n_subbands,
            "total_dim": total,
            "compressed_dim": self.compressed_dim,
            "compression_ratio": self.get_compression_ratio(),
            "reduction": self.reduction,
            "quant_bits": self.quant_bits,
        }

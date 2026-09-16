"""Protocol definitions for the evaluation framework.

These Protocols decouple the framework from concrete implementations:
- TaskAdapter: defines a task (data shape, preprocessing, default metrics)
- DataAdapter: unified interface over different dataset implementations
- ModelAdapter: unified interface over different model implementations
- Metric: a single, runnable metric with declared dependencies

Using typing.Protocol (structural subtyping) lets user-supplied models be
plugged in without inheriting any base class, as long as they quack like
a forward-only model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# DataAdapter
# ---------------------------------------------------------------------------

@runtime_checkable
class DataAdapter(Protocol):
    """Unified dataset interface for any CSI feedback task.

    Implementations must:
    - Provide a torch DataLoader for any split ("train", "val", "test", or
      user-defined names like "ood_eval").
    - Optionally provide the raw complex tensor for noise injection and
      covariance-based analyses.
    - Report per-sample environment index for per-map analysis.
    """

    def get_loader(
        self,
        split: str,
        batch_size: int = 128,
        num_workers: int = 4,
        shuffle: bool = False,
    ) -> DataLoader: ...

    def get_complex_raw(self, split: str) -> Optional[np.ndarray]:
        """Return raw complex CSI [N, Nt, Nr, Nf] for noise injection.
        Return None if not available (e.g. delay-angle task)."""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Return dict of metadata (nt, nr, nf, n_subbands, subband_size, ...)."""
        ...

    def env_index(self, sample_idx: int) -> int:
        """Map a flat sample index to its environment/map index."""
        ...

    @property
    def n_samples(self, split: str = "test") -> int: ...


# ---------------------------------------------------------------------------
# ModelAdapter
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelAdapter(Protocol):
    """Unified model interface.

    Only `forward` and `get_input_shape` are strictly required. All other
    methods are optional; the framework probes and skips metrics that
    require a missing capability.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def get_input_shape(self) -> Tuple[int, ...]: ...

    # ---- Optional capabilities (probed via hasattr / get_capability) ----
    def encode(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return compressed code if encoder is separate from autoencoder."""
        ...

    def get_compression_ratio(self) -> Optional[float]: ...

    def get_model_info(self) -> Dict[str, Any]: ...

    def get_quant_bits(self) -> int:
        """Return effective quantization bit width (0 = no quantization)."""
        ...

    def estimate_macs(self, batch_size: int = 1) -> Optional[int]: ...

    def get_state_dict_mb(self) -> float:
        """Return actual on-disk size of model parameters (in MB)."""
        ...

    def task_name(self) -> str:
        """The CSI task this model targets (e.g. 'eigenvector_feedback')."""
        ...


# ---------------------------------------------------------------------------
# TaskAdapter
# ---------------------------------------------------------------------------

@runtime_checkable
class TaskAdapter(Protocol):
    """A task = data adapter factory + metric defaults + report layout.

    A task knows how to build its DataAdapter from a dataset config dict and
    knows which metrics to compute by default.
    """

    name: str
    input_layout: str           # "paper" | "image" | "complex"
    output_layout: str
    primary_metric: str         # "sgcs" | "nmse" | "evm"

    def build_data(
        self,
        dataset_cfg: Dict[str, Any],
        splits: Tuple[str, ...] = ("train", "val", "test"),
    ) -> DataAdapter: ...

    def default_metrics(self) -> List[str]:
        """Return the list of metric names to run by default."""
        ...


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

@runtime_checkable
class Metric(Protocol):
    """A single, runnable metric.

    Each metric declares:
    - name (used as the dict key in the report)
    - category (one of "task_performance", "storage", "computation",
      "robustness")
    - higher_is_better (for sort/compare)
    - requires: a set of capability strings that must be satisfied
        ("model.encode", "model.compression_ratio", "model.quant_bits",
         "data.complex_raw", "data.metadata.env", ...)
    """

    name: str
    category: str
    higher_is_better: bool
    requires: frozenset

    def compute(self, ctx: "EvalContext") -> Dict[str, Any]:
        """Return a JSON-serializable dict of metric values."""
        ...


# ---------------------------------------------------------------------------
# Capability probing
# ---------------------------------------------------------------------------

# Canonical capability strings used in Metric.requires:
CAP_MODEL_ENCODE = "model.encode"
CAP_MODEL_COMPRESSION_RATIO = "model.compression_ratio"
CAP_MODEL_QUANT_BITS = "model.quant_bits"
CAP_MODEL_MACS = "model.macs"
CAP_MODEL_INFO = "model.info"
CAP_DATA_COMPLEX_RAW = "data.complex_raw"
CAP_DATA_ENV_INDEX = "data.env_index"
CAP_TRAINING_HISTORY = "training.history"
CAP_RAW_CSI = "raw_csi.path"

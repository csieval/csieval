"""Eigenvector feedback task: WAIR-D baseline.

This task is the only task shipped with the framework. It supports two
preprocessing modes:

  * "eigenvector" (default) - Per-subband dominant eigenvector extraction
    (3GPP TypeI codebook convention).  2.6 GHz (Nt=32, Nr=4, Nf=104, K=13)
    and 7 GHz (Nt=256, Nr=8, Nf=52, K=13) are both supported.
  * "delay_angle" - Industry-standard delay-angle domain transform
    (Rx averaging, 2-D DFT, energy-based angle truncation).

All preprocessing is implemented in vendored helpers under ``_internal``,
so this module has **zero runtime dependency** on the rest of the
CSIFeedback-Evaluation-1 project (dataset/, utils/, models/...).

Layouts (model input):
  eigenvector_feedback : [B, 2, Nt, K]   (paper layout)
  delay_angle_csi      : [B, 2, N_angle, N_delay]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...core.registries import TaskRegistry
from ...metrics.robustness import register_eigen_preprocess_fn
from ._internal._wair_d import WAIREigenDataset, WAIR_CONFIGS
from ._internal._delay_angle_csi import WAIRDelayAngleDataset
from ._internal._eigen_preprocess import (
    compute_subband_eigenvectors,
    eigenvectors_to_model_input,
)


# ---------------------------------------------------------------------------
# Defaults: previously lived in configs/ev_csinet_2_6GHz.yaml
# ---------------------------------------------------------------------------

DEFAULT_PRESET_2_6GHZ: Dict[str, Any] = {
    "task": "eigenvector_feedback",
    "data_path": "data/Dataset/wair_d_output/2_6GHz",
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

DEFAULT_PRESET_7GHZ: Dict[str, Any] = {
    "task": "eigenvector_feedback",
    "data_path": "data/Dataset/wair_d_output/7GHz",
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
# DataAdapter
# ---------------------------------------------------------------------------

class EigenvectorDataAdapter:
    """Unified DataAdapter for both eigenvector and delay-angle preprocessing.

    The choice of preprocessing is made by the ``preprocessing`` field of the
    dataset config dict: "eigenvector" (default) or "delay_angle".
    """

    def __init__(self, dataset_cfg: Dict[str, Any], splits: Tuple[str, ...]):
        self._cfg = dict(dataset_cfg)
        self._splits = splits
        self._datasets: Dict[str, Any] = {}
        self._n_samples_cache: Dict[str, int] = {}
        n_bs = int(self._cfg.get("n_bs_per_map", 5))
        n_ue = int(self._cfg.get("n_ue_per_map", 30))
        self._samples_per_env = n_bs * n_ue
        self.preprocessing = self._cfg.get("preprocessing", "eigenvector")

    # ---- public Protocol surface ----

    def get_loader(
        self,
        split: str,
        batch_size: int = 128,
        num_workers: int = 4,
        shuffle: bool = False,
    ) -> DataLoader:
        ds = self._ensure(split)
        return DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )

    def get_complex_raw(self, split: str) -> Optional[np.ndarray]:
        """Return raw complex CSI [N, Nt, Nr, Nf] for noise injection.

        Only available for the eigenvector preprocessing path. For the
        delay-angle path this returns ``None`` and the noise-robustness
        metric is skipped gracefully.
        """
        try:
            ds = self._ensure(split)
            raw = getattr(ds, "_raw", None)
            return raw if raw is not None else None
        except Exception:
            return None

    def get_noisy_eig_cache_path(
        self, split: str, snr_db: float, seed: Optional[int] = None
    ) -> Optional[Path]:
        """Path to a cached eigenvector array for (split, snr_db, seed).

        Returns None if no caching is available for this preprocessing
        (e.g. delay-angle path). Used by the SNR-NMSE metric to avoid
        re-running eigh on noisy CSI across runs.
        """
        try:
            ds = self._ensure(split)
            cache_dir = getattr(ds, "cache_dir", None)
            if cache_dir is None:
                return None
            cfg = self._cfg.get("config", "2_6GHz")
            sc = getattr(ds, "scenario", "I")
            sb = getattr(ds, "subband_size", 8)
            sd = "clean" if (snr_db is None or snr_db >= 9999) else f"snr{int(snr_db)}dB_s{seed}"
            return Path(cache_dir) / f"{cfg}_{split}{sc}_noisy_{sd}_sb{sb}.npy"
        except Exception:
            return None

    def get_metadata(self) -> Dict[str, Any]:
        ds = self._ensure("test")
        cfg_name = self._cfg.get("config", "2_6GHz")
        info = WAIR_CONFIGS.get(cfg_name, WAIR_CONFIGS["2_6GHz"])
        nt, nr, nf = tuple(info["raw_shape"])
        subband_size = int(self._cfg.get("subband_size") or info["subband_size"])
        return {
            "task": "eigenvector_feedback",
            "preprocessing": self.preprocessing,
            "config": cfg_name,
            "raw_shape": list(info["raw_shape"]),
            "nt": nt, "nr": nr, "nf": nf,
            "n_subbands": nf // subband_size,
            "subband_size": subband_size,
            "n_bs_per_map": int(self._cfg.get("n_bs_per_map", 5)),
            "n_ue_per_map": int(self._cfg.get("n_ue_per_map", 30)),
            "input_shape": list(getattr(ds, "shape", (2, nt, nf // subband_size))),
        }

    def env_index(self, sample_idx: int) -> int:
        return sample_idx // self._samples_per_env

    @property
    def env_index_array(self) -> np.ndarray:
        """Full env_index array of shape [n_samples], one environment ID per sample.

        Computed lazily once per adapter.
        """
        if not hasattr(self, "_env_index_array"):
            n = self.n_samples()
            self._env_index_array = (
                np.arange(n, dtype=np.int32) // self._samples_per_env
            )
        return self._env_index_array

    def n_samples(self, split: Optional[str] = None) -> int:
        """Number of samples for a split.

        When called as ``obj.n_samples()`` (no arg), uses ``self._split`` if
        set (for OOD adapters) otherwise falls back to "test".
        When called with an explicit ``split=`` argument, uses that value.
        """
        if split is None:
            split = getattr(self, "_split", "test")
        return self._n_samples_cache.get(split, 0)

    # ---- internals ----

    def _ensure(self, split: str):
        if split not in self._datasets:
            if self.preprocessing == "delay_angle":
                ds = WAIRDelayAngleDataset(
                    data_path=self._cfg["path"],
                    split=split,
                    config=self._cfg.get("config", "7GHz"),
                    scenario=self._cfg.get("scenario", "I"),
                    n_angle_keep=int(self._cfg.get("n_angle_keep", 32)),
                    n_delay_keep=int(self._cfg.get("n_delay_keep", 32)),
                    truncate_angle=bool(self._cfg.get("truncate_angle", True)),
                    normalize=bool(self._cfg.get("normalize", True)),
                    cache_dir=self._cfg.get("cache_dir"),
                    use_cache=bool(self._cfg.get("use_cache", True)),
                    max_samples=self._cfg.get("max_samples"),
                )
            else:
                ds = WAIREigenDataset(
                    data_path=self._cfg["path"],
                    split=split,
                    config=self._cfg.get("config", "2_6GHz"),
                    scenario=self._cfg.get("scenario", "I"),
                    subband_size=self._cfg.get("subband_size"),
                    cache_dir=self._cfg.get("cache_dir"),
                    use_cache=bool(self._cfg.get("use_cache", True)),
                    phase_fix=bool(self._cfg.get("phase_fix", True)),
                    normalize_input_power=bool(self._cfg.get("normalize_input_power", False)),
                    eig_batch_size=int(self._cfg.get("eig_batch_size", 64)),
                    use_gpu=bool(self._cfg.get("use_gpu", True)),
                    subband_aggregation=self._cfg.get("subband_aggregation", "stack"),
                )
            self._datasets[split] = ds
            self._n_samples_cache[split] = len(ds)
        return self._datasets[split]


# ---------------------------------------------------------------------------
# TaskAdapter
# ---------------------------------------------------------------------------

@TaskRegistry.register("eigenvector_feedback")
class EigenvectorFeedbackTask:
    """Eigenvector / delay-angle CSI feedback task."""

    name = "eigenvector_feedback"
    input_layout = "paper"
    output_layout = "paper"
    primary_metric = "sgcs"

    def __init__(self):
        # Self-register the eigen-preprocess callback so that the
        # SNR-NMSE-Slope metric can use it without depending on the
        # original ``dataset/`` package. This is a one-line hook.
        register_eigen_preprocess_fn(_eigen_preprocess_callback)
        pass

    def build_data(
        self,
        dataset_cfg: Dict[str, Any],
        splits: Tuple[str, ...] = ("test",),
    ) -> EigenvectorDataAdapter:
        return EigenvectorDataAdapter(dataset_cfg, splits)

    def build_ood_adapter(
        self,
        dataset_cfg: Dict[str, Any],
        target_split: str = "ood",
    ) -> "EigenvectorDataAdapter":
        """Build a *secondary* DataAdapter for an OOD target dataset.

        Used by the cross-scenario / generalization metrics. The returned
        adapter addresses the same cache directory as the primary one
        (so the eigenvector cache is shared) but loads from a different
        ``data_path`` / ``split``. If the path does not exist or does
        not contain the expected raw file, the call returns ``None``
        and the metric falls back to in-distribution evaluation.
        """
        try:
            data_path = Path(dataset_cfg["path"])
            if not (data_path / f"DATA_H{target_split}{dataset_cfg.get('scenario', 'I')}.npy").exists() \
               and not (data_path / f"DATA_H{'ood' if target_split=='ood' else 'all'}{dataset_cfg.get('scenario', 'I')}.npy").exists():
                print(f"[tasks/eigenvector_feedback] OOD dataset {data_path} not found")
                return None
        except Exception as e:
            print(f"[tasks/eigenvector_feedback] OOD adapter build failed: {e}")
            return None
        cfg = dict(dataset_cfg)
        cfg["split"] = target_split
        adapter = EigenvectorDataAdapter(cfg, splits=(target_split,))
        # Store split and per-map geometry so generalization metrics can
        # use them.  _n_samples_cache stays as the property-defined dict
        # (initialized empty by __init__; filled lazily by _ensure / first
        # .get_loader() call).
        adapter._split = target_split
        adapter._n_bs_per_map = int(dataset_cfg.get("n_bs_per_map", 5))
        adapter._n_ue_per_map = int(dataset_cfg.get("n_ue_per_map", 30))
        return adapter

    def default_metrics(self) -> Optional[list]:
        return None


# ---------------------------------------------------------------------------
# Top-level eigen-preprocess callback used by the SNR-NMSE-Slope metric.
# Signature: (noisy_H_complex[N,Nt,Nr,Nf], subband_size) -> x_numpy[N,2,Nt,K]
# ---------------------------------------------------------------------------

def _eigen_preprocess_callback(noisy_H: np.ndarray, subband_size: int) -> np.ndarray:
    """Compute per-subband dominant eigenvectors and convert to model input.

    Self-contained: uses the vendored ``_internal._eigen_preprocess`` helpers
    so this does not depend on the original project's dataset/ package.
    """
    from ._internal._eigen_preprocess import compute_subband_eigenvectors_auto
    W = compute_subband_eigenvectors_auto(
        noisy_H,
        subband_size=int(subband_size),
        phase_fix=True,
        normalize_input_power=False,
        batch_size=64,
        subband_aggregation="stack",
    )
    return eigenvectors_to_model_input(W, layout="paper")

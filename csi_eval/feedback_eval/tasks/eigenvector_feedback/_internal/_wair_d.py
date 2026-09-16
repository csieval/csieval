"""WAIR-D dataset loader for eigenvector-based CSI feedback.

This file is a copy of ``dataset/wair_d.py`` from the original
CSIFeedback-Evaluation-1 project, vendored into csi_feedback_eval
to make the package self-contained. The only changes are:
  * ``from .eigen_preprocess import ...`` is rewritten to a relative
    import ``from ._eigen_preprocess import ...`` so the helper lives in
    the same ``_internal`` namespace.

Expected generated files keep the raw channel shape:
    DATA_HtrainI.npy, DATA_HvalI.npy, DATA_HtestI.npy
where every array has complex shape [N, Nt, Nr, Nf].

This loader returns eigenvector feedback tensors [2, K, Nt], where K is the
number of feedback subbands.  It can cache preprocessed eigenvectors to avoid
repeating eigendecomposition every epoch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from ._eigen_preprocess import (
    compute_subband_eigenvectors,
    compute_subband_eigenvectors_auto,
    eigenvectors_to_model_input,
)


WAIR_CONFIGS: Dict[str, Dict] = {
    "7GHz": {
        "raw_shape": (256, 8, 52),
        "subband_size": 4,
        "n_subbands": 13,
        "carrier": "6_0",
        "description": "Around 7 GHz TDD; 52 sampled carriers, PRG/subband size 4.",
    },
    "2_6GHz": {
        "raw_shape": (32, 4, 104),
        "subband_size": 8,
        "n_subbands": 13,
        "carrier": "2_6",
        "description": "Around 2.6 GHz FDD; 104 sampled carriers, PRG/subband size 8.",
    },
}


@dataclass
class WAIRSample:
    x: torch.Tensor           # [2, K, Nt]
    raw: Optional[torch.Tensor] = None  # optional complex raw [Nt,Nr,Nf] as torch.complex64


class WAIREigenDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        config: str = "7GHz",
        scenario: str = "I",
        subband_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        phase_fix: bool = True,
        normalize_input_power: bool = False,
        return_raw: bool = False,
        eig_batch_size: int = 64,
        max_samples: Optional[int] = None,
        use_gpu: bool = True,
        subband_aggregation: str = "stack",
    ) -> None:
        if subband_aggregation not in ("stack", "average"):
            raise ValueError(
                f"subband_aggregation must be 'stack' or 'average', got {subband_aggregation!r}"
            )
        if config not in WAIR_CONFIGS:
            raise ValueError(f"Unknown WAIR config {config}. Options: {list(WAIR_CONFIGS)}")
        self.data_path = Path(data_path)
        self.split = split
        self.config = config
        self.scenario = scenario
        self.info = WAIR_CONFIGS[config]
        self.raw_shape = tuple(self.info["raw_shape"])
        self.nt, self.nr, self.nf = self.raw_shape
        self.subband_size = int(subband_size or self.info["subband_size"])
        if self.nf % self.subband_size != 0:
            raise ValueError(f"Nf={self.nf} not divisible by subband_size={self.subband_size}")
        self.n_subbands = self.nf // self.subband_size
        self.phase_fix = phase_fix
        self.normalize_input_power = normalize_input_power
        self.return_raw = return_raw
        self.use_cache = use_cache
        self.eig_batch_size = eig_batch_size
        self.use_gpu = use_gpu
        self.subband_aggregation = subband_aggregation

        self._raw = self._load_raw()
        if max_samples is not None:
            self._raw = self._raw[:max_samples]
        self._validate_raw_shape()

        if cache_dir is None:
            cache_dir = self.data_path / "eig_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._x = self._load_or_create_cache()

    def _candidate_paths(self):
        """Filesystem locations of the raw CSI array.

        Supports two filename conventions:
          1. ``DATA_H{test|train|val}{scenario}.npy`` (train/val/test splits)
          2. ``DATA_Hood{scenario}.npy`` and ``DATA_Hall{scenario}.npy``
             (cross-scenario OOD datasets produced by
             ``scripts/generate_part1_subset.py`` and
             ``scripts/generate_part2.py``). The split name ``ood`` maps
             to ``ood`` and ``all`` maps to ``all`` so that the same
             ``self.split`` value used in cache keys remains consistent.
        """
        split = self.split or "test"
        split_to_prefix = {
            "ood": "ood",
            "all": "all",
        }
        prefix = split_to_prefix.get(split, split)
        base = f"DATA_H{prefix}{self.scenario}"
        return [self.data_path / f"{base}.npy", self.data_path / f"{base}.mat"]

    def _load_raw(self) -> np.ndarray:
        for p in self._candidate_paths():
            if p.exists() and p.suffix == ".npy":
                return np.load(p, mmap_mode=None)
            if p.exists() and p.suffix == ".mat":
                mat = sio.loadmat(p)
                for k, v in mat.items():
                    if not k.startswith("__"):
                        return v
        raise FileNotFoundError(
            f"Cannot find DATA_H{self.split}{self.scenario}.npy/.mat under {self.data_path}"
        )

    def _validate_raw_shape(self) -> None:
        if self._raw.ndim == 3:
            # [N, Nt*Nr, Nf] legacy format
            N = self._raw.shape[0]
            if self._raw.shape[1] != self.nt * self.nr or self._raw.shape[2] != self.nf:
                raise ValueError(f"Cannot reshape legacy raw {self._raw.shape} to {self.raw_shape}")
            self._raw = self._raw.reshape(N, self.nt, self.nr, self.nf)
        if self._raw.ndim != 4:
            raise ValueError(f"Expected raw [N,Nt,Nr,Nf], got {self._raw.shape}")
        if tuple(self._raw.shape[1:]) != self.raw_shape:
            raise ValueError(f"Raw shape mismatch: got {self._raw.shape[1:]}, expected {self.raw_shape}")
        if not np.iscomplexobj(self._raw):
            self._raw = self._raw.astype(np.complex64)

    def _cache_path(self) -> Path:
        flags = []
        flags.append("pf1" if self.phase_fix else "pf0")
        flags.append("norm1" if self.normalize_input_power else "norm0")
        flags.append(f"agg{self.subband_aggregation[:3]}")  # aggsta / aggave
        flags.append("paper")   # layout = paper [2, Nt, K]
        return self.cache_dir / (
            f"{self.config}_{self.split}{self.scenario}_"
            f"Nt{self.nt}_Nr{self.nr}_Nf{self.nf}_sb{self.subband_size}_"
            f"{'_'.join(flags)}.npy"
        )

    def _load_or_create_cache(self) -> np.ndarray:
        p = self._cache_path()
        meta_path = p.with_suffix(".json")
        if self.use_cache and p.exists():
            x = np.load(p, mmap_mode=None)
            return x.astype(np.float32, copy=False)

        eig_fn = compute_subband_eigenvectors_auto if self.use_gpu else compute_subband_eigenvectors
        W = eig_fn(
            self._raw,
            subband_size=self.subband_size,
            phase_fix=self.phase_fix,
            normalize_input_power=self.normalize_input_power,
            batch_size=self.eig_batch_size,
            subband_aggregation=self.subband_aggregation,
        )
        x = eigenvectors_to_model_input(W, layout="paper")
        if self.use_cache:
            np.save(p, x.astype(np.float32))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.config_info, f, indent=2)
        return x.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._x[idx]).float()
        if self.return_raw:
            raw = torch.from_numpy(self._raw[idx].astype(np.complex64))
            return x, raw
        return x

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (2, self.nt, self.n_subbands)   # [2, Nt, K] matching paper Fig 2

    @property
    def config_info(self) -> Dict:
        return {
            "config": self.config,
            "split": self.split,
            "scenario": self.scenario,
            "raw_shape": list(self.raw_shape),
            "input_shape": list(self.shape),
            "subband_size": self.subband_size,
            "n_subbands": self.n_subbands,
            "n_samples": len(self._raw),
            "phase_fix": self.phase_fix,
            "normalize_input_power": self.normalize_input_power,
            "subband_aggregation": self.subband_aggregation,
            "task": "eigenvector_feedback",
        }


class WAIRDataModule:
    def __init__(
        self,
        data_path: str,
        config: str = "7GHz",
        scenario: str = "I",
        subband_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        phase_fix: bool = True,
        normalize_input_power: bool = False,
        eig_batch_size: int = 64,
        max_samples: Optional[int] = None,
        use_gpu: bool = True,
        subband_aggregation: str = "stack",
    ) -> None:
        self.data_path = data_path
        self.config = config
        self.scenario = scenario
        self.subband_size = subband_size
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.phase_fix = phase_fix
        self.normalize_input_power = normalize_input_power
        self.eig_batch_size = eig_batch_size
        self.max_samples = max_samples
        self.use_gpu = use_gpu
        self.subband_aggregation = subband_aggregation

    def dataset(self, split: str, return_raw: bool = False) -> WAIREigenDataset:
        return WAIREigenDataset(
            data_path=self.data_path,
            split=split,
            config=self.config,
            scenario=self.scenario,
            subband_size=self.subband_size,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            phase_fix=self.phase_fix,
            normalize_input_power=self.normalize_input_power,
            return_raw=return_raw,
            eig_batch_size=self.eig_batch_size,
            max_samples=self.max_samples,
            use_gpu=self.use_gpu,
            subband_aggregation=self.subband_aggregation,
        )

    def loader(
        self,
        split: str,
        batch_size: int,
        shuffle: Optional[bool] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        return_raw: bool = False,
    ) -> DataLoader:
        if shuffle is None:
            shuffle = split == "train"
        ds = self.dataset(split, return_raw=return_raw)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=False)

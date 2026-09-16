"""Delay-Angle Domain CSI preprocessing for WAIR-D.

This file is a copy of ``dataset/delay_angle_csi.py`` from the original
CSIFeedback-Evaluation-1 project, vendored into csi_feedback_eval
to make the package self-contained. The only changes are:
  * ``from .wair_d import WAIR_CONFIGS`` is rewritten to a relative import
    ``from ._wair_d import WAIR_CONFIGS`` so the helper lives in the same
    ``_internal`` namespace.

This module implements the industry-standard CSI preprocessing pipeline:
  1. Average Rx antennas: [N, Nt, Nr, Nf] -> [N, Nt, Nf]
  2. 2-D DFT (Delay-Angle domain transform): [N, Nt, Nf] -> [N, Nt, Nf]
  3. Delay domain truncation: Keep all Nf (52 for 7GHz) due to sparse delay domain
  4. Angle domain truncation (energy-based selective sampling): 256 -> 32
  5. Complex to real: [N, Nt, Nf] -> [N, 2, Nt, Nf]

Reference:
  - MATLAB Communication Toolbox: Wiener Filtering for CSI Feedback
  - 3GPP TR 38.901: Channel model for frequency ranges above 6 GHz
  - WAIR-D dataset documentation with standard EVM configuration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from ._wair_d import WAIR_CONFIGS


@dataclass
class WAIRDelayAngleSample:
    x: torch.Tensor  # [2, N_angle, N_delay]
    raw: Optional[torch.Tensor] = None  # optional complex raw [Nt, Nr, Nf]


class WAIRDelayAngleDataset(Dataset):
    """WAIR-D Delay-Angle Domain CSI reconstruction dataset.

    This dataset implements the industry-standard preprocessing pipeline for
    CSI feedback tasks, including:
      - Rx antenna averaging
      - 2-D DFT for delay-angle domain transform
      - Energy-based angle domain truncation
      - Complex to real conversion
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        config: str = "7GHz",
        scenario: str = "I",
        n_angle_keep: int = 32,
        n_delay_keep: int = 32,
        truncate_angle: bool = True,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        if config not in WAIR_CONFIGS:
            raise ValueError(f"Unknown WAIR config {config}. Options: {list(WAIR_CONFIGS)}")

        self.data_path = Path(data_path)
        self.split = split
        self.config = config
        self.scenario = scenario
        self.info = WAIR_CONFIGS[config]
        self.raw_shape = tuple(self.info["raw_shape"])
        self.nt, self.nr, self.nf = self.raw_shape

        # Preprocessing parameters
        self.n_angle_keep = int(n_angle_keep)
        self.n_delay_keep = int(n_delay_keep)
        self.truncate_angle = bool(truncate_angle)
        self.normalize = bool(normalize)
        self.use_cache = bool(use_cache)
        self.max_samples = max_samples

        # Setup cache
        if cache_dir is None:
            cache_dir = self.data_path / "delay_angle_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check cache existence BEFORE loading raw data to skip 96GB load
        cache_path = self._cache_path()
        idx_path = cache_path.parent / f"{cache_path.stem}_angle_idx.npy"
        cache_exists = self.use_cache and cache_path.exists() and idx_path.exists()

        if cache_exists:
            # Cache exists: skip 96GB raw data loading
            self._raw = None
            self._x, self._selected_angle_idx = self._load_cache()
        else:
            # No cache: load raw data and create cache
            self._raw = self._load_raw()
            if max_samples is not None:
                self._raw = self._raw[:max_samples]
            self._validate_raw_shape()
            self._x, self._selected_angle_idx = self._create_cache()

    def _candidate_paths(self):
        base = f"DATA_H{self.split}{self.scenario}"
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
        else:
            self._raw = self._raw.astype(np.complex64, copy=False)

    def _cache_path(self) -> Path:
        trunc = "1" if self.truncate_angle else "0"
        norm = "1" if self.normalize else "0"
        return self.cache_dir / (
            f"{self.config}_{self.split}{self.scenario}_"
            f"Nt{self.nt}_Nr{self.nr}_Nf{self.nf}_"
            f"nangle{self.n_angle_keep}_ndelay{self.n_delay_keep}_trunc{trunc}_norm{norm}.npy"
        )

    def _compute_selected_angle_idx(self, H_delay_angle: np.ndarray) -> np.ndarray:
        """Compute angle indices based on energy distribution."""
        angle_energy = np.mean(np.abs(H_delay_angle) ** 2, axis=(0, 2))  # [Nt]
        topk_idx = np.argsort(angle_energy)[-self.n_angle_keep:]
        return np.sort(topk_idx)

    def _preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the industry-standard preprocessing pipeline."""
        H = np.mean(self._raw, axis=2)            # [N, Nt, Nf]
        N = H.shape[0]

        H_delay = np.fft.ifft(H, axis=2) * np.sqrt(self.nf)
        H_delay_angle = np.fft.fft(H_delay, axis=1) / np.sqrt(self.nt)

        if self.n_delay_keep < self.nf:
            delay_energy = np.mean(np.abs(H_delay_angle) ** 2, axis=(0, 1))  # [Nf]
            topk_delay_idx = np.argsort(delay_energy)[-self.n_delay_keep:]
            H_delay_angle = H_delay_angle[:, :, topk_delay_idx]

        selected_angle_idx = self._compute_selected_angle_idx(H_delay_angle)

        if self.truncate_angle:
            H_truncated = H_delay_angle[:, selected_angle_idx, :]
        else:
            H_truncated = H_delay_angle
            selected_angle_idx = np.arange(self.nt)

        x = np.stack([H_truncated.real, H_truncated.imag], axis=1).astype(np.float32)

        if self.normalize:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            std = x.std(axis=(0, 2, 3), keepdims=True)
            x = (x - mean) / (std + 1e-8)
            x = x * 0.5 + 0.5

        return x.astype(np.float32), selected_angle_idx

    def _load_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self._cache_path()
        idx_path = p.parent / f"{p.stem}_angle_idx.npy"
        x = np.load(p, mmap_mode=None).astype(np.float32)
        selected_angle_idx = np.load(idx_path)
        if self.max_samples is not None:
            x = x[:self.max_samples]
        return x, selected_angle_idx

    def _create_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self._cache_path()
        meta_path = p.with_suffix(".json")
        idx_path = p.parent / f"{p.stem}_angle_idx.npy"

        x, selected_angle_idx = self._preprocess()

        if self.use_cache:
            np.save(p, x.astype(np.float32))
            np.save(idx_path, selected_angle_idx)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.config_info, f, indent=2)

        return x.astype(np.float32, copy=False), selected_angle_idx

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self._x[idx]).float()

    @property
    def shape(self) -> Tuple[int, int, int]:
        n_angle = self.n_angle_keep if self.truncate_angle else self.nt
        n_delay = self.n_delay_keep if self.n_delay_keep < self.nf else self.nf
        return (2, n_angle, n_delay)

    @property
    def config_info(self) -> Dict:
        return {
            "config": self.config,
            "split": self.split,
            "scenario": self.scenario,
            "raw_shape": list(self.raw_shape),
            "input_shape": list(self.shape),
            "n_angle_keep": self.n_angle_keep,
            "n_delay_keep": self.n_delay_keep,
            "truncate_angle": self.truncate_angle,
            "normalize": self.normalize,
            "selected_angle_idx": self.selected_angle_idx.tolist() if hasattr(self, "_selected_angle_idx") else None,
            "n_samples": len(self._x) if hasattr(self, "_x") and self._x is not None else len(self._raw),
            "task": "delay_angle_csi",
            "preprocessing_steps": [
                "Step 1: Average Rx antennas [N, Nt, Nr, Nf] -> [N, Nt, Nf]",
                "Step 2: 2-D DFT (Delay-Angle domain transform)",
                "Step 3: Delay domain truncation (energy-based selective sampling)",
                "Step 4: Angle domain truncation (energy-based selective sampling)",
                "Step 5: Complex to real conversion",
            ],
            "metric_note": "NMSE/SGCS computed on delay-angle domain [2, N_angle, N_delay] tensors.",
        }

    @property
    def selected_angle_idx(self) -> np.ndarray:
        if hasattr(self, "_selected_angle_idx"):
            return self._selected_angle_idx
        return np.arange(self.nt)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_raw", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._raw = None


class WAIRDelayAngleDataModule:
    def __init__(
        self,
        data_path: str,
        config: str = "7GHz",
        scenario: str = "I",
        n_angle_keep: int = 32,
        n_delay_keep: int = 32,
        truncate_angle: bool = True,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.data_path = data_path
        self.config = config
        self.scenario = scenario
        self.n_angle_keep = n_angle_keep
        self.n_delay_keep = n_delay_keep
        self.truncate_angle = truncate_angle
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_samples = max_samples

    def dataset(self, split: str) -> WAIRDelayAngleDataset:
        return WAIRDelayAngleDataset(
            data_path=self.data_path,
            split=split,
            config=self.config,
            scenario=self.scenario,
            n_angle_keep=self.n_angle_keep,
            n_delay_keep=self.n_delay_keep,
            truncate_angle=self.truncate_angle,
            normalize=self.normalize,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            max_samples=self.max_samples,
        )

    def loader(
        self,
        split: str,
        batch_size: int,
        shuffle: Optional[bool] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        if shuffle is None:
            shuffle = split == "train"
        ds = self.dataset(split)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

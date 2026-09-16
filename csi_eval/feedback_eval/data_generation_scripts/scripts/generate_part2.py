#!/usr/bin/env python3
"""Generate Part 2 (Scenario II) WAIR-D CSI feedback arrays.

WAIR-D Part 2 characteristics:
  - 100 maps from global cities
  - 1 BS location per map
  - 10000 UE positions per map
  - Dense deployment: fundamentally different from Part 1 (sparse: 5 BS × 30 UE)
  - This is the KEY dataset for cross-deployment-configuration generalization eval.

Output:
  DATA_HallII.npy  (all 100 maps, 1M samples total = 100 maps × 1 BS × 10000 UE)
  dataset_info.json

No train/val/test split by default — this dataset is purely for OOD evaluation.
If you want a split, use --sample-per-map to take a random subset per map.

Usage:
  python scripts/generate_part2.py --config 7GHz \
      --scenario-root data/Dataset/data/scenario_2 \
      --output data/Dataset/wair_d_output/7GHz_part2

  python scripts/generate_part2.py --config 2_6GHz \
      --scenario-root data/Dataset/data/scenario_2 \
      --output data/Dataset/wair_d_output/2_6GHz_part2

NOTE: This generates 100 maps × 10000 UE = 1,000,000 samples per config.
      With complex64 [N, Nt, Nr, Nf] shape, disk space per config:
        - 7GHz: 1M × 256 × 52 × 8 bytes ≈ 100 GB  (use --sample-per-map)
        - 2.6GHz: 1M × 32 × 104 × 8 bytes ≈ 25 GB
      Use --sample-per-map to reduce memory/disk requirements.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

CONFIGS: Dict[str, Dict] = {
    "7GHz": {
        "carrierFreq": "6_0",
        "BWGHz": 0.01872,
        "sampledCarriers": 52,
        "carrierSampleInterval": 12,
        "Nt": [2, 16, 32],
        "Nr": [2, 4, 1],
        "spacing_t": [0.5, 0.5, 0.8],
        "spacing_r": [0.5, 0.5, 0.5],
        "elements_per_port_z": 4,
        "maxPathNum": 1000,
    },
    "2_6GHz": {
        "carrierFreq": "2_6",
        "BWGHz": 0.01872,
        "sampledCarriers": 104,
        "carrierSampleInterval": 12,
        "Nt": [2, 8, 8],
        "Nr": [2, 2, 1],
        "spacing_t": [0.5, 0.5, 0.5],
        "spacing_r": [0.5, 0.5, 0.5],
        "elements_per_port_z": 4,
        "maxPathNum": 1000,
    },
}


def antenna_position(N, spacing, basis=None):
    if basis is None:
        basis = np.eye(3)
    N0, N1, N2 = N
    d0, d1, d2 = spacing
    p0 = d0 * np.linspace(-(N0 - 1) * 0.5, (N0 - 1) * 0.5, N0)[None, :] * basis[:, 0:1]
    p1 = d1 * np.linspace(-(N1 - 1) * 0.5, (N1 - 1) * 0.5, N1)[None, :] * basis[:, 1:2]
    p2 = d2 * np.linspace(-(N2 - 1) * 0.5, (N2 - 1) * 0.5, N2)[None, :] * basis[:, 2:3]
    p = p0[:, :, None, None] + p1[:, None, :, None] + p2[:, None, None, :]
    return p.reshape((3, np.prod(N)))


def array_response(angle, position, sorted_idx):
    rx = np.sin(angle[sorted_idx, 0]) * np.cos(angle[sorted_idx, 1])
    ry = np.sin(angle[sorted_idx, 0]) * np.sin(angle[sorted_idx, 1])
    rz = np.cos(angle[sorted_idx, 0])
    r = np.stack([rx, ry, rz], axis=1)
    return np.exp(1j * 2 * np.pi * r @ position)


def process_single_channel(H_path_gain: np.ndarray, P: Dict, cfg: Dict) -> np.ndarray:
    Nt = cfg["Nt"]
    Nr = cfg["Nr"]
    sampled_carriers = int(cfg["sampledCarriers"])
    subcarriers = sampled_carriers * int(cfg["carrierSampleInterval"])
    fc_ghz = float(cfg["carrierFreq"].replace("_", "."))
    f_ghz = np.linspace(-0.5 * cfg["BWGHz"], 0.5 * cfg["BWGHz"], sampled_carriers) + fc_ghz

    tau = np.asarray(P["taud"])
    sorted_idx = np.argsort(tau)[: int(cfg.get("maxPathNum", 1000))]
    tau_sorted = tau[sorted_idx]
    doa = np.asarray(P["doa"])
    dod = np.asarray(P["dod"])

    pos_t = antenna_position(Nt, cfg["spacing_t"], np.eye(3))
    pos_r = antenna_position(Nr, cfg["spacing_r"], np.eye(3))
    res_t = array_response(dod, pos_t, sorted_idx)
    res_r = array_response(doa, pos_r, sorted_idx)

    norm_H = H_path_gain[sorted_idx] / np.sqrt(subcarriers)
    ofdm_H = norm_H[:, None] * np.exp(-2j * np.pi * tau_sorted[:, None] * f_ghz[None, :])
    CFR = np.sum(ofdm_H[:, None, None, :] * res_t[:, :, None, None] * res_r[:, None, :, None], axis=0)

    Nt_x, Nt_y, Nt_z = Nt
    Nr_x, Nr_y, Nr_z = Nr
    z_elem = int(cfg.get("elements_per_port_z", 4))
    Nt_z_port = Nt_z // z_elem
    CFR = CFR.reshape(Nt_x, Nt_y, Nt_z, Nr_x, Nr_y, Nr_z, sampled_carriers)
    CFR = CFR.reshape(Nt_x, Nt_y, Nt_z_port, z_elem, Nr_x, Nr_y, Nr_z, sampled_carriers)
    CFR = np.sum(CFR, axis=3) / np.sqrt(z_elem)

    Nt_port = Nt_x * Nt_y * Nt_z_port
    Nr_total = Nr_x * Nr_y * Nr_z
    channel = CFR.reshape(Nt_port, Nr_total, sampled_carriers).astype(np.complex64)
    channel = channel / np.sqrt(np.mean(np.abs(channel) ** 2) + 1e-12)
    return channel.astype(np.complex64)


def process_scenario2_env(args):
    """Process a single Scenario 2 environment.

    Part 2 has only 1 BS (bs0) but 10000 UE per map.
    Returns channels for all UE at bs0, or a sampled subset if sample_ue is set.
    """
    env_dir, cfg, bs_idx, ue_indices, sample_per_map = args
    env_dir = Path(env_dir)
    carrier = cfg["carrierFreq"]

    H_data = np.load(env_dir / f"H_{carrier}_G.npy", allow_pickle=True, encoding="latin1").item()
    P_data = np.load(env_dir / "Path.npy", allow_pickle=True, encoding="latin1").item()

    key = f"bs{bs_idx}_ue"
    channels = []
    for ue_idx in ue_indices:
        full_key = f"{key}{ue_idx:05d}"
        channels.append(process_single_channel(H_data[full_key], P_data[full_key], cfg))

    return channels


def main():
    p = argparse.ArgumentParser(
        description="Generate Part 2 (Scenario II) WAIR-D CSI feedback arrays. "
                    "Part 2 has 1 BS × 10000 UE per map (dense deployment)."
    )
    p.add_argument("--config", choices=list(CONFIGS), required=True,
                   help="Frequency config: 7GHz or 2_6GHz")
    p.add_argument("--scenario-root", required=True,
                   help="Path to WAIR-D scenario_2 data directory")
    p.add_argument("--output", required=True,
                   help="Output directory for generated .npy files")
    p.add_argument("--sample-per-map", type=int, default=None,
                   help="Randomly sample N UE per map to reduce data size. "
                        "Default: all 10000 UE. Recommended: 500-1000 for quick eval.")
    p.add_argument("--workers", type=int, default=min(mp.cpu_count(), 8))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    root = Path(args.scenario_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Find all scenario_2 environment directories
    env_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name)
    )
    if not env_dirs:
        raise FileNotFoundError(f"No numeric env folders in {root}")
    n_maps = len(env_dirs)
    print(f"Found {n_maps} Part 2 environments in {root}")

    bs_idx = 0  # Part 2 has only bs0

    # Determine UE indices per map
    rng = np.random.default_rng(args.seed)
    if args.sample_per_map is not None:
        n_ue_total = 10000
        n_sample = args.sample_per_map
        ue_indices = rng.choice(n_ue_total, size=n_sample, replace=False)
        ue_indices = sorted(ue_indices.tolist())
        total_samples = n_maps * n_sample
        print(f"Sampling {n_sample} UE per map ({total_samples} total samples)")
    else:
        ue_indices = list(range(10000))
        total_samples = n_maps * 10000
        print(f"Using all 10000 UE per map ({total_samples} total samples)")

    # Build jobs — one job per environment
    jobs = [(str(e), cfg, bs_idx, ue_indices, args.sample_per_map) for e in env_dirs]

    channels: List[np.ndarray] = []
    if args.workers <= 1:
        for j in tqdm(jobs, desc="Part2 env"):
            channels.extend(process_scenario2_env(j))
    else:
        with mp.Pool(args.workers) as pool:
            for res in tqdm(pool.imap(process_scenario2_env, jobs),
                            total=len(jobs), desc="Part2 env"):
                channels.extend(res)

    arr = np.asarray(channels, dtype=np.complex64)
    print(f"Generated array shape: {arr.shape}  ({arr.nbytes / 1e9:.1f} GB)")

    # Save all data (no split — this is purely for OOD evaluation)
    np.save(out / "DATA_HallII.npy", arr)

    Nt_x, Nt_y, Nt_z = cfg["Nt"]
    info = {
        "config": args.config,
        "scenario": "II",
        "split": "all",
        "raw_shape": [Nt_x * Nt_y * (Nt_z // cfg["elements_per_port_z"]),
                      cfg["Nr"][0] * cfg["Nr"][1] * cfg["Nr"][2],
                      cfg["sampledCarriers"]],
        "output_convention": "[N, Nt, Nr, Nf] complex64",
        "n_maps": n_maps,
        "n_bs_per_map": 1,
        "n_ue_per_map": len(ue_indices),
        "n_total": len(channels),
        "sampled_ue_per_map": args.sample_per_map,
        "config_detail": {k: v for k, v in cfg.items() if k != "Nt" and k != "Nr"},
    }
    with open(out / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out / 'DATA_HallII.npy'}  {arr.shape}")
    print(f"Saved: {out / 'dataset_info.json'}")
    print(f"\nPart 2 data summary:")
    print(f"  Config: {args.config}")
    print(f"  Maps: {n_maps}")
    print(f"  BS per map: 1 (dense deployment)")
    print(f"  UE per map: {len(ue_indices)}")
    print(f"  Total samples: {len(channels)}")
    print(f"  Shape: {arr.shape}")


if __name__ == "__main__":
    main()

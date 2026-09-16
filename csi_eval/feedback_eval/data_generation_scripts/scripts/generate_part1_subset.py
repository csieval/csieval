#!/usr/bin/env python3
"""Generate Part 1 (Scenario I) NEW SUBSET WAIR-D CSI feedback arrays.

This generates maps from Part 1 that are NOT in your existing dataset (maps 0-999).
For example, if your existing dataset used maps 0-999, this can generate maps 1000-9999.

Why this matters for cross-scenario eval:
  - Part 1 has 10000 maps from 40+ global cities
  - Maps 0-999 (your current data) are statistically equivalent to maps 1000-9999
  - Testing on Part1 NEW (maps 1000+) measures generalization to NEW Part1 environments
  - Unlike Part2, Part1 NEW keeps the same deployment config (5 BS × 30 UE)

Output:
  DATA_HoodI.npy  (OOD evaluation set, no train/val split)
  dataset_info.json

Usage:
  # Generate maps 1000-1099 (100 new Part1 maps)
  python scripts/generate_part1_subset.py --config 7GHz \
      --scenario-root data/Dataset/data/scenario_1 \
      --output data/Dataset/wair_d_output/7GHz_part1_new \
      --map-start 1000 --map-count 100

  # Generate maps 1000-1149 (150 new Part1 maps, 5 BS × 30 UE = 22500 samples)
  python scripts/generate_part1_subset.py --config 2_6GHz \
      --scenario-root data/Dataset/data/scenario_1 \
      --output data/Dataset/wair_d_output/2_6GHz_part1_new \
      --map-start 1000 --map-count 150

  # Quick test: just 10 maps
  python scripts/generate_part1_subset.py --config 7GHz \
      --scenario-root data/Dataset/data/scenario_1 \
      --output data/Dataset/wair_d_output/7GHz_part1_new_test \
      --map-start 1000 --map-count 10 --workers 4
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


def process_env(args):
    """Process a single Part 1 environment with multiple BS and UE positions."""
    env_dir, cfg, bs_list, ue_list = args
    env_dir = Path(env_dir)
    carrier = cfg["carrierFreq"]

    H_data = np.load(env_dir / f"H_{carrier}_G.npy", allow_pickle=True, encoding="latin1").item()
    P_data = np.load(env_dir / "Path.npy", allow_pickle=True, encoding="latin1").item()

    out = []
    for bs in bs_list:
        for ue in ue_list:
            key = f"bs{bs}_ue{ue}"
            out.append(process_single_channel(H_data[key], P_data[key], cfg))
    return out


def main():
    p = argparse.ArgumentParser(
        description="Generate Part 1 NEW SUBSET WAIR-D CSI feedback arrays. "
                    "Generate maps from Part 1 that are not in your existing dataset."
    )
    p.add_argument("--config", choices=list(CONFIGS), required=True)
    p.add_argument("--scenario-root", required=True,
                   help="Path to WAIR-D scenario_1 data directory")
    p.add_argument("--output", required=True,
                   help="Output directory for generated .npy files")
    p.add_argument("--map-start", type=int, required=True,
                   help="Starting map index (e.g. 1000 to skip maps 0-999)")
    p.add_argument("--map-count", type=int, default=100,
                   help="Number of maps to generate (default: 100)")
    p.add_argument("--bs-list", default="0,1,2,3,4",
                   help="BS locations per map (default: 0,1,2,3,4)")
    p.add_argument("--ue-list", default=",".join(str(i) for i in range(30)),
                   help="UE positions per BS (default: 0-29)")
    p.add_argument("--workers", type=int, default=min(mp.cpu_count(), 16))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    root = Path(args.scenario_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # All available Part 1 environments
    all_envs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name)
    )
    if not all_envs:
        raise FileNotFoundError(f"No numeric env folders in {root}")
    total_available = len(all_envs)
    print(f"Total Part 1 environments available: {total_available}")

    # Select the requested map range
    start = args.map_start
    end = start + args.map_count
    if end > total_available:
        print(f"WARNING: requested maps {start}-{end-1} but only {total_available} available. "
              f"Adjusting to maps {start}-{total_available-1} ({total_available - start} maps).")
        end = total_available

    selected_envs = all_envs[start:end]
    selected_ids = [int(e.name) for e in selected_envs]
    n_maps = len(selected_envs)
    print(f"Selected maps {start}-{end-1}: env IDs {selected_ids[0]}-{selected_ids[-1]} ({n_maps} maps)")

    bs_list = [int(x) for x in args.bs_list.split(",") if x]
    ue_list = [int(x) for x in args.ue_list.split(",") if x]
    samples_per_map = len(bs_list) * len(ue_list)
    total_samples = n_maps * samples_per_map

    print(f"BS per map: {bs_list} ({len(bs_list)} locations)")
    print(f"UE per BS: {len(ue_list)} positions")
    print(f"Samples per map: {samples_per_map}")
    print(f"Total samples: {total_samples}")

    # Build jobs
    jobs = [(str(e), cfg, bs_list, ue_list) for e in selected_envs]

    channels: List[np.ndarray] = []
    if args.workers <= 1:
        for j in tqdm(jobs, desc="Part1 NEW env"):
            channels.extend(process_env(j))
    else:
        with mp.Pool(args.workers) as pool:
            for res in tqdm(pool.imap(process_env, jobs),
                            total=len(jobs), desc="Part1 NEW env"):
                channels.extend(res)

    arr = np.asarray(channels, dtype=np.complex64)
    print(f"Generated array shape: {arr.shape}  ({arr.nbytes / 1e6:.1f} MB)")

    # Save as OOD evaluation set (no split)
    np.save(out / "DATA_HoodI.npy", arr)

    Nt_x, Nt_y, Nt_z = cfg["Nt"]
    info = {
        "config": args.config,
        "scenario": "I",
        "split": "ood_eval",
        "raw_shape": [Nt_x * Nt_y * (Nt_z // cfg["elements_per_port_z"]),
                      cfg["Nr"][0] * cfg["Nr"][1] * cfg["Nr"][2],
                      cfg["sampledCarriers"]],
        "output_convention": "[N, Nt, Nr, Nf] complex64",
        "map_range": [start, end],
        "map_env_ids": selected_ids,
        "n_maps": n_maps,
        "n_bs_per_map": len(bs_list),
        "n_ue_per_map": len(ue_list),
        "n_total": len(channels),
        "bs_list": bs_list,
        "ue_list": ue_list,
        "config_detail": {k: v for k, v in cfg.items() if k != "Nt" and k != "Nr"},
    }
    with open(out / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out / 'DATA_HoodI.npy'}  {arr.shape}")
    print(f"Saved: {out / 'dataset_info.json'}")
    print(f"\nPart 1 NEW subset summary:")
    print(f"  Config: {args.config}")
    print(f"  Map range: {start}-{end-1} (env IDs {selected_ids[0]}-{selected_ids[-1]})")
    print(f"  Maps: {n_maps}")
    print(f"  BS per map: {len(bs_list)}")
    print(f"  UE per BS: {len(ue_list)}")
    print(f"  Total samples: {len(channels)}")
    print(f"  Shape: {arr.shape}")
    print(f"\nRegister this dataset in config YAML:")
    print(f"  extra_datasets:")
    print(f"    - key: \"Part1_S1_{args.config}_ood_eval_maps{start}-{end-1}\"")
    print(f"      data_path: \"{out}\"")
    print(f"      config: \"{args.config}\"")
    print(f"      scenario: \"I\"")
    print(f"      split: \"ood_eval\"")
    print(f"      n_maps: {n_maps}")
    print(f"      n_bs_per_map: {len(bs_list)}")
    print(f"      n_ue_per_map: {len(ue_list)}")


if __name__ == "__main__":
    main()

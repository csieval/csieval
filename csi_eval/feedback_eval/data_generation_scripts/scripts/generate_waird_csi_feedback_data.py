#!/usr/bin/env python3
"""Generate WAIR-D CSI feedback arrays with shape [N, Nt, Nr, Nf].

This is a cleaned version of the old generator logic. It aligns with the EVM
configuration used by the YAML files and applies the 1-drive-4 port mapping on
BS z-axis antenna elements.

Expected raw WAIR-D layout:
  <scenario-root>/<env_id>/H_6_0_G.npy or H_2_6_G.npy
  <scenario-root>/<env_id>/Path.npy

Output:
  DATA_HtrainI.npy, DATA_HvalI.npy, DATA_HtestI.npy
where each file has complex64 shape [N, Nt_port, Nr_total, sampled_carriers].
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

# Avoid numpy oversubscription in multi-processing.
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


def array_response(angle, position, sorted_path):
    rx = np.sin(angle[sorted_path, 0]) * np.cos(angle[sorted_path, 1])
    ry = np.sin(angle[sorted_path, 0]) * np.sin(angle[sorted_path, 1])
    rz = np.cos(angle[sorted_path, 0])
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
    res_t = array_response(dod, pos_t, sorted_idx)  # [path, Nt_total]
    res_r = array_response(doa, pos_r, sorted_idx)  # [path, Nr_total]

    norm_H = H_path_gain[sorted_idx] / np.sqrt(subcarriers)
    ofdm_H = norm_H[:, None] * np.exp(-2j * np.pi * tau_sorted[:, None] * f_ghz[None, :])
    CFR = np.sum(ofdm_H[:, None, None, :] * res_t[:, :, None, None] * res_r[:, None, :, None], axis=0)
    # [Nt_total, Nr_total, Nf]

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
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS), required=True)
    p.add_argument("--scenario-root", required=True, help="Path to WAIR-D data/scenario_1")
    p.add_argument("--output", required=True)
    p.add_argument("--num-envs", type=int, default=1000)
    p.add_argument("--bs-list", default="0,1,2,3,4")
    p.add_argument("--ue-list", default=",".join(str(i) for i in range(30)))
    p.add_argument("--workers", type=int, default=min(mp.cpu_count(), 16))
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    root = Path(args.scenario_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bs_list = [int(x) for x in args.bs_list.split(",") if x]
    ue_list = [int(x) for x in args.ue_list.split(",") if x]
    env_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda p: int(p.name))
    env_dirs = env_dirs[: args.num_envs]
    if not env_dirs:
        raise FileNotFoundError(f"No numeric env folders found in {root}")

    jobs = [(str(e), cfg, bs_list, ue_list) for e in env_dirs]
    channels: List[np.ndarray] = []
    if args.workers <= 1:
        for j in tqdm(jobs, desc="env"):
            channels.extend(process_env(j))
    else:
        with mp.Pool(args.workers) as pool:
            for res in tqdm(pool.imap(process_env, jobs), total=len(jobs), desc="env"):
                channels.extend(res)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(channels))
    channels = [channels[i] for i in idx]
    n_total = len(channels)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    splits = {
        "train": channels[:n_train],
        "val": channels[n_train:n_train + n_val],
        "test": channels[n_train + n_val:],
    }
    for split, data in splits.items():
        arr = np.asarray(data, dtype=np.complex64)
        np.save(out / f"DATA_H{split}I.npy", arr)
        print(split, arr.shape, out / f"DATA_H{split}I.npy")

    Nt_x, Nt_y, Nt_z = cfg["Nt"]
    Nr_x, Nr_y, Nr_z = cfg["Nr"]
    info = {
        "config": args.config,
        "raw_shape": [Nt_x * Nt_y * (Nt_z // cfg["elements_per_port_z"]), Nr_x * Nr_y * Nr_z, cfg["sampledCarriers"]],
        "output_convention": "[N,Nt,Nr,Nf] complex64",
        "n_total": n_total,
        "splits": {k: len(v) for k, v in splits.items()},
        "config_detail": cfg,
    }
    with open(out / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("Saved dataset_info.json")


if __name__ == "__main__":
    main()

"""Markdown report writer."""

from __future__ import annotations

from pathlib import Path

from ..core.report import EvalReport


CATEGORY_LABELS = {
    "task_performance": "1. Task Performance",
    "storage": "2. Deployment & Storage",
    "computation": "3. Computation Efficiency",
    "robustness": "4. Robustness & Generalization",
    "comparison": "Comparison",
}


def save(report: EvalReport, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"

    lines: list[str] = []
    meta = report.meta
    _task_label = {"prediction": "Prediction", "feedback": "Feedback"}.get(
        meta.get("task", ""), "Feedback")
    lines.append(f"# CSI {_task_label} Evaluation Report")
    lines.append("")
    lines.append(f"- **Task**: `{meta.get('task', '?')}`")
    lines.append(f"- **Device**: `{meta.get('device', '?')}`")
    if "checkpoint" in meta and meta["checkpoint"]:
        lines.append(f"- **Checkpoint**: `{meta['checkpoint']}`")
    lines.append(f"- **Timestamp**: `{meta.get('timestamp', '?')}`")
    if "model_name" in meta:
        lines.append(f"- **Model**: `{meta['model_name']}`")
    if "model_size_mb" in meta:
        lines.append(f"- **Size**: `{meta['model_size_mb']:.2f} MB`")
    lines.append("")

    by_cat: dict[str, list] = {}
    for r in report.records:
        by_cat.setdefault(r.category, []).append(r)

    # Detect prediction format
    is_prediction = _detect_prediction_mode(report.sub_results)

    sub = report.sub_results or {}

    # Render task_performance: Prediction 模式下用 9 指标网格表格（与 HTML 一致）
    tp_sub = sub.get("task_performance") or {}
    has_tp_grid = bool(tp_sub) and any(
        isinstance(v, dict) and "metrics" in v for v in tp_sub.values()
    )

    if is_prediction and has_tp_grid:
        lines.append("## 1. Task Performance")
        lines.append("")
        for model_name, payload in tp_sub.items():
            lines.extend(_render_task_performance_grid_md(payload, model_name))
            lines.append("")
    else:
        # 非 prediction 模式：原 flat table 渲染
        for cat_key in ("task_performance", "storage", "computation"):
            recs = by_cat.get(cat_key, [])
            if not recs:
                continue
            lines.append(f"## {CATEGORY_LABELS[cat_key]}")
            lines.append("")
            lines.append("| Metric | Value | Unit | Direction |")
            lines.append("|---|---|---|---|")
            for r in recs:
                arrow = "↑ higher better" if r.higher_is_better else "↓ lower better"
                val_str = _fmt(r.value)
                lines.append(f"| `{r.name}` | {val_str} | {r.unit} | {arrow} |")
            lines.append("")

    if is_prediction:
        # Prediction 模式：跳过 robustness 已经在 _render_prediction_md 中渲染
        # 仍需渲染 storage / computation
        for cat_key in ("storage", "computation"):
            recs = by_cat.get(cat_key, [])
            if not recs:
                continue
            lines.append(f"## {CATEGORY_LABELS[cat_key]}")
            lines.append("")
            lines.append("| Metric | Value | Unit | Direction |")
            lines.append("|---|---|---|---|")
            for r in recs:
                arrow = "↑ higher better" if r.higher_is_better else "↓ lower better"
                val_str = _fmt(r.value)
                lines.append(f"| `{r.name}` | {val_str} | {r.unit} | {arrow} |")
            lines.append("")
        lines.extend(_render_prediction_md(sub, report.meta))
    else:
        snr_curve = sub.get("robustness.snr_nmse.per_snr") or sub.get("snr_nmse.per_snr") or sub.get("snr_nmse_curve")
        quant_curve = sub.get("robustness.quant.per_quant_bits") or sub.get("quant.per_quant_bits") or sub.get("quantization_robustness_curve")
        ood = sub.get("ood")

        has_snr = bool(snr_curve and isinstance(snr_curve, list))
        has_quant = bool(quant_curve and isinstance(quant_curve, list) and bool(quant_curve))
        has_ood = bool(ood)

        if has_snr or has_quant or has_ood:
            lines.append("## 4. Robustness & Generalization")
            lines.append("")
            lines.append("> Per 3GPP TR 38.843 V19.0.0 Section 6.2, **Case 1** is the in-distribution "
                         "baseline (train &amp; test on the same scenario). "
                         "**Case 2** (cross-scenario zero-shot) and **Case 3** "
                         "(cross-scenario fine-tune) are evaluated per OOD target in "
                         "Section 4.3 below.")
            lines.append("")

        # 4.1 Noise robustness
        if has_snr:
            lines.append("### 4.1 Noise Robustness (SNR sweep)")
            lines.append("")
            lines.append("| SNR (dB) | NMSE (dB) | SGCS |")
            lines.append("|---|---|---|")
            for pt in snr_curve:
                snr = pt.get("snr_db", pt.get("snr", 0))
                if snr >= 9999:
                    continue
                lines.append(f"| {snr:.1f} | {_fmt(pt.get('nmse_db'))} | {_fmt(pt.get('sgcs'))} |")
            lines.append("")

        # 4.2 Quantization robustness
        if has_quant:
            lines.append("### 4.2 Quantization Robustness (per bit-width)")
            lines.append("")
            lines.append("| Bit-width | NMSE (dB) | SGCS |")
            lines.append("|---|---|---|")
            for pt in quant_curve:
                lines.append(
                    f"| {pt.get('quant_bits', '?')} | {_fmt(pt.get('nmse_db'))} | {_fmt(pt.get('sgcs'))} |"
                )
            lines.append("")

        # 4.3 Cross-Scenario Evaluation
        if has_ood:
            lines.append("### 4.3 Cross-Scenario Evaluation")
            lines.append("")
            lines.append("> **All Cross-Scenario Evaluation uses 2-bit quantization.** "
                         "Aggregation: per-map mean. Case 3 values shown are 10-shot results.")
            lines.append("")

            def _find_n10(per_n):
                if not per_n:
                    return None
                for row in per_n:
                    if row.get("n_support") == 10:
                        return row
                best, best_diff = None, float("inf")
                for row in per_n:
                    d = abs(row.get("n_support", 0) - 10)
                    if d < best_diff:
                        best_diff = d
                        best = row
                return best

            source_baseline = {}
            for tgt_sub in ood.values():
                fs = (tgt_sub.get("metrics") or {}).get("fine_tune") or {}
                c1_nmse = fs.get("case1_nmse_db")
                c1_sgcs = fs.get("case1_sgcs")
                if c1_nmse is not None:
                    source_baseline["nmse_db"] = float(c1_nmse)
                if c1_sgcs is not None:
                    source_baseline["sgcs"] = float(c1_sgcs)

            for tgt_name, tgt_sub in ood.items():
                tgt_meta = tgt_sub.get("target", {}) or {}
                desc = tgt_meta.get("description") or f"path={tgt_meta.get('path', '?')}"
                lines.append(f"#### {tgt_name}")
                lines.append(f"_{desc}_")
                lines.append("")
                if "error" in tgt_sub:
                    lines.append(f"> **Error:** {tgt_sub['error']}")
                    lines.append("")
                    continue
                m = tgt_sub.get("metrics", {}) or {}

                c1 = m.get("id_baseline") or {}
                c2 = m.get("zero_shot") or {}
                c3 = m.get("fine_tune") or {}
                per_n = c3.get("per_n", []) or []

                sb_nmse = source_baseline.get("nmse_db") or c1.get("nmse_db")
                sb_sgcs = source_baseline.get("sgcs") or c1.get("sgcs")
                c2_nmse = c2.get("nmse_db")
                c2_sgcs = c2.get("sgcs")
                c2_std_nmse = c2.get("std_nmse_db")
                c2_std_sgcs = c2.get("std_sgcs")
                c1_std_nmse = c1.get("std_nmse_db")
                c1_std_sgcs = c1.get("std_sgcs")

                n10 = _find_n10(per_n)
                if n10:
                    c3_nmse = n10.get("nmse_db")
                    c3_sgcs = n10.get("sgcs")
                    c3_n = n10.get("n_support", 10)
                    c3_tag = f" ({c3_n}-shot)"
                else:
                    c3_nmse = c3.get("best_nmse_db")
                    c3_sgcs = c3.get("best_sgcs")
                    c3_tag = ""

                gap_nmse_c2 = (c2_nmse - sb_nmse) if (c2_nmse is not None and sb_nmse is not None) else None
                gap_nmse_c3 = (c3_nmse - sb_nmse) if (c3_nmse is not None and sb_nmse is not None) else None
                gdr_c2 = ((c2_sgcs - sb_sgcs) / sb_sgcs) if (c2_sgcs is not None and sb_sgcs is not None and sb_sgcs != 0) else None
                gdr_c3 = ((c3_sgcs - sb_sgcs) / sb_sgcs) if (c3_sgcs is not None and sb_sgcs is not None and sb_sgcs != 0) else None

                lines.append("| Metric | Baseline | Cross-scenario Zero-Shot | Cross-scenario Fine-Tune |")
                lines.append("|---|---|---|---|")
                lines.append(f"| NMSE — Per-Map Mean (dB) | `{_opt(sb_nmse)}` | `{_opt(c2_nmse)}` | `{_opt(c3_nmse)}`{c3_tag} |")
                lines.append(f"| SGCS — Per-Map Mean | `{_opt(sb_sgcs)}` | `{_opt(c2_sgcs)}` | `{_opt(c3_sgcs)}`{c3_tag} |")
                lines.append(f"| Per-Map NMSE — Std (dB) | `{_opt(c1_std_nmse)}` | `{_opt(c2_std_nmse)}` | `—` |")
                lines.append(f"| Per-Map SGCS — Std | `{_opt(c1_std_sgcs)}` | `{_opt(c2_std_sgcs)}` | `—` |")
                lines.append(f"| Gap NMSE (dB) | `—` | `{_opt(gap_nmse_c2)}` | `{_opt(gap_nmse_c3)}` |")
                lines.append(f"| SGCS Generalized Decay Rate | `—` | `{_opt(gdr_c2)}` | `{_opt(gdr_c3)}` |")
                lines.append("")

                if per_n:
                    lines.append(f"**Few-Shot Fine-Tuning Curve (n_support=[0,5,10,20,50,100,300])**")
                    lines.append("")
                    lines.append("| n_support | NMSE (dB) | SGCS |")
                    lines.append("|---|---|---|")
                    for r in per_n:
                        n = r.get("n_support", "?")
                        tag = " _(Zero-Shot)_" if r.get("is_zeroshot") else ""
                        nmse = _opt(r.get("nmse_db"))
                        sgcs_v = r.get("sgcs")
                        sgcs_str = _opt(sgcs_v) if sgcs_v is not None and sgcs_v == sgcs_v else "—"
                        lines.append(f"| {n}{tag} | {nmse} | {sgcs_str} |")
                    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _render_task_performance_grid_md(payload: Dict[str, Any], model_name: str) -> list:
    """渲染 Prediction 模式下的 Task Performance 9 指标 Markdown 表格。

    表结构（与 HTML 一致）：
      - 行：task in ('joint', 'spatial', 'frequency')
      - 列：mask_ratio in (0.25, 0.5, 0.75)
      - 输出两张表：NMSE / SGCS

    Args:
        payload: sub_results['task_performance'][model_name]，形如
                 {'metrics': {...}, 'tasks': [...], 'mask_ratios': [...]}
        model_name: 当前模型名（用于标题）
    """
    lines = []
    tasks = payload.get('tasks') or ('joint', 'spatial', 'frequency')
    ratios = payload.get('mask_ratios') or (0.25, 0.5, 0.75)
    metrics = payload.get('metrics') or {}

    def _lookup(task: str, ratio: float):
        key = f"{task}@{ratio}"
        return metrics.get(key)

    task_labels = {
        'joint': 'Joint (Spatial-Frequency)',
        'spatial': 'Spatial-only',
        'frequency': 'Frequency-only',
    }

    ratio_pcts = [f"{int(round(r * 100))}%" for r in ratios]
    header = "| Task \\ Mask Ratio | " + " | ".join(ratio_pcts) + " | Direction |"
    sep = "|---|" + "|".join(["---"] * len(ratios)) + "|---|"

    # NMSE 表
    lines.append("### 1.1 NMSE (dB) — ↓ lower better")
    lines.append("")
    lines.append("> **mask_ratio 语义**：mask 比例（被 mask 的 cells 占比），"
                 "值越大表示输入越稀疏。**预期方向**：mask_ratio 越大，NMSE 应越接近 0（性能越差）。")
    lines.append("")
    lines.append(f"**Model**: {model_name}")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for task in tasks:
        cells = []
        for r in ratios:
            entry = _lookup(task, r)
            v = entry.get('nmse_db') if entry else None
            cells.append(_fmt(v) if v is not None else "—")
        tlabel = task_labels.get(task, task)
        lines.append(f"| `{tlabel}` | " + " | ".join(cells) + " | ↓ lower better |")

    lines.append("")

    # SGCS 表
    lines.append("### 1.2 SGCS — ↑ higher better")
    lines.append("")
    lines.append("> **mask_ratio 语义**：同上。**预期方向**：mask_ratio 越大，SGCS 应越小。")
    lines.append("")
    lines.append(f"**Model**: {model_name}")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for task in tasks:
        cells = []
        for r in ratios:
            entry = _lookup(task, r)
            v = entry.get('sgcs') if entry else None
            cells.append(_fmt(v) if v is not None else "—")
        tlabel = task_labels.get(task, task)
        lines.append(f"| `{tlabel}` | " + " | ".join(cells) + " | ↑ higher better |")


    # 四个 SGCS stream/layer：stream index 按协方差特征值从大到小排列。
    lines.append("### 1.3 SGCS per stream — covariance eigenvalue order")
    lines.append("")
    lines.append("> Stream 1→4 按各子带平均协方差矩阵的特征值/奇异值从大到小定义；"
                 "逐流 SGCS 分数本身不做数值排序，以保持 layer-to-layer 比较语义。")
    lines.append("")
    lines.append(header.replace("Task \\ Mask Ratio", "Task / Stream \\ Mask Ratio"))
    lines.append(sep)
    for task in tasks:
        tlabel = task_labels.get(task, task)
        for stream_idx in range(4):
            cells = []
            for r in ratios:
                entry = _lookup(task, r)
                streams = (entry or {}).get('sgcs_streams') or []
                v = streams[stream_idx] if stream_idx < len(streams) else None
                cells.append(_fmt(v) if v is not None else "—")
            lines.append(
                f"| `{tlabel} / S{stream_idx + 1}` | " + " | ".join(cells) +
                " | eigenvalue ↓ |"
            )

    lines.append("")
    return lines


def _detect_prediction_mode(sub: dict) -> bool:
    """检测是否为 prediction 格式"""
    rob = sub.get("robustness", {})
    if isinstance(rob, dict):
        if any(k in rob for k in ("cross_mask", "cross_ratio_degradation",
                                   "cross_scenario", "noise_robustness", "snr_curve")):
            return True
        for v in rob.values():
            if isinstance(v, dict) and any(k in v for k in (
                "cross_mask", "cross_ratio_degradation", "cross_scenario", "noise_robustness"
            )):
                return True
    return False


def _render_prediction_md(sub: dict, meta: dict) -> list:
    """渲染 prediction 格式的鲁棒性 Markdown 内容"""
    lines = []
    rob = sub.get("robustness", {})

    model_names = list(rob.keys()) if isinstance(rob, dict) and not any(
        isinstance(v, dict) for v in rob.values()
    ) else [k for k, v in rob.items() if isinstance(v, dict)]
    if not model_names:
        model_names = [meta.get("model_name", "model")]

    for model_name in model_names:
        rdict = rob.get(model_name, rob) if isinstance(rob, dict) else {}

        lines.append("## 4. Robustness & Generalization")
        lines.append("")

        cm = rdict.get("cross_mask", {})
        if cm:
            lines.append("### 4.1 Cross-Mask Generalization")
            lines.append("")
            lines.append("| Mask Pair | ΔNMSE (dB) | ΔSGCS Avg | ΔS1 | ΔS2 | ΔS3 | ΔS4 |")
            lines.append("|---|---|---|---|---|---|---|")
            for pair, vals in cm.items():
                streams = list(vals.get('delta_sgcs_streams') or [])[:4] + [None] * 4
                lines.append(
                    f"| `{pair}` | {_opt(vals.get('delta_nmse'))} | {_opt(vals.get('delta_sgcs'))} | "
                    + " | ".join(_opt(v) for v in streams[:4]) + " |"
                )
            lines.append("")

        crd = rdict.get("cross_ratio_degradation", {})
        if crd:
            lines.append("### 4.2 Cross Mask-Ratio Generalization")
            lines.append("")
            lines.append("> **统一语义**：mask_ratio = mask 比例（mask 占比越高，模型越差）。"
                         "本节中 (高→低) 表示**先评估较高 mask_ratio，再评估较低 mask_ratio**，"
                         "ΔNMSE = NMSE(低 mask_ratio) − NMSE(高 mask_ratio)。"
                         "负值 ΔNMSE 表示**降低 mask 比例带来性能提升**。")
            lines.append("")
            lines.append("| Mask Ratio Pair | ΔNMSE (dB) | ΔSGCS Avg | ΔS1 | ΔS2 | ΔS3 | ΔS4 |")
            lines.append("|---|---|---|---|---|---|---|")
            for pair, vals in crd.items():
                streams = list(vals.get('delta_sgcs_streams') or [])[:4] + [None] * 4
                lines.append(
                    f"| `{pair}` | {_opt(vals.get('delta_nmse'))} | {_opt(vals.get('delta_sgcs'))} | "
                    + " | ".join(_opt(v) for v in streams[:4]) + " |"
                )
            lines.append("")

        nr = rdict.get("noise_robustness", {})
        if nr:
            snr_keys = sorted(
                [k for k in nr.keys() if k.startswith("snr_")],
                key=lambda k: float(k.split("_")[1]) if "_" in k else float("inf")
            )
            if snr_keys:
                lines.append("### 4.3 Noise Robustness (SNR sweep)")
                lines.append("")
                lines.append("| SNR (dB) | NMSE (dB) | SGCS Avg | S1 | S2 | S3 | S4 |")
                lines.append("|---|---|---|---|---|---|---|")
                for k in snr_keys:
                    snr_str = k.split("_", 1)[1]
                    if snr_str == "inf":
                        continue
                    try:
                        snr_val = float(snr_str)
                    except ValueError:
                        continue
                    pt = nr[k]
                    streams = list(pt.get('sgcs_streams') or [])[:4]
                    streams += [None] * (4 - len(streams))
                    lines.append(
                        f"| {snr_val:.0f} | {_fmt(pt.get('nmse_db'))} | {_fmt(pt.get('sgcs'))} | "
                        + " | ".join(_fmt(v) for v in streams) + " |"
                    )
                lines.append("")

        cs = rdict.get("cross_scenario", {})
        if cs:
            lines.append("### 4.4 Cross-Scenario (S1 → S2)")
            lines.append("")
            lines.append("| Scenario | NMSE (dB) | SGCS Avg | S1 | S2 | S3 | S4 |")
            lines.append("|---|---|---|---|---|---|---|")
            s1_streams = list(cs.get('s1_sgcs_streams') or [])[:4] + [None] * 4
            s2_streams = list(cs.get('s2_sgcs_streams') or [])[:4] + [None] * 4
            lines.append(
                f"| S1 (in-distribution) | {_fmt(cs.get('s1_nmse_db'))} | {_fmt(cs.get('s1_sgcs'))} | "
                + " | ".join(_fmt(v) for v in s1_streams[:4]) + " |"
            )
            lines.append(
                f"| S2 (cross-scenario) | {_fmt(cs.get('s2_nmse_db'))} | {_fmt(cs.get('s2_sgcs'))} | "
                + " | ".join(_fmt(v) for v in s2_streams[:4]) + " |"
            )
            lines.append(f"| **Δ (S2−S1)** | **{_fmt(cs.get('delta_nmse'))}** | **{_fmt(cs.get('delta_sgcs_percent'))}%** | — | — | — | — |")
            lines.append("")

    return lines


def _opt(v) -> str:
    """Format a value for the OOD MD table: float with 4dp, None → em-dash."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.3e}"
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(val)}" for k, val in list(v.items())[:5])
    if isinstance(v, list):
        if len(v) > 5:
            return f"[{', '.join(_fmt(x) for x in v[:3])}…({len(v)} items)]"
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)

"""HTML report writer (single self-contained file, with embedded Plotly).

Falls back to plain tables if plotly is not available.

Layout
------
1. Task Performance
2. Deployment & Storage
3. Computation Efficiency
4. Generalization & Robustness (covers noise robustness, quantization,
   OOD cross-scenario, per-map breakdown and few-shot recovery in a
   single section, per the user's spec)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.report import EvalReport


CATEGORY_LABELS = {
    "task_performance": "1. Task Performance",
    "storage": "2. Deployment & Storage",
    "computation": "3. Computation Efficiency",
    "robustness": "4. Robustness & Generalization",
    "comparison": "Comparison",
}


_PLOTLY_HEAD = """
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
"""


def _load_icon_data_uri() -> str:
    """Load csi_feedback_eval/static/icon.png and return a data: URI string.

    Returns an empty string if the icon file is missing or unreadable.
    """
    candidates = [
        Path(__file__).resolve().parent.parent / "static" / "icon.png",
        Path(__file__).resolve().parent / "static" / "icon.png",
    ]
    for p in candidates:
        if p.is_file():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                return f"data:image/png;base64,{b64}"
            except OSError:
                return ""
    return ""


def save(report: EvalReport, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.html"

    by_cat: Dict[str, List] = {k: [] for k in CATEGORY_LABELS}
    for r in report.records:
        by_cat.setdefault(r.category, []).append(r)

    sub = report.sub_results or {}
    snr_curve = sub.get("robustness.snr_nmse.per_snr") or sub.get("snr_nmse.per_snr") or sub.get("snr_nmse_curve")
    quant_curve = sub.get("robustness.quant.per_quant_bits") or sub.get("quant.per_quant_bits") or sub.get("quantization_robustness_curve")

    icon_uri = _load_icon_data_uri()

    html: List[str] = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append("<title>CSI Evaluation Report</title>")
    html.append(_PLOTLY_HEAD)
    html.append(_css(icon_uri))
    html.append("</head><body>")
    html.append(_hero(report.meta, icon_uri))

    # 探测是否为 prediction 格式（robustness.sub_results 含 cross_mask / noise_robustness）
    is_prediction = _detect_prediction_mode(sub)

    # Tables for the first three categories (Task / Deployment / Computation)
    for cat_key in ("task_performance", "storage", "computation"):
        recs = by_cat.get(cat_key, [])
        sub_tp = sub.get("task_performance", {}) if cat_key == "task_performance" else None
        has_grid = bool(sub_tp) and isinstance(sub_tp, dict) and any(
            isinstance(v, dict) and "grid" in v
            for v in sub_tp.values()
        )
        if cat_key == "task_performance" and has_grid:
            # Prediction 模式：使用 9 指标 (spatial/frequency/joint × 3 ratios) 表格渲染
            html.append(f"<section class='card'><h2>{CATEGORY_LABELS[cat_key]}</h2>")
            html.append(_table_for_task_performance_grid(sub_tp))
            html.append("</section>")
            continue
        if not recs:
            continue
        html.append(f"<section class='card'><h2>{CATEGORY_LABELS[cat_key]}</h2>")
        html.append(_table_for_records(recs))
        html.append("</section>")

    if is_prediction:
        chart_idx = _render_prediction_robustness(html, sub, report.meta)
    else:
        # ---- Section 4: Robustness & Generalization (feedback OOD style) ----
        chart_idx = 0
        ood = sub.get("ood")
        has_snr = bool(snr_curve and isinstance(snr_curve, list))
        has_quant = bool(quant_curve and isinstance(quant_curve, list) and len(quant_curve) > 0)
        has_ood = bool(ood)

        if has_snr or has_quant or has_ood:
            html.append(f"<section class='card'><h2>{CATEGORY_LABELS['robustness']}</h2>")
            html.append("<div class='callout'>"
                        "<b>Per 3GPP TR 38.843 V19.0.0 §6.2</b> — "
                        "<b>Case 2</b> (cross-scenario zero-shot) and <b>Case 3</b> "
                        "(cross-scenario fine-tune) are evaluated per OOD target in "
                        "Section 4.3 below. Sections 4.1–4.2 (Noise/Quantization "
                        "robustness) use full precision."
                        "</div>")

        # 4a. Noise robustness: table + NMSE & SGCS vs SNR chart
        if has_snr:
            chart_idx += 1
            html.append("<div class='subcard'><h3>4.1 Noise Robustness <span class='h3-tag'>SNR sweep</span></h3>")
            snrs = [pt.get("snr_db", pt.get("snr", 0)) for pt in snr_curve if pt.get("snr_db", 9999) < 9999]
            nmse = [pt.get("nmse_db", 0) for pt in snr_curve if pt.get("snr_db", 9999) < 9999]
            sgcs_vals = [pt.get("sgcs", None) for pt in snr_curve if pt.get("snr_db", 9999) < 9999]
            html.append("<table>")
            html.append("<thead><tr><th>SNR (dB)</th><th>NMSE (dB)</th><th>SGCS</th></tr></thead><tbody>")
            for pt in snr_curve:
                snr = pt.get("snr_db", pt.get("snr", 0))
                if snr >= 9999:
                    continue
                html.append(
                    f"<tr><td>{snr:.1f}</td>"
                    f"<td>{_fmt(pt.get('nmse_db'))}</td>"
                    f"<td>{_fmt(pt.get('sgcs'))}</td></tr>"
                )
            html.append("</tbody></table>")
            traces = [{"x": snrs, "y": nmse, "mode": "lines+markers", "name": "NMSE (dB)", "yaxis": "y1"}]
            if all(s is not None for s in sgcs_vals):
                traces.append({
                    "x": snrs, "y": sgcs_vals, "mode": "lines+markers",
                    "name": "SGCS", "yaxis": "y2",
                })
                html.append(f"<div id='c{chart_idx}' class='chart'></div>")
                html.append(_plotly(
                    f"c{chart_idx}", traces,
                    "NMSE / SGCS vs SNR",
                    "SNR (dB)", "NMSE (dB)",
                    secondary_y=True,
                ))
            else:
                html.append(f"<div id='c{chart_idx}' class='chart'></div>")
                html.append(_plotly(
                    f"c{chart_idx}", traces,
                    "NMSE vs SNR",
                    "SNR (dB)", "NMSE (dB)",
                ))
            html.append("</div>")

        # 4b. Quantization robustness
        if has_quant:
            chart_idx += 1
            html.append("<div class='subcard'><h3>4.2 Quantization Robustness <span class='h3-tag'>per bit-width</span></h3>")
            html.append("<table>")
            html.append("<thead><tr><th>Bit-width</th><th>NMSE (dB)</th><th>SGCS</th></tr></thead><tbody>")
            for pt in quant_curve:
                qb = pt.get("quant_bits", "?")
                html.append(
                    f"<tr><td><code>{qb}</code></td>"
                    f"<td>{_fmt(pt.get('nmse_db'))}</td>"
                    f"<td>{_fmt(pt.get('sgcs'))}</td></tr>"
                )
            html.append("</tbody></table>")
            bits = [pt["quant_bits"] for pt in quant_curve]
            nmse = [pt["nmse_db"] for pt in quant_curve]
            sgcs = [pt.get("sgcs", 0) for pt in quant_curve]
            html.append(f"<div id='c{chart_idx}' class='chart'></div>")
            html.append(_plotly(
                f"c{chart_idx}",
                [
                    {"x": bits, "y": nmse, "mode": "lines+markers",
                     "name": "NMSE (dB)", "yaxis": "y1"},
                    {"x": bits, "y": sgcs, "mode": "lines+markers",
                     "name": "SGCS", "yaxis": "y2"},
                ],
                "NMSE / SGCS vs Quantization Bit-width",
                "Quant bits", "NMSE (dB)",
                secondary_y=True,
            ))
            html.append("</div>")

        chart_idx = _render_ood_section(html, report, start_chart_idx=chart_idx,
                                        source_baseline={})

        if has_snr or has_quant or has_ood:
            html.append("</section>")

    # Skipped (kept at the end as a single block)
    if report.skipped:
        html.append("<section class='card'><h2>Skipped</h2><ul class='skipped'>")
        for s in report.skipped:
            html.append(f"<li><code>{s['name']}</code>: {s['reason']}</li>")
        html.append("</ul></section>")

    html.append("</div>")  # close .container
    html.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_path


def _detect_prediction_mode(sub: Dict[str, Any]) -> bool:
    """检测是否为 prediction 格式（robustness.sub_results 含 cross_mask 等字段）"""
    rob = sub.get("robustness", {})
    if isinstance(rob, dict):
        # prediction 格式的 robustness 包含 cross_mask / cross_ratio_degradation / cross_scenario
        if any(k in rob for k in ("cross_mask", "cross_ratio_degradation", "cross_scenario",
                                   "noise_robustness", "snr_curve")):
            return True
        # 也有可能是 {model_name: {...}} 嵌套格式
        for v in rob.values():
            if isinstance(v, dict) and any(k in v for k in (
                "cross_mask", "cross_ratio_degradation", "cross_scenario", "noise_robustness"
            )):
                return True
    return False


def _render_prediction_robustness(html: List[str], sub: Dict[str, Any],
                                   meta: Dict[str, Any]) -> int:
    """渲染 prediction 格式的鲁棒性子结果（Cross-mask / SNR / Cross-scenario）"""
    html.append(f"<section class='card'><h2>{CATEGORY_LABELS['robustness']}</h2>")

    chart_idx = 0

    # 提取 {model_name: robustness_dict}
    rob = sub.get("robustness", {})
    # 支持直接是 dict 或 {model_name: dict} 嵌套
    model_names = list(rob.keys()) if isinstance(rob, dict) and not any(
        isinstance(v, dict) for v in rob.values()
    ) else [k for k, v in rob.items() if isinstance(v, dict)]
    if not model_names:
        model_names = list(meta.get("model_names", [meta.get("model_name", "model")]))

    for model_name in model_names:
        rdict = rob.get(model_name, rob) if isinstance(rob, dict) else {}

        # Cross-mask Δ
        cm = rdict.get("cross_mask", {})
        if cm:
            html.append("<div class='subcard'><h3>Cross-Mask Generalization</h3>")
            html.append("<table>")
            html.append("<thead><tr><th>Mask Pair</th><th>ΔNMSE (dB)</th>"
                       "<th>ΔSGCS Avg</th><th>ΔS1</th><th>ΔS2</th><th>ΔS3</th><th>ΔS4</th>"
                       "<th>Direction</th></tr></thead><tbody>")
            for pair, vals in cm.items():
                dn = vals.get("delta_nmse")
                ds = vals.get("delta_sgcs")
                streams = list(vals.get("delta_sgcs_streams") or [])[:4] + [None] * 4
                arrow = "↓ lower better" if dn is not None and dn > 0 else "↑ higher better"
                stream_cells = "".join(f"<td>{_fmt(v)}</td>" for v in streams[:4])
                html.append(
                    f"<tr><td><code>{pair}</code></td>"
                    f"<td>{_fmt(dn)}</td><td>{_fmt(ds)}</td>{stream_cells}"
                    f"<td>{arrow}</td></tr>"
                )
            html.append("</tbody></table></div>")

        # Cross-ratio degradation
        crd = rdict.get("cross_ratio_degradation", {})
        if crd:
            html.append("<div class='subcard'><h3>Cross Mask-Ratio Generalization</h3>")
            html.append("<table>")
            html.append("<thead><tr><th>Mask Ratio Pair</th><th>ΔNMSE (dB)</th>"
                       "<th>ΔSGCS Avg</th><th>ΔS1</th><th>ΔS2</th><th>ΔS3</th><th>ΔS4</th>"
                       "<th>Direction</th></tr></thead><tbody>")
            for pair, vals in crd.items():
                dn = vals.get("delta_nmse")
                ds = vals.get("delta_sgcs")
                streams = list(vals.get("delta_sgcs_streams") or [])[:4] + [None] * 4
                arrow = "↓ lower better" if dn is not None and dn > 0 else "↑ higher better"
                stream_cells = "".join(f"<td>{_fmt(v)}</td>" for v in streams[:4])
                html.append(
                    f"<tr><td><code>{pair}</code></td>"
                    f"<td>{_fmt(dn)}</td><td>{_fmt(ds)}</td>{stream_cells}"
                    f"<td>{arrow}</td></tr>"
                )
            html.append("</tbody></table></div>")

        # SNR robustness
        nr = rdict.get("noise_robustness", {})
        if nr:
            # 收集 SNR 列表
            snr_keys = sorted(
                [k for k in nr.keys() if k.startswith("snr_")],
                key=lambda k: float(k.split("_")[1]) if "_" in k else float("inf")
            )
            if snr_keys:
                chart_idx += 1
                html.append("<div class='subcard'><h3>Noise Robustness <span class='h3-tag'>SNR sweep</span></h3>")
                html.append("<table>")
                html.append("<thead><tr><th>SNR (dB)</th><th>NMSE (dB)</th><th>SGCS Avg</th>"
                            "<th>S1</th><th>S2</th><th>S3</th><th>S4</th></tr></thead><tbody>")
                snrs, nmse_vals, sgcs_vals = [], [], []
                for k in snr_keys:
                    snr_str = k.split("_", 1)[1]
                    if snr_str == "inf":
                        continue
                    try:
                        snr_val = float(snr_str)
                    except ValueError:
                        continue
                    pt = nr[k]
                    snrs.append(snr_val)
                    nmse_vals.append(pt.get("nmse_db"))
                    sgcs_vals.append(pt.get("sgcs"))
                    streams = list(pt.get("sgcs_streams") or [])[:4] + [None] * 4
                    stream_cells = "".join(f"<td>{_fmt(v)}</td>" for v in streams[:4])
                    html.append(
                        f"<tr><td>{snr_val:.0f}</td>"
                        f"<td>{_fmt(pt.get('nmse_db'))}</td>"
                        f"<td>{_fmt(pt.get('sgcs'))}</td>{stream_cells}</tr>"
                    )
                html.append("</tbody></table>")
                if snrs:
                    traces = [{"x": snrs, "y": nmse_vals,
                               "mode": "lines+markers", "name": "NMSE (dB)", "yaxis": "y1"}]
                    if all(s is not None for s in sgcs_vals):
                        traces.append({"x": snrs, "y": sgcs_vals,
                                       "mode": "lines+markers", "name": "SGCS", "yaxis": "y2"})
                        html.append(f"<div id='c{chart_idx}' class='chart'></div>")
                        html.append(_plotly(
                            f"c{chart_idx}", traces,
                            "NMSE / SGCS vs SNR",
                            "SNR (dB)", "NMSE (dB)",
                            secondary_y=True,
                        ))
                    else:
                        html.append(f"<div id='c{chart_idx}' class='chart'></div>")
                        html.append(_plotly(
                            f"c{chart_idx}", traces,
                            "NMSE vs SNR",
                            "SNR (dB)", "NMSE (dB)",
                        ))
                html.append("</div>")

        # Cross-scenario (S1→S2)
        cs = rdict.get("cross_scenario", {})
        if cs:
            html.append("<div class='subcard'><h3>Cross-Scenario (S1 → S2)</h3>")
            html.append("<table>")
            html.append("<thead><tr><th>Scenario</th><th>NMSE (dB)</th>"
                       "<th>SGCS Avg</th><th>S1</th><th>S2</th><th>S3</th><th>S4</th>"
                       "<th>Direction</th></tr></thead><tbody>")
            s1n, s2n = cs.get("s1_nmse_db"), cs.get("s2_nmse_db")
            s1s, s2s = cs.get("s1_sgcs"), cs.get("s2_sgcs")
            dn = cs.get("delta_nmse")
            ds_pct = cs.get("delta_sgcs_percent")
            arrow = "↓ lower better" if dn is not None and dn > 0 else "↑ higher better"
            s1_streams = list(cs.get("s1_sgcs_streams") or [])[:4] + [None] * 4
            s2_streams = list(cs.get("s2_sgcs_streams") or [])[:4] + [None] * 4
            s1_cells = "".join(f"<td>{_fmt(v)}</td>" for v in s1_streams[:4])
            s2_cells = "".join(f"<td>{_fmt(v)}</td>" for v in s2_streams[:4])
            html.append(f"<tr><td>S1 (in-distribution)</td>"
                        f"<td>{_fmt(s1n)}</td><td>{_fmt(s1s)}</td>{s1_cells}<td>baseline</td></tr>")
            html.append(f"<tr><td>S2 (cross-scenario)</td>"
                        f"<td>{_fmt(s2n)}</td><td>{_fmt(s2s)}</td>{s2_cells}<td>baseline</td></tr>")
            html.append(f"<tr><td>Δ (S2−S1)</td>"
                        f"<td>{_fmt(dn)}</td><td>{_fmt(ds_pct)}%</td>"
                        f"<td>—</td><td>—</td><td>—</td><td>—</td><td>{arrow}</td></tr>")
            html.append("</tbody></table></div>")

    html.append("</section>")
    return chart_idx


def _table_for_records(recs: List) -> str:
    """Render a 4-column table for a list of MetricRecord."""
    out = ["<table>"]
    out.append("<thead><tr><th>Metric</th><th>Value</th><th>Unit</th><th>Direction</th></tr></thead><tbody>")
    for r in recs:
        if r.higher_is_better:
            arrow = '<span class="arrow-up" title="higher is better">↑</span>'
        else:
            arrow = '<span class="arrow-dn" title="lower is better">↓</span>'
        out.append(
            f"<tr><td><code>{r.name}</code></td><td>{_fmt(r.value)}</td>"
            f"<td>{r.unit}</td><td>{arrow}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _table_for_task_performance_grid(sub_tp: Dict[str, Any]) -> str:
    """渲染 Prediction 模式下的 Task Performance 9 指标表格。

    表结构：
      - 三个表格，分别对应 NMSE (dB)、SGCS 与汇总。
      - 行：task in ('joint', 'spatial', 'frequency')
      - 列：mask_ratio in (0.25, 0.5, 0.75)

    Args:
        sub_tp: ``sub_results['task_performance']``，形如
            {model_name: {'grid': [{...}, ...], 'tasks': [...], 'mask_ratios': [...]}}

    Returns:
        HTML 表格字符串。
    """
    if not sub_tp:
        return ""
    # 选取第一个模型作为数据来源（多模型场景下按需扩展）
    model_name, payload = next(iter(sub_tp.items()))
    tasks = payload.get('tasks') or ('joint', 'spatial', 'frequency')
    ratios = payload.get('mask_ratios') or (0.25, 0.5, 0.75)
    metrics = payload.get('metrics') or {}
    n_samples = payload.get('n_samples')

    def _lookup(task, ratio):
        key = f"{task}@{ratio}"
        if key in metrics:
            return metrics[key]
        # 兜底：从 grid 列表里找匹配项
        for entry in payload.get('grid', []) or []:
            if entry.get('task') == task and entry.get('mask_ratio') == ratio:
                return entry
        return None

    ratio_pcts = [f"{int(round(r * 100))}%" for r in ratios]

    # NMSE 表
    nmse_rows = []
    for task in tasks:
        cells = []
        for r in ratios:
            entry = _lookup(task, r)
            v = entry.get('nmse_db') if entry else None
            cells.append(_fmt(v))
        nmse_rows.append((task, cells))

    # SGCS 表
    sgcs_rows = []
    for task in tasks:
        cells = []
        for r in ratios:
            entry = _lookup(task, r)
            v = entry.get('sgcs') if entry else None
            cells.append(_fmt(v))
        sgcs_rows.append((task, cells))

    # SGCS 逐 stream/layer 表；stream index 按协方差特征值降序定义。
    sgcs_stream_rows = {i: [] for i in range(4)}
    for stream_idx in range(4):
        for task in tasks:
            cells = []
            for r in ratios:
                entry = _lookup(task, r)
                streams = (entry or {}).get('sgcs_streams') or []
                v = streams[stream_idx] if stream_idx < len(streams) else None
                cells.append(_fmt(v))
            sgcs_stream_rows[stream_idx].append((task, cells))

    def _render_table(title: str, unit: str, direction: str,
                      rows, arrow_cls: str) -> str:
        out = [f"<h3>{title}</h3>", "<table>"]
        header_cells = "".join(
            f"<th style='text-align:center;'>{pct}</th>" for pct in ratio_pcts
        )
        out.append(
            "<thead><tr><th style='text-align:left;'>Task \\ Mask Ratio</th>"
            f"{header_cells}<th style='text-align:center;'>Direction</th></tr></thead>"
        )
        out.append("<tbody>")
        for task, cells in rows:
            tlabel = {'joint': 'Joint (Spatial-Frequency)',
                      'spatial': 'Spatial-only',
                      'frequency': 'Frequency-only'}.get(task, task)
            cell_html = "".join(
                f"<td style='text-align:center;'><code>{c}</code></td>" for c in cells
            )
            out.append(
                f"<tr><td><code>{tlabel}</code></td>"
                f"{cell_html}"
                f"<td style='text-align:center;'><span class='{arrow_cls}' "
                f"title='{direction}'>{direction}</span></td></tr>"
            )
        out.append("</tbody></table>")
        return "\n".join(out)

    parts = []
    parts.append("<p class='hint'>")
    parts.append("</p>")
    parts.append(_render_table(
        "NMSE(dB)", "dB", "↓ lower is better",
        nmse_rows, "arrow-dn",
    ))
    parts.append(_render_table(
        "SGCS (average of 4 ordered streams)", "", "↑ higher is better",
        sgcs_rows, "arrow-up",
    ))
    for stream_idx in range(4):
        parts.append(_render_table(
            f"SGCS Stream {stream_idx + 1}", "", "↑ higher is better",
            sgcs_stream_rows[stream_idx], "arrow-up",
        ))
    return "\n".join(parts)


def _render_ood_section(html: List[str], report: EvalReport,
                        start_chart_idx: int = 0,
                        source_baseline: Optional[Dict[str, float]] = None) -> int:
    """Render the OOD (Cross-Scenario) section.

    Per target we emit:
      - A 6-row comparison table: 4 metrics × all 3 columns + 2 gap metrics × all 3 columns.
        Column headers: Baseline | Cross-scenario Zero-Shot | Cross-scenario Fine-Tune (10-shot).
      - A few-shot fine-tuning curve for n_support=[0,5,10,20,50,100,300].
      - A per-map scatter plot from Zero-Shot evaluation.

    All Cross-Scenario Evaluation uses 2-bit quantization.
    """
    ood = (report.sub_results or {}).get("ood")
    if not ood:
        return start_chart_idx
    html.append("<div class='subcard'><h3>4.3 Cross-Scenario Evaluation <span class='h3-tag'>2-bit quantization</span></h3>")
    html.append("<div class='callout'>"
                "<b>All Cross-Scenario Evaluation uses 2-bit quantization.</b> "
                "Aggregation: per-map mean (average of per-environment means). "
                "Case 3 values shown are 10-shot results."
                "</div>")

    chart_idx = start_chart_idx

    def _get_n10_row(per_n: List[Dict]) -> Optional[Dict]:
        """Find n_support=10 row; fall back to closest available n."""
        TARGET = 10
        if not per_n:
            return None
        for row in per_n:
            if row.get("n_support") == TARGET:
                return row
        # Fall back: closest n_support value
        best, best_diff = None, float("inf")
        for row in per_n:
            diff = abs(row.get("n_support", 0) - TARGET)
            if diff < best_diff:
                best_diff = diff
                best = row
        return best

    for tgt_name, sub in ood.items():
        tgt_meta = sub.get("target", {}) or {}
        desc = tgt_meta.get("description") or f"path={tgt_meta.get('path', '?')}"
        html.append(f"<h4>{tgt_name} <span style='color:#666; font-weight:normal; "
                    f"font-size:13px;'>— {desc}</span></h4>")
        if "error" in sub:
            html.append(f"<p style='color:#b91c1c;'><b>Error:</b> {sub['error']}</p>")
            continue
        m = sub.get("metrics", {}) or {}

        c1 = m.get("id_baseline") or {}
        c2 = m.get("zero_shot") or {}
        c3 = m.get("fine_tune") or {}

        # Read authoritative values from the metric records stored in the OOD sub-results.
        # The robustness runner stores each metric's result dict under sub["metrics"][name].
        gap_rec = m.get("gap_nmse") or {}
        decay_rec = m.get("sgcs_decay_rate") or {}

        per_n = c3.get("per_n", []) or []

        sb_nmse = c1.get("nmse_db") or c1.get("value")
        sb_sgcs = c1.get("sgcs")
        c2_nmse = c2.get("nmse_db") or c2.get("value")
        c2_sgcs = c2.get("sgcs")
        c2_std_nmse = c2.get("std_nmse_db")
        c2_std_sgcs = c2.get("std_sgcs")
        c1_std_nmse = c1.get("std_nmse_db")
        c1_std_sgcs = c1.get("std_sgcs")

        # Use authoritative values from metric records directly
        gap_nmse_c2 = gap_rec.get("value")
        gap_nmse_c3 = None  # gap is per target, not per shot; show in Zero-Shot column
        gdr_c2 = decay_rec.get("value")
        gdr_c3 = None

        # 10-shot values (default for Case 3 column)
        n10_row = _get_n10_row(per_n)
        if n10_row:
            c3_nmse = n10_row.get("nmse_db")
            c3_sgcs = n10_row.get("sgcs")
            c3_n = n10_row.get("n_support", 10)
            c3_tag = f" ({c3_n}-shot)"
        else:
            c3_nmse = c3.get("best_nmse_db")
            c3_sgcs = c3.get("best_sgcs")
            c3_tag = ""

        def _row(label, b_val, c2_val, c3_val, arrow, note=""):
            b_str = _opt(b_val)
            c2_str = _opt(c2_val)
            c3_str = _opt(c3_val) if c3_val is not None else "—"
            arrow_cls = "arrow-up" if arrow == "↑" else "arrow-dn"
            arrow_html = f'<span class="{arrow_cls}">{arrow}</span>'
            return (f"<tr><td>{label}{note}</td>"
                    f"<td><code>{b_str}</code></td>"
                    f"<td><code>{c2_str}</code></td>"
                    f"<td><code>{c3_str}</code></td>"
                    f"<td>{arrow_html}</td></tr>")

        html.append("<table>")
        html.append("<thead><tr><th style='width:32%;'>Metric</th>"
                   "<th>Baseline</th>"
                   "<th>Cross-scenario Zero-Shot</th>"
                   "<th>Cross-scenario Fine-Tune</th>"
                   "<th>Direction</th></tr></thead><tbody>")
        html.append(_row("NMSE — Per-Map Mean (dB)",
                        sb_nmse, c2_nmse, c3_nmse, "↓"))
        html.append(_row("SGCS — Per-Map Mean",
                        sb_sgcs, c2_sgcs, c3_sgcs, "↑"))
        html.append(_row("Per-Map NMSE — Std (dB)",
                        c1_std_nmse, c2_std_nmse, None, "↓"))
        html.append(_row("Per-Map SGCS — Std",
                        c1_std_sgcs, c2_std_sgcs, None, "↓"))
        html.append(_row("Gap NMSE (dB)",
                        None, gap_nmse_c2, None, "↓"))
        html.append(_row("SGCS Generalized Decay Rate",
                        None, gdr_c2, None, "↓"))
        html.append("</tbody></table>")

        # Few-shot curve: always plot n_support=[0,5,10,20,50,100,300]
        if per_n:
            chart_idx += 1
            html.append(f"<h5>Few-Shot Fine-Tuning Curve — {tgt_name}</h5>")
            html.append(f"<div id='c{chart_idx}' class='chart'></div>")
            n_support_vals = [r.get("n_support", i) for i, r in enumerate(per_n)]
            nmse_vals = [r.get("nmse_db") for r in per_n]
            sgcs_vals = [r.get("sgcs") for r in per_n]
            fewshot_traces = [
                {"x": n_support_vals, "y": nmse_vals,
                 "mode": "lines+markers", "name": "NMSE (dB)", "yaxis": "y1"},
            ]
            if all(v is not None and not (isinstance(v, float) and v != v) for v in sgcs_vals):
                fewshot_traces.append({
                    "x": n_support_vals, "y": sgcs_vals,
                    "mode": "lines+markers", "name": "SGCS", "yaxis": "y2",
                })
            # Horizontal reference lines: Baseline and Zero-Shot
            ref_lines = []
            def _add_ref(name, yval, yref_key):
                if yval is not None:
                    try:
                        ref_lines.append({
                            "type": "line", "xref": "paper", "yref": yref_key,
                            "x0": 0, "x1": 1,
                            "y0": float(yval), "y1": float(yval),
                            "line": {"color": "#888", "dash": "dash", "width": 1},
                            "name": f"{name}",
                        })
                    except (TypeError, ValueError):
                        pass
            _add_ref("Baseline NMSE", sb_nmse, "y1")
            _add_ref("Zero-Shot NMSE", c2_nmse, "y1")
            _add_ref("Baseline SGCS", sb_sgcs, "y2")
            _add_ref("Zero-Shot SGCS", c2_sgcs, "y2")
            html.append(_plotly(
                f"c{chart_idx}", fewshot_traces,
                f"Few-Shot Fine-Tuning — {tgt_name}",
                "n_support", "NMSE (dB)",
                secondary_y=True, ref_lines=ref_lines,
            ))

        # Per-map scatter plot (from Zero-Shot)
        per_env_nmse = c2.get("per_map_nmse_db")
        per_env_sgcs = c2.get("per_map_sgcs")
        if isinstance(per_env_nmse, list) and per_env_nmse:
            chart_idx += 1
            html.append(f"<h5>Per-Map Distribution — {tgt_name} (Zero-Shot, 2-bit)</h5>")
            html.append(f"<div id='c{chart_idx}' class='chart'></div>")
            traces: List[Dict[str, Any]] = [{
                "x": list(range(len(per_env_nmse))),
                "y": per_env_nmse,
                "mode": "markers",
                "name": "Per-map NMSE (dB)",
            }]
            sec_y = False
            if isinstance(per_env_sgcs, list) and len(per_env_sgcs) == len(per_env_nmse):
                traces.append({
                    "x": list(range(len(per_env_sgcs))),
                    "y": per_env_sgcs,
                    "mode": "markers",
                    "name": "Per-map SGCS",
                    "yaxis": "y2",
                })
                sec_y = True
            html.append(_plotly(
                f"c{chart_idx}", traces,
                f"Per-Map Distribution — {tgt_name}",
                "Map index", "NMSE (dB)",
                secondary_y=sec_y,
            ))

    html.append("</div>")
    return chart_idx + 1


def _safe_float(v) -> float:
    try:
        f = float(v)
        return f if f == f else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _last_chart_idx(html: List[str]) -> int:
    """Return the largest chart div id seen so far in ``html`` (so the
    OOD charts don't collide with the in-distribution chart ids)."""
    import re
    idx = 0
    for line in html:
        for m in re.finditer(r"id='c(\d+)'", line):
            idx = max(idx, int(m.group(1)))
    return idx


def _opt(v, unit: str = "") -> str:
    """Format a value for the OOD table: float with 4dp, None → em-dash."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return f"{v} {unit}".strip()


def _css(icon_uri: str = "") -> str:
    icon_url = icon_uri or "data:image/svg+xml;utf8," + (
        "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
        "%3Crect width='64' height='64' rx='14' fill='%234f46e5'/%3E"
        "%3Ctext x='32' y='42' font-size='34' text-anchor='middle' "
        "font-family='Arial' fill='white' font-weight='bold'%3EC%3C/text%3E%3C/svg%3E"
    )
    return f"""
<style>
  :root {{
    --bg: #f4f5fb;
    --card: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #e2e8f0;
    --accent: #4f46e5;
    --accent-2: #7c3aed;
    --good: #059669;
    --bad: #dc2626;
    --good-bg: #ecfdf5;
    --bad-bg: #fef2f2;
    --shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.06);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background:
      radial-gradient(1100px 600px at 0% 0%, #eef2ff 0%, transparent 60%),
      radial-gradient(900px 500px at 100% 0%, #fdf2f8 0%, transparent 55%),
      var(--bg);
    color: var(--ink);
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}
  .container {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px 56px; }}

  /* Hero */
  .hero {{
    position: relative;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #ec4899 100%);
    color: #fff;
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 26px;
    box-shadow: 0 18px 40px rgba(79, 70, 229, 0.25);
    overflow: hidden;
  }}
  .hero::after {{
    content: ""; position: absolute; right: -90px; top: -90px;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 60%);
  }}
  .hero-row {{ display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }}
  .hero-icon {{
    width: 64px; height: 64px; border-radius: 14px; flex: 0 0 auto;
    background: rgba(255,255,255,0.18);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
  }}
  .hero-icon img {{ width: 100%; height: 100%; object-fit: cover; }}
  .hero h1 {{
    margin: 0; padding: 0; border: 0; color: #fff;
    font-size: 26px; font-weight: 700; letter-spacing: -0.01em;
  }}
  .hero .subtitle {{
    margin: 4px 0 0; opacity: 0.86; font-size: 13px;
  }}
  .chips {{ margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{
    background: rgba(255,255,255,0.16);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.22);
    padding: 5px 11px; border-radius: 999px;
    font-size: 12.5px; backdrop-filter: blur(6px);
  }}
  .chip b {{ font-weight: 600; opacity: 0.85; margin-right: 4px; }}

  /* Cards */
  .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 18px 22px 22px;
    margin-bottom: 22px;
    box-shadow: var(--shadow);
  }}
  .subcard {{
    background: #fbfbff;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 18px 18px;
    margin: 14px 0 18px;
  }}
  h2 {{
    margin: 0 0 14px 0; padding: 0 0 10px 0;
    border: 0; border-bottom: 1px solid var(--line);
    font-size: 18px; font-weight: 700; letter-spacing: -0.01em;
  }}
  h3 {{
    margin: 0 0 12px 0; padding: 0;
    font-size: 15px; font-weight: 600; color: #1e293b;
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }}
  .h3-tag {{
    background: #eef2ff; color: #4338ca;
    font-size: 11.5px; font-weight: 600;
    padding: 2px 8px; border-radius: 999px;
    letter-spacing: 0.01em;
  }}
  .callout {{
    background: linear-gradient(90deg, #eef2ff 0%, #fdf4ff 100%);
    border: 1px solid #e0e7ff;
    border-left: 3px solid var(--accent);
    color: #1e293b;
    border-radius: 8px;
    padding: 10px 14px; font-size: 13px;
    margin: 6px 0 14px;
  }}

  /* Tables */
  table {{
    width: 100%; border-collapse: separate; border-spacing: 0;
    margin: 4px 0 6px;
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 10px; overflow: hidden;
    font-size: 13.5px;
  }}
  thead th {{
    background: #f8fafc; color: #475569; font-weight: 600;
    text-align: left; padding: 9px 12px;
    border-bottom: 1px solid var(--line);
    font-size: 12.5px; letter-spacing: 0.02em; text-transform: uppercase;
  }}
  tbody td {{
    padding: 8px 12px; border-top: 1px solid #f1f5f9;
    vertical-align: middle;
  }}
  tbody tr:nth-child(even) td {{ background: #fafbff; }}
  tbody tr:hover td {{ background: #f5f7ff; }}
  code {{
    font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
    font-size: 12.5px;
    background: #f1f5f9; padding: 1px 6px; border-radius: 4px;
    color: #1e293b;
  }}
  .arrow-up {{ color: var(--good); font-weight: 700; }}
  .arrow-dn {{ color: var(--bad);  font-weight: 700; }}
  .tag {{
    display: inline-block; font-size: 11.5px; padding: 1px 8px; border-radius: 999px;
    background: #eef2ff; color: #4338ca; font-weight: 600; margin-left: 6px;
  }}
  .tag-good {{ background: var(--good-bg); color: var(--good); }}
  .tag-bad  {{ background: var(--bad-bg);  color: var(--bad);  }}

  .chart {{
    width: 100%; height: 420px; margin: 8px 0 4px;
    background: #fff; border: 1px solid var(--line);
    border-radius: 10px;
  }}
  .skipped {{ margin: 4px 0 0; padding-left: 18px; }}
  .skipped li {{ margin: 4px 0; }}

  a, a:visited {{ color: var(--accent); }}
  @media (max-width: 720px) {{
    .hero h1 {{ font-size: 22px; }}
    .card, .subcard {{ padding: 14px 14px 16px; }}
    thead th, tbody td {{ padding: 8px 9px; font-size: 12.5px; }}
  }}
</style>
"""


def _hero(meta: Dict[str, Any], icon_uri: str) -> str:
    """Render the gradient hero header with icon, title and meta chips."""
    task = meta.get("task", "feedback")
    model_name = meta.get("model_name") or meta.get("checkpoint") or (
        {"prediction": "CSI Prediction Model"}.get(task, "CSI Feedback Model"))
    title = f"CSI {{}} Evaluation Report".format(
        {"prediction": "Prediction", "feedback": "Feedback"}.get(task, "Feedback"))
    subtitle = {
        "prediction": "In-distribution, cross-mask, SNR and cross-scenario generalization assessment",
        "feedback": "In-distribution, robustness and cross-scenario generalization assessment",
    }.get(task, "CSI evaluation assessment")

    chips: List[str] = []
    chip_order = ("task", "device", "checkpoint", "timestamp", "model_name", "model_size_mb")
    for k in chip_order:
        v = meta.get(k)
        if v is None:
            continue
        label = {
            "task": "Task",
            "device": "Device",
            "checkpoint": "Checkpoint",
            "timestamp": "Timestamp",
            "model_name": "Model",
            "model_size_mb": "Size",
        }.get(k, k)
        if k == "model_size_mb":
            try:
                v = f"{float(v):.2f} MB"
            except (TypeError, ValueError):
                v = f"{v} MB"
        chips.append(f"<span class='chip'><b>{label}</b>{v}</span>")
    icon_tag = (
        f"<img src=\"{icon_uri}\" alt='icon'>" if icon_uri else ""
    )
    return (
        "<div class='container'>"
        "<div class='hero'>"
        "<div class='hero-row'>"
        f"<div class='hero-icon'>{icon_tag}</div>"
        "<div>"
        f"<h1>{title}</h1>"
        f"<div class='subtitle'>{subtitle} &middot; <b>{model_name}</b></div>"
        "</div>"
        "</div>"
        f"<div class='chips'>{''.join(chips)}</div>"
        "</div>"
    )


def _plotly(div_id: str, traces: List[Dict[str, Any]], title: str,
            x_title: str, y_title: str, secondary_y: bool = False,
            ref_lines: Optional[List[Dict[str, Any]]] = None) -> str:
    layout: Dict[str, Any] = {
        "title": title,
        "xaxis": {"title": x_title},
        "yaxis": {"title": y_title},
        "margin": {"l": 60, "r": 30, "t": 40, "b": 50},
    }
    if secondary_y:
        layout["yaxis2"] = {"title": "SGCS", "overlaying": "y", "side": "right"}
    if ref_lines:
        layout["shapes"] = [r for r in ref_lines if r.get("type") == "line"]
        layout["annotations"] = [
            {
                "xref": "paper", "yref": r.get("yref", "y1"),
                "x": 1, "xanchor": "right",
                "y": r.get("y0", 0), "yanchor": "bottom",
                "text": r.get("name", ""),
                "showarrow": False,
                "font": {"size": 10, "color": "#666"},
            }
            for r in ref_lines if r.get("type") == "line"
        ]
    return (
        f"<script>Plotly.newPlot('{div_id}', "
        f"{json.dumps(traces)}, {json.dumps(layout)}, "
        f"{{displayModeBar: false, responsive: true}});</script>"
    )


def _fmt(v) -> str:
    if v is None:
        return "<i>—</i>"
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

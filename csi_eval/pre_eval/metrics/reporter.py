"""
结果报告器
==========
统一输出 JSON / HTML / Markdown 三种格式。
"""
import os
import json
import base64
import datetime
from typing import Dict, List, Any
from pathlib import Path


class EvalReporter:
    """评估结果汇总与输出"""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or '.'
        self.timestamp = (datetime.datetime.now(datetime.timezone.utc)
                          + datetime.timedelta(hours=8)
                          ).strftime('%Y-%m-%d %H:%M:%S CST')
        self._results: Dict[str, Dict] = {}
        # model_name -> { fig_type: base64_str }
        self._fig_b64: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # 注册 / 更新结果
    # ------------------------------------------------------------------
    def add(self, model_name: str, metrics: Dict):
        self._results[model_name] = metrics

    def get(self, model_name: str) -> Dict:
        return self._results.get(model_name, {})

    def set_figures(self, model_name: str, fig_dir: str):
        """从 fig_dir 加载 4 张鲁棒性图片并 base64 编码，嵌入 HTML"""
        fig_types = [
            ('cross_ratio_nmse_curve.png', 'cross_ratio_nmse'),
            ('cross_ratio_sgcs_curve.png', 'cross_ratio_sgcs'),
            ('noise_nmse_curve.png', 'noise_nmse'),
            ('noise_sgcs_curve.png', 'noise_sgcs'),
        ]
        encoded = {}
        for fname, key in fig_types:
            fpath = os.path.join(fig_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    encoded[key] = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
        self._fig_b64[model_name] = encoded

    # ------------------------------------------------------------------
    # JSON 输出
    # ------------------------------------------------------------------
    def to_json(self, path: str = None) -> str:
        """写入 JSON 文件，返回路径"""
        if path is None:
            path = os.path.join(self.output_dir, 'eval_results.json')
        data = {
            'timestamp': self.timestamp,
            'output_dir': self.output_dir,
            'results': self._results,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False,
                      allow_nan='replace', default=str)
        return path

    # ------------------------------------------------------------------
    # Markdown 输出
    # ------------------------------------------------------------------
    def to_markdown(self, path: str = None) -> str:
        """写入 Markdown 报告，返回路径"""
        if path is None:
            path = os.path.join(self.output_dir, 'eval_results.md')

        lines = [
            '# CSI Pre-Evaluation Report',
            f'**生成时间**: {self.timestamp}',
            '',
            '## 目录',
            '',
        ]
        for name in self._results:
            lines.append(f'- [{name}](#{self._anchor(name)})')
        lines += ['', '---', '']
        for model_name, m in self._results.items():
            lines += self._model_section_md(model_name, m)

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return path

    def _anchor(self, name: str) -> str:
        return name.lower().replace(' ', '-')

    def _model_section_md(self, name: str, m: Dict) -> List[str]:
        lines = [f'## {name} <a id="{self._anchor(name)}"></a>', '']

        tm = m.get('task_metrics', {})
        if tm:
            lines += [
                '### 任务性能 (Joint)', '',
                f'| 指标 | 值 |', f'|------|----|',
                f"| NMSE (dB) | {tm.get('nmse_db', 'N/A')} |",
                f"| SGCS (Avg) | {tm.get('sgcs', 'N/A')} |",
                f"| SGCS Stream 1 | {tm.get('sgcs_stream_1', 'N/A')} |",
                f"| SGCS Stream 2 | {tm.get('sgcs_stream_2', 'N/A')} |",
                f"| SGCS Stream 3 | {tm.get('sgcs_stream_3', 'N/A')} |",
                f"| SGCS Stream 4 | {tm.get('sgcs_stream_4', 'N/A')} |",
                f"| 样本数 | {tm.get('n_samples', 'N/A')} |", '',
            ]

        sm = m.get('storage_metrics', {})
        if sm:
            lines += [
                '### 存储与部署', '',
                f"| 参数量 (M) | {sm.get('params_M', 'N/A')} |",
                f"| FP32 (MB) | {sm.get('fp32_MB', 'N/A')} |",
                f"| INT8 (MB) | {sm.get('int8_MB', 'N/A')} |", '',
            ]

        cm = m.get('compute_metrics', {})
        if cm:
            lines += [
                '### 计算效率', '',
                f"| Latency (ms) | {cm.get('latency_mean_ms', 'N/A')} ± {cm.get('latency_std_ms', 'N/A')} |",
                f"| FLOPs (G) | {cm.get('flops_G', 'N/A')} |",
                f"| Peak Memory (MB) | {cm.get('peak_memory_MB', 'N/A')} |", '',
            ]

        rob = m.get('robustness', {})
        if rob:
            cm_r = rob.get('cross_mask', {})
            if cm_r:
                lines += ['### Cross-mask Δ', '']
                for pair, vals in cm_r.items():
                    lines.append(
                        f"- **{pair}**: ΔNMSE = {vals.get('delta_nmse', 'N/A')} dB, "
                        f"ΔSGCS = {vals.get('delta_sgcs', 'N/A')}"
                    )
                lines.append('')

            crd = rob.get('cross_ratio_degradation', {})
            if crd:
                lines += ['### Cross Port-Ratio Δ', '']
                lines.append(f"| 退化对 | ΔNMSE (dB) | ΔSGCS |")
                lines.append(f"|------|------|------|")
                for pair, vals in crd.items():
                    lines.append(
                        f"| {pair} | {vals.get('delta_nmse', 'N/A')} | {vals.get('delta_sgcs', 'N/A')} |"
                    )
                lines.append('')

            cs = rob.get('cross_scenario', {})
            if cs:
                lines += [
                    '### 跨场景 (S1→S2)', '',
                    f"| 场景 | NMSE (dB) | SGCS Avg | S1 | S2 | S3 | S4 |",
                    f"|------|------|------|------|------|------|------|",
                    f"| S1 | {cs.get('s1_nmse_db', 'N/A')} | {cs.get('s1_sgcs', 'N/A')} | "
                    + " | ".join(str(x) for x in (list(cs.get('s1_sgcs_streams') or [])[:4] + ['N/A'] * 4)[:4]) + " |",
                    f"| S2 | {cs.get('s2_nmse_db', 'N/A')} | {cs.get('s2_sgcs', 'N/A')} | "
                    + " | ".join(str(x) for x in (list(cs.get('s2_sgcs_streams') or [])[:4] + ['N/A'] * 4)[:4]) + " |",
                    f"| Δ | {cs.get('delta_nmse', 'N/A')} | {cs.get('delta_sgcs_percent', 'N/A')}% | — | — | — | — |", '',
                ]

            noise = rob.get('noise_robustness', {})
            if noise:
                lines += ['### 噪声鲁棒性', '']
                lines.append(f"| SNR (dB) | NMSE (dB) | SGCS Avg | S1 | S2 | S3 | S4 |")
                lines.append(f"|------|------|------|------|------|------|------|")
                for snr in [5, 10, 15, 20, 25, 30, None]:
                    key = f'snr_{snr}' if snr is not None else 'snr_inf'
                    if key in noise:
                        nd = noise[key]
                        snr_str = '∞' if snr is None else str(snr)
                        streams = (list(nd.get('sgcs_streams') or [])[:4] + ['N/A'] * 4)[:4]
                        lines.append(
                            f"| {snr_str} | {nd.get('nmse_db', 'N/A')} | {nd.get('sgcs', 'N/A')} | "
                            + " | ".join(str(x) for x in streams) + " |"
                        )
                lines.append('')

        lines += ['---', '']
        return lines

    # ------------------------------------------------------------------
    # HTML 输出
    # ------------------------------------------------------------------
    def to_html(self, path: str = None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, 'eval_results.html')
        md_path = path.replace('.html', '.md')
        self.to_markdown(md_path)
        html = self._build_html()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        return path

    def _build_html(self) -> str:
        import html as html_lib
        def esc(s): return html_lib.escape(str(s), quote=True)

        model_cards = []
        for model_name, m in self._results.items():
            figs = self._fig_b64.get(model_name, {})
            model_cards.append(self._model_card_html(model_name, m, figs))

        summary_rows = self._build_summary_table()

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CSI Pre-Evaluation Report</title>
<style>
  :root {{
    --bg: #0a0e1a;
    --surface: #111827;
    --surface-2: #1a2035;
    --border: #1e2d45;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --accent: #818cf8;
    --accent-dim: rgba(129,140,248,0.12);
    --accent-glow: rgba(129,140,248,0.25);
    --green: #34d399;
    --red: #f87171;
    --blue: #38bdf8;
    --orange: #fb923c;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{ font-family: var(--font); background: var(--bg); color: var(--text); min-height: 100vh; line-height: 1.6; }}

  /* Header */
  .header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 3rem;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }}
  .header-inner {{ max-width: 1400px; margin: auto; }}
  .header-title {{ font-size: 1.5rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.6rem; }}
  .header-meta {{ margin-top: 0.4rem; font-size: 0.78rem; color: var(--text-muted); display: flex; gap: 1.2rem; }}
  .header-badge {{ display: inline-flex; align-items: center; background: var(--accent-dim); border: 1px solid var(--accent-glow); color: var(--accent); border-radius: 20px; padding: 0.1rem 0.6rem; font-size: 0.72rem; font-weight: 600; }}

  /* Body */
  .container {{ max-width: 1400px; margin: auto; padding: 2rem 3rem 4rem; }}
  .section {{ margin-bottom: 2.5rem; }}
  .section-label {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem; }}
  .section-label::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}

  /* Summary Table */
  .summary-table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--surface); border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }}
  .summary-table thead tr {{ background: var(--surface-2); }}
  .summary-table th {{ padding: 0.7rem 1.2rem; text-align: left; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); border-bottom: 1px solid var(--border); }}
  .summary-table td {{ padding: 0.75rem 1.2rem; font-size: 0.85rem; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  .summary-table tr:last-child td {{ border-bottom: none; }}
  .summary-table tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .model-name {{ font-weight: 700; color: #fff; }}
  .num {{ font-variant-numeric: tabular-nums; }}
  .c-blue {{ color: var(--blue); font-weight: 600; }}
  .c-green {{ color: var(--green); font-weight: 600; }}
  .c-orange {{ color: var(--orange); }}
  .c-red {{ color: var(--red); }}

  /* Model Cards Grid */
  .model-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr)); gap: 1.5rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; transition: border-color 0.2s, box-shadow 0.2s; }}
  .card:hover {{ border-color: var(--accent-glow); box-shadow: 0 0 30px var(--accent-dim); }}
  .card-header {{ background: var(--surface-2); padding: 0.9rem 1.3rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 0.8rem; }}
  .card-icon {{ width: 34px; height: 34px; border-radius: 8px; background: var(--accent-dim); border: 1px solid var(--accent-glow); display: flex; align-items: center; justify-content: center; font-size: 1rem; }}
  .card-title {{ font-size: 0.95rem; font-weight: 700; color: #fff; }}
  .card-tag {{ margin-left: auto; background: var(--accent-dim); color: var(--accent); border: 1px solid var(--accent-glow); border-radius: 6px; padding: 0.1rem 0.5rem; font-size: 0.65rem; font-weight: 600; }}
  .card-body {{ padding: 1.1rem 1.3rem; }}
  .subsection {{ margin-bottom: 1.1rem; }}
  .subsection:last-child {{ margin-bottom: 0; }}
  .sub-title {{ font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem; }}

  /* Metric Cards Grid */
  .metrics-g {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.5rem; }}
  .mcard {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 0.65rem 0.5rem; text-align: center; }}
  .mcard-lbl {{ font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }}
  .mcard-val {{ font-size: 1.1rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.2; }}
  .mcard-unit {{ font-size: 0.62rem; color: var(--text-muted); margin-top: 0.05rem; }}

  /* Mini Table */
  .mini-t {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  .mini-t th {{ text-align: left; padding: 0.28rem 0.5rem; color: var(--text-muted); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); }}
  .mini-t td {{ padding: 0.35rem 0.5rem; border-bottom: 1px solid rgba(30,45,69,0.5); vertical-align: middle; }}
  .mini-t tr:last-child td {{ border-bottom: none; }}

  /* Charts Grid */
  .charts-g {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem; margin-top: 0.8rem; }}
  .chart-wrap {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 0.7rem; text-align: center; }}
  .chart-title {{ font-size: 0.65rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }}
  .chart-wrap img {{ width: 100%; border-radius: 6px; }}
  .no-chart {{ color: var(--text-muted); font-size: 0.75rem; padding: 1.5rem; }}

  /* Cross-scenario bar */
  .sbar {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 0.4rem; align-items: center; margin-top: 0.5rem; }}
  .sbox {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 0.45rem 0.7rem; text-align: center; }}
  .sbox-lbl {{ font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .sbox-val {{ font-size: 0.95rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
  .sarrow {{ color: var(--text-muted); font-size: 1.1rem; text-align: center; }}
  .dpill {{ display: inline-block; border-radius: 20px; padding: 0.1rem 0.4rem; font-size: 0.68rem; font-weight: 600; margin-top: 0.4rem; }}
  .dpill.neg {{ background: rgba(248,113,113,0.15); color: var(--red); }}
  .dpill.pos {{ background: rgba(52,211,153,0.15); color: var(--green); }}

  /* Footer */
  .footer {{ text-align: center; color: var(--text-muted); font-size: 0.75rem; padding: 1.5rem 0; border-top: 1px solid var(--border); margin-top: 2rem; }}

  @media (max-width: 768px) {{
    .header {{ padding: 1.2rem 1.5rem; }}
    .container {{ padding: 1rem 1.5rem 2rem; }}
    .model-grid {{ grid-template-columns: 1fr; }}
    .charts-g {{ grid-template-columns: 1fr; }}
    .metrics-g {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <div class="header-title">
      <span>&#128246;</span> CSI Pre-Evaluation Report
      <span class="header-badge">&#9679; Auto-generated</span>
    </div>
    <div class="header-meta">
      <span>Models: {len(self._results)}</span>
    </div>
  </div>
</div>

<div class="container">

  <div class="section">
    <div class="section-label">&#9881; Model Comparison</div>
    <table class="summary-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>NMSE (dB)</th>
          <th>SGCS</th>
          <th>S1</th><th>S2</th><th>S3</th><th>S4</th>
          <th>Params</th>
          <th>Latency</th>
          <th>&#916;NMSE&#8594;grid</th>
          <th>&#916;SGCS&#8594;grid</th>
        </tr>
      </thead>
      <tbody>
        {summary_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-label">&#128202; Per-Model Details</div>
    <div class="model-grid">
      {''.join(model_cards)}
    </div>
  </div>

</div>

<div class="footer">Generated by CSI Pre-Evaluation Package</div>

<script id="results-data" type="application/json">{json.dumps(self._results, indent=2, ensure_ascii=False, allow_nan='replace', default=str)}</script>
</body>
</html>"""

    def _build_summary_table(self) -> str:
        rows = []
        for name, m in self._results.items():
            tm = m.get('task_metrics', {})
            sm = m.get('storage_metrics', {})
            cm = m.get('compute_metrics', {})
            rob = m.get('robustness', {})
            cm_delta = rob.get('cross_mask', {}).get('comb\u2192grid', {})
            nmse_v = tm.get('nmse_db')
            sgcs_v = tm.get('sgcs')
            sgcs_streams = list(tm.get('sgcs_streams') or [])[:4]
            if not sgcs_streams:
                sgcs_streams = [tm.get(f'sgcs_stream_{i}') for i in range(1, 5)]
            sgcs_streams += [None] * (4 - len(sgcs_streams))
            params_v = sm.get('params_M')
            lat_v = cm.get('latency_mean_ms')
            dn = cm_delta.get('delta_nmse')
            ds = cm_delta.get('delta_sgcs')

            def fmt(v, dec=4):
                if v is None: return '<span style="color:var(--text-muted)">—</span>'
                return f'{v:.{dec}f}'

            rows.append(f'<tr><td><span class="model-name">{self._esc(name)}</span></td>'
                        f'<td class="num c-blue">{fmt(nmse_v)}</td>'
                        f'<td class="num c-green">{fmt(sgcs_v)}</td>'
                        f'<td class="num c-green">{fmt(sgcs_streams[0])}</td>'
                        f'<td class="num c-green">{fmt(sgcs_streams[1])}</td>'
                        f'<td class="num c-green">{fmt(sgcs_streams[2])}</td>'
                        f'<td class="num c-green">{fmt(sgcs_streams[3])}</td>'
                        f'<td class="num c-orange">{fmt(params_v,1)} M</td>'
                        f'<td class="num">{fmt(lat_v,2)} ms</td>'
                        f'<td class="num c-red">{fmt(dn)}</td>'
                        f'<td class="num c-red">{fmt(ds,4)}</td></tr>')
        return ''.join(rows)

    def _model_card_html(self, name: str, m: Dict, figs: Dict[str, str]) -> str:
        def esc(s): return self._esc(str(s))
        def fmt(v, dec=4):
            if v is None: return '<span style="color:var(--text-muted)">—</span>'
            return f'{v:.{dec}f}'

        # Task Metrics
        tm = m.get('task_metrics', {})
        task_section = ''
        if tm:
            task_section = f"""<div class="subsection">
  <div class="sub-title">&#128200; Task Performance</div>
  <div class="metrics-g">
    <div class="mcard">
      <div class="mcard-lbl">NMSE (dB)</div>
      <div class="mcard-val" style="color:var(--blue)">{fmt(tm.get('nmse_db'))}</div>
      <div class="mcard-unit">dB</div>
    </div>
    <div class="mcard">
      <div class="mcard-lbl">SGCS</div>
      <div class="mcard-val" style="color:var(--green)">{fmt(tm.get('sgcs'))}</div>
      <div class="mcard-unit">scale</div>
    </div>
    <div class="mcard">
      <div class="mcard-lbl">SGCS S1/S2/S3/S4</div>
      <div class="mcard-val" style="color:var(--green);font-size:0.95rem">{', '.join(fmt(v) for v in ((list(tm.get('sgcs_streams') or [])[:4] + [None] * 4)[:4]))}</div>
      <div class="mcard-unit">eigenvalue-descending streams</div>
    </div>
    <div class="mcard">
      <div class="mcard-lbl">Samples</div>
      <div class="mcard-val" style="color:var(--orange)">{fmt(tm.get('n_samples'), 0)}</div>
      <div class="mcard-unit">N</div>
    </div>
  </div>
</div>"""

        # Storage
        sm = m.get('storage_metrics', {})
        storage_rows = ''
        if sm:
            for lbl, key in [('FP32','fp32_MB'),('FP16','fp16_MB'),('INT8','int8_MB'),('Params','params_M')]:
                v = sm.get(key)
                storage_rows += f'<tr><td>{lbl}</td><td class="num c-blue">{fmt(v,2)} MB</td></tr>' if key != 'params_M' else f'<tr><td>{lbl}</td><td class="num c-blue">{fmt(v,1)} M</td></tr>'
        storage_section = f"""<div class="subsection">
  <div class="sub-title">&#128190; Storage</div>
  <table class="mini-t"><tr><th>Format</th><th>Size</th></tr>{storage_rows}</table>
</div>""" if storage_rows else ''

        # Compute
        cm = m.get('compute_metrics', {})
        perf_rows = ''
        if cm:
            perf_rows = (f'<tr><td>Latency</td><td class="num">{fmt(cm.get("latency_mean_ms"),2)} \u00b1 {fmt(cm.get("latency_std_ms"),2)} ms</td></tr>'
                         f'<tr><td>Throughput</td><td class="num">{fmt(cm.get("throughput_fps"),0)} fps</td></tr>'
                         f'<tr><td>FLOPs</td><td class="num">{fmt(cm.get("flops_G"))} G</td></tr>'
                         f'<tr><td>Peak Mem</td><td class="num">{fmt(cm.get("peak_memory_MB"),1)} MB</td></tr>')
        compute_section = f"""<div class="subsection">
  <div class="sub-title">&#9889; Compute</div>
  <table class="mini-t"><tr><th>Metric</th><th>Value</th></tr>{perf_rows}</table>
</div>""" if perf_rows else ''

        # Cross-mask
        rob = m.get('robustness', {})
        cm_r = rob.get('cross_mask', {})
        cross_mask_rows = ''
        if cm_r:
            for pair, vals in cm_r.items():
                dn = vals.get('delta_nmse', 0)
                ds = vals.get('delta_sgcs', 0)
                cross_mask_rows += f'<tr><td>{esc(pair)}</td><td class="num {"c-red" if dn > 0 else "c-green"}">{fmt(dn)} dB</td><td class="num {"c-red" if ds > 0 else "c-green"}">{fmt(ds)}</td></tr>'
        cross_mask_section = f"""<div class="subsection">
  <div class="sub-title">&#128260; Cross-Mask &#916;</div>
  <table class="mini-t"><tr><th>Pair</th><th>&#916;NMSE</th><th>&#916;SGCS</th></tr>{cross_mask_rows}</table>
</div>""" if cross_mask_rows else ''

        # Cross-scenario
        cs = rob.get('cross_scenario', {})
        cs_section = ''
        if cs:
            def inf(v, dec=4):
                if v is None: return '—'
                if v in (float('inf'), float('-inf')): return '\u221e'
                return f'{v:.{dec}f}'
            s1n, s2n = cs.get('s1_nmse_db'), cs.get('s2_nmse_db')
            dn, dp = cs.get('delta_nmse'), cs.get('delta_sgcs_percent')
            s2_ok = s2n is not None and s2n != float('-inf')
            s2n_str = inf(s2n)
            cs_section = f"""<div class="subsection">
  <div class="sub-title">&#127757; Cross-Scenario (S1&#8594;S2)</div>
  <div class="sbar">
    <div class="sbox">
      <div class="sbox-lbl">S1 NMSE</div>
      <div class="sbox-val" style="color:var(--blue)">{inf(s1n)}</div>
    </div>
    <div class="sarrow">&#8594;</div>
    <div class="sbox">
      <div class="sbox-lbl">S2 NMSE</div>
      <div class="sbox-val" style="color:{'var(--blue)' if s2_ok else 'var(--red)'}">{s2n_str}</div>
    </div>
  </div>
  <div style="text-align:center;">
    <span class="dpill neg">&#916;NMSE = {inf(dn)} dB</span>
    <span class="dpill neg">&#916;SGCS = {inf(dp)}%</span>
  </div>
</div>"""



        # Charts
        chart_meta = [
            ('cross_ratio_nmse', '&#128200; Port-Ratio vs NMSE'),
            ('cross_ratio_sgcs', '&#128200; Port-Ratio vs SGCS'),
            ('noise_nmse', '&#128266; SNR vs NMSE'),
            ('noise_sgcs', '&#128266; SNR vs SGCS'),
        ]
        chart_items = []
        for key, title in chart_meta:
            b64 = figs.get(key)
            if b64:
                chart_items.append(f"""<div class="chart-wrap">
  <div class="chart-title">{title}</div>
  <img src="{esc(b64)}" alt="{esc(title)}">
</div>""")
            else:
                chart_items.append(f"""<div class="chart-wrap">
  <div class="chart-title">{title}</div>
  <div class="no-chart">No data</div>
</div>""")
        charts_section = f"""<div class="subsection">
  <div class="sub-title">&#128444; Robustness Curves</div>
  <div class="charts-g">{''.join(chart_items)}</div>
</div>"""

        # Assemble
        return f"""<div class="card">
  <div class="card-header">
    <div class="card-icon">&#129302;</div>
    <div class="card-title">{esc(name)}</div>
    <span class="card-tag">Joint</span>
  </div>
  <div class="card-body">
    {task_section}
    {storage_section}
    {compute_section}
    {cross_mask_section}
    {cs_section}
    {charts_section}
  </div>
</div>"""

    def _esc(self, s):
        import html
        return html.escape(str(s), quote=True)

    # ------------------------------------------------------------------
    # 综合导出
    # ------------------------------------------------------------------
    def save_all(self, output_dir: str = None) -> Dict[str, str]:
        """同时导出 JSON + Markdown + HTML"""
        if output_dir:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        paths = {
            'json': self.to_json(),
            'markdown': self.to_markdown(),
            'html': self.to_html(),
        }
        return paths
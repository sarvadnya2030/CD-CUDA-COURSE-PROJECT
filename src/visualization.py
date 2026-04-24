"""
visualization.py — Plotly interactive dashboard for CUDA auto-tuner results.

Generates a self-contained HTML file with:
  1. Roofline scatter  — arithmetic intensity vs GFLOP/s, roofline boundary
  2. Convergence curves — latency over search iterations per kernel
  3. Speedup bars       — best-variant speedup vs baseline per kernel
  4. Occupancy heatmap  — block_size × register_count → occupancy %
  5. Timing violin      — per-sample distribution for best variant

Usage:
    python -m src.visualization --output=results/dashboard.html
    python -m src.visualization --kernel=matmul --output=results/matmul.html
    python -m src.visualization --ascii          (terminal summary only)
"""

import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# RTX 2070 hardware limits (sm_75)
PEAK_GFLOPS  = 7500.0   # FP32 GFLOP/s
PEAK_BW_GBS  = 448.0    # GB/s
RIDGE_POINT  = PEAK_GFLOPS / PEAK_BW_GBS   # ~16.74 FLOP/byte

SUPPORTED_KERNELS = ["matmul", "softmax", "reduction", "layernorm", "attention"]

# GitHub dark-palette colours for each kernel
_PALETTE = {
    "matmul":    "#58a6ff",
    "softmax":   "#7ee787",
    "reduction": "#d29922",
    "layernorm": "#f78166",
    "attention": "#bc8cff",
}
_DEFAULT_COLOR = "#8b949e"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_tuning(kernels: List[str]) -> Dict[str, Any]:
    out = {}
    for k in kernels:
        p = RESULTS_DIR / f"{k}_tuning.json"
        if p.exists():
            with open(p) as f:
                out[k] = json.load(f)
    return out


def _load_convergence(kernel: str) -> List[float]:
    p = RESULTS_DIR / f"{kernel}_convergence.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    # Convergence JSON stores list of {params, ms} dicts or just ms values
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return [d.get("ms", d.get("mean_ms", 0.0)) for d in data]
        return [float(v) for v in data]
    return []


# ── Chart builders ────────────────────────────────────────────────────────────

def _roofline_chart(results: Dict[str, Any]):
    """Scatter on roofline plot — one marker per kernel best variant."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    # Roofline boundary
    ai_range = [0.1, 1000.0]
    roof_y = [min(ai * PEAK_BW_GBS, PEAK_GFLOPS) for ai in
              [10**x for x in [math.log10(ai_range[0]) + i * 0.05
                                for i in range(int((math.log10(ai_range[1]) -
                                                    math.log10(ai_range[0])) / 0.05) + 1)]]]
    roof_x = [10**x for x in [math.log10(ai_range[0]) + i * 0.05
                               for i in range(len(roof_y))]]

    fig = go.Figure()

    # Roofline boundary line
    fig.add_trace(go.Scatter(
        x=roof_x, y=roof_y,
        mode="lines",
        name="Roofline boundary",
        line=dict(color="#30363d", width=2, dash="dot"),
        hoverinfo="skip",
    ))

    # Memory-bound and compute-bound region labels
    fig.add_annotation(x=math.log10(0.3), y=math.log10(500),
                       text="Memory bound", showarrow=False,
                       font=dict(color="#8b949e", size=11))
    fig.add_annotation(x=math.log10(100), y=math.log10(3000),
                       text="Compute bound", showarrow=False,
                       font=dict(color="#8b949e", size=11))

    # Ridge point vertical line
    fig.add_shape(type="line",
                  x0=math.log10(RIDGE_POINT), x1=math.log10(RIDGE_POINT),
                  y0=0, y1=math.log10(PEAK_GFLOPS * 1.5),
                  line=dict(color="#444c56", width=1, dash="dash"))

    # One scatter marker per kernel
    for kernel, data in results.items():
        best = data.get("best", {})
        ai   = best.get("arithmetic_intensity", None)
        gf   = best.get("achieved_gflops",      None)
        if ai is None or gf is None or ai <= 0 or gf <= 0:
            continue
        color = _PALETTE.get(kernel, _DEFAULT_COLOR)
        fig.add_trace(go.Scatter(
            x=[ai], y=[gf],
            mode="markers+text",
            name=kernel,
            marker=dict(size=14, color=color,
                        line=dict(width=2, color="white")),
            text=[kernel],
            textposition="top center",
            textfont=dict(color=color, size=11),
            hovertemplate=(
                f"<b>{kernel}</b><br>"
                "AI: %{x:.2f} FLOP/byte<br>"
                "GFLOP/s: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="Roofline Model — RTX 2070 (sm_75)", font=dict(size=16)),
        xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log",
                   gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(title="Achieved GFLOP/s", type="log",
                   range=[math.log10(0.1), math.log10(PEAK_GFLOPS * 2)],
                   gridcolor="#21262d", linecolor="#30363d"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
        **_dark_layout(),
    )
    return fig


def _convergence_chart(results: Dict[str, Any]):
    """Line chart: latency over search iterations per kernel."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    fig = go.Figure()
    has_data = False

    for kernel in results:
        curve = _load_convergence(kernel)
        if not curve:
            # Fall back to variants list if convergence JSON absent
            variants = results[kernel].get("variants", [])
            curve = [v.get("mean_ms", 0.0) for v in variants if "mean_ms" in v]
        if not curve:
            continue
        has_data = True
        # Running best
        best_so_far = []
        curr_best = float("inf")
        for v in curve:
            curr_best = min(curr_best, v)
            best_so_far.append(curr_best)

        color = _PALETTE.get(kernel, _DEFAULT_COLOR)
        fig.add_trace(go.Scatter(
            x=list(range(len(curve))), y=curve,
            mode="lines",
            name=f"{kernel} (all)",
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.4,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(best_so_far))), y=best_so_far,
            mode="lines",
            name=f"{kernel} (best)",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{kernel}</b> — iter %{{x}}<br>"
                "Best latency: %{y:.3f} ms<extra></extra>"
            ),
        ))

    if not has_data:
        return None

    fig.update_layout(
        title=dict(text="Search Convergence — Latency over Iterations",
                   font=dict(size=16)),
        xaxis=dict(title="Iteration", gridcolor="#21262d"),
        yaxis=dict(title="Latency (ms)", gridcolor="#21262d"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
        **_dark_layout(),
    )
    return fig


def _speedup_bars(results: Dict[str, Any]):
    """Grouped bar chart: baseline vs best-variant latency, annotated speedup."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    kernels   = [k for k in results if results[k].get("best")]
    baselines = [results[k].get("baseline_ms", 0.0) for k in kernels]
    bests     = [results[k]["best"].get("mean_ms", 0.0) for k in kernels]
    speedups  = [b / t if t > 0 else 0 for b, t in zip(baselines, bests)]
    colors    = [_PALETTE.get(k, _DEFAULT_COLOR) for k in kernels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline",
        x=kernels, y=baselines,
        marker_color="#444c56",
        text=[f"{v:.2f} ms" for v in baselines],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Best variant",
        x=kernels, y=bests,
        marker_color=colors,
        text=[f"{v:.2f} ms" for v in bests],
        textposition="outside",
    ))

    # Speedup annotations
    for i, (k, sp) in enumerate(zip(kernels, speedups)):
        if sp > 0:
            fig.add_annotation(
                x=k, y=max(baselines[i], bests[i]) * 1.15,
                text=f"{sp:.2f}x",
                showarrow=False,
                font=dict(color="#7ee787" if sp >= 1.0 else "#f85149", size=12),
            )

    fig.update_layout(
        title=dict(text="Speedup: Baseline vs Best Variant", font=dict(size=16)),
        barmode="group",
        xaxis=dict(title="Kernel", gridcolor="#21262d"),
        yaxis=dict(title="Latency (ms)", gridcolor="#21262d"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
        **_dark_layout(),
    )
    return fig


def _occupancy_heatmap(results: Dict[str, Any]):
    """
    Heatmap of occupancy % across block_size × register count.
    Reads from variants that have occupancy and registers fields.
    Falls back to a synthetic grid from sm_75 limits if no real data.
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        return None

    # Collect (block_size, registers, occupancy) triples from all kernels
    points = []
    for kernel, data in results.items():
        for v in data.get("variants", []):
            bs  = v.get("block_size") or v.get("params", {}).get("block_size")
            reg = v.get("registers_per_thread")
            occ = v.get("occupancy")
            if bs and reg and occ is not None:
                points.append((int(bs), int(reg), float(occ) * 100))

    if not points:
        # Synthetic sm_75 grid: compute_occupancy for a range of (regs, smem=0)
        block_sizes = [64, 128, 256, 512, 1024]
        reg_counts  = list(range(8, 65, 8))
        MAX_REGS_PER_SM  = 65536
        MAX_WARPS_PER_SM = 32
        z = []
        for reg in reg_counts:
            row = []
            for bs in block_sizes:
                warps = bs // 32
                if reg == 0:
                    row.append(100.0)
                    continue
                warps_by_regs = MAX_REGS_PER_SM // (reg * 32)
                active_warps  = min(warps_by_regs // warps * warps, MAX_WARPS_PER_SM)
                occ = min(active_warps / MAX_WARPS_PER_SM * 100, 100.0)
                row.append(occ)
            z.append(row)
        x_labels = [str(b) for b in block_sizes]
        y_labels = [str(r) for r in reg_counts]
    else:
        block_sizes = sorted(set(p[0] for p in points))
        reg_counts  = sorted(set(p[1] for p in points))
        lookup = {(p[0], p[1]): p[2] for p in points}
        z = [[lookup.get((bs, reg), float("nan"))
              for bs in block_sizes] for reg in reg_counts]
        x_labels = [str(b) for b in block_sizes]
        y_labels = [str(r) for r in reg_counts]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0,  "#1a1f2e"],
            [0.25, "#1f3a5f"],
            [0.5,  "#1f6b5f"],
            [0.75, "#2ea04320"],
            [1.0,  "#7ee787"],
        ],
        zmin=0, zmax=100,
        colorbar=dict(title="Occupancy %", ticksuffix="%",
                      tickfont=dict(color="#c9d1d9")),
        hovertemplate=(
            "Block size: %{x}<br>"
            "Registers: %{y}<br>"
            "Occupancy: %{z:.1f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text="Theoretical Occupancy — sm_75 (RTX 2070)",
                   font=dict(size=16)),
        xaxis=dict(title="Block Size (threads)"),
        yaxis=dict(title="Registers per Thread"),
        **_dark_layout(),
    )
    return fig


def _timing_violin(results: Dict[str, Any]):
    """Box/violin plots of raw per-sample timing distributions."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    fig = go.Figure()
    has_data = False

    for kernel, data in results.items():
        best = data.get("best", {})
        samples = best.get("raw_samples") or best.get("samples")
        if not samples:
            continue
        has_data = True
        color = _PALETTE.get(kernel, _DEFAULT_COLOR)
        fig.add_trace(go.Violin(
            y=samples,
            name=kernel,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.6,
            line_color=color,
            hovertemplate=(
                f"<b>{kernel}</b><br>"
                "Latency: %{y:.3f} ms<extra></extra>"
            ),
        ))

    if not has_data:
        return None

    fig.update_layout(
        title=dict(text="Timing Distribution — Best Variant (30 samples)",
                   font=dict(size=16)),
        yaxis=dict(title="Latency (ms)", gridcolor="#21262d"),
        xaxis=dict(title="Kernel"),
        showlegend=False,
        **_dark_layout(),
    )
    return fig


# ── Dark theme helper ─────────────────────────────────────────────────────────

def _dark_layout() -> dict:
    return dict(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="monospace"),
        margin=dict(l=60, r=40, t=60, b=50),
        height=420,
    )


# ── HTML assembly ─────────────────────────────────────────────────────────────

def generate_html_dashboard(
    kernel_results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> str:
    """
    Build a self-contained HTML file with all five Plotly charts.
    Returns the HTML string; also writes to output_path if given.
    """
    try:
        import plotly.io as pio
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    charts = {
        "roofline":    _roofline_chart(kernel_results),
        "convergence": _convergence_chart(kernel_results),
        "speedup":     _speedup_bars(kernel_results),
        "occupancy":   _occupancy_heatmap(kernel_results),
        "violin":      _timing_violin(kernel_results),
    }

    # Serialise each chart to an HTML div (no full page, no external CDN call
    # per chart — we'll embed one copy of plotly.js at the top)
    divs = {}
    plotlyjs_injected = False
    plotlyjs_tag = ""

    for name, fig in charts.items():
        if fig is None:
            divs[name] = f'<p style="color:#8b949e;padding:20px">No data for {name}</p>'
            continue
        include_js = not plotlyjs_injected
        html_chunk = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn" if include_js else False,
            div_id=f"chart_{name}",
        )
        if include_js:
            plotlyjs_injected = True
        divs[name] = html_chunk

    n_kernels = len(kernel_results)
    best_speedup = max(
        (d.get("best", {}).get("speedup", 0.0) for d in kernel_results.values()),
        default=0.0,
    )
    kernels_str = ", ".join(kernel_results.keys()) or "—"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CUDA Auto-Tuner Dashboard</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      font-family: 'Segoe UI', system-ui, monospace;
      background: #0d1117; color: #c9d1d9; padding: 24px;
    }}
    h1 {{ color: #58a6ff; font-size: 1.8em; margin-bottom: 6px; }}
    .sub {{ color: #8b949e; margin-bottom: 28px; font-size: 0.95em; }}
    .stats-row {{
      display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px;
    }}
    .stat-card {{
      flex: 1; min-width: 140px; background: #161b22;
      border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px;
    }}
    .stat-card .val  {{ font-size: 1.6em; font-weight: bold; color: #7ee787; }}
    .stat-card .lbl  {{ font-size: 0.8em; color: #8b949e; margin-top: 4px; }}
    .section {{
      background: #161b22; border: 1px solid #30363d;
      border-radius: 10px; padding: 20px; margin-bottom: 20px;
    }}
    .section h2 {{ color: #58a6ff; font-size: 1.1em; margin-bottom: 14px; }}
    .grid-2 {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
    }}
    @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>CUDA Auto-Tuner Dashboard</h1>
  <p class="sub">RTX 2070 (sm_75, Turing) &mdash; {kernels_str}</p>

  <div class="stats-row">
    <div class="stat-card">
      <div class="val">{n_kernels}</div>
      <div class="lbl">Kernels tuned</div>
    </div>
    <div class="stat-card">
      <div class="val">{best_speedup:.2f}x</div>
      <div class="lbl">Best speedup</div>
    </div>
    <div class="stat-card">
      <div class="val">{PEAK_GFLOPS/1000:.1f} TFLOP/s</div>
      <div class="lbl">Peak compute (FP32)</div>
    </div>
    <div class="stat-card">
      <div class="val">{PEAK_BW_GBS:.0f} GB/s</div>
      <div class="lbl">Peak memory BW</div>
    </div>
    <div class="stat-card">
      <div class="val">{RIDGE_POINT:.1f}</div>
      <div class="lbl">Ridge point (FLOP/byte)</div>
    </div>
  </div>

  <div class="section">
    <h2>Roofline Model</h2>
    {divs["roofline"]}
  </div>

  <div class="grid-2">
    <div class="section">
      <h2>Search Convergence</h2>
      {divs["convergence"]}
    </div>
    <div class="section">
      <h2>Speedup vs Baseline</h2>
      {divs["speedup"]}
    </div>
  </div>

  <div class="grid-2">
    <div class="section">
      <h2>Occupancy Heatmap</h2>
      {divs["occupancy"]}
    </div>
    <div class="section">
      <h2>Timing Distributions</h2>
      {divs["violin"]}
    </div>
  </div>
</body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")
        print(f"Dashboard written → {output_path}")

    return html


# ── ASCII fallback (terminal) ─────────────────────────────────────────────────

def _ascii_bar(label: str, val: float, max_val: float, width: int = 40) -> str:
    n = int((val / max_val) * width) if max_val > 0 else 0
    return f"  {label:<20} {'█'*n}{'░'*(width-n)} {val:.3f} ms"


def print_terminal_dashboard(results: Dict[str, Any]):
    print("\n" + "=" * 72)
    print("  CUDA AUTO-TUNER — TERMINAL DASHBOARD")
    print("=" * 72)
    print(f"  Hardware: RTX 2070 (sm_75)  |  "
          f"Peak: {PEAK_GFLOPS/1000:.1f} TFLOP/s  |  "
          f"BW: {PEAK_BW_GBS:.0f} GB/s")
    print()

    for kernel, data in results.items():
        best     = data.get("best", {})
        baseline = data.get("baseline_ms", 0.0)
        best_ms  = best.get("mean_ms", 0.0)
        speedup  = best.get("speedup", baseline / best_ms if best_ms else 0)
        ai       = best.get("arithmetic_intensity", 0.0)
        gflops   = best.get("achieved_gflops", 0.0)
        bound    = best.get("bound_type", "?")
        occ      = best.get("occupancy", 0.0)
        n_tested = data.get("n_variants", "?")

        print(f"  ── {kernel.upper()} ─────────────────────────────")
        max_ms = max(baseline, best_ms) if best_ms else baseline
        if baseline > 0:
            print(_ascii_bar("baseline", baseline, max_ms))
        if best_ms > 0:
            print(_ascii_bar("best variant", best_ms, max_ms))
        print(f"  {'Speedup':<20} {speedup:.2f}x  |  "
              f"AI={ai:.2f} FLOP/byte  |  "
              f"bound={bound}  |  "
              f"occ={occ*100:.0f}%  |  "
              f"GFLOP/s={gflops:.1f}  |  "
              f"variants={n_tested}")
        print()

    print("=" * 72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CUDA auto-tuner visualization dashboard")
    parser.add_argument("--kernel", default=None,
                        help="Kernel(s) to include (comma-separated, default: all)")
    parser.add_argument("--output", default=None,
                        help="Output HTML file path (e.g. results/dashboard.html)")
    parser.add_argument("--ascii", action="store_true",
                        help="Print ASCII summary to terminal")
    args = parser.parse_args()

    kernels = SUPPORTED_KERNELS
    if args.kernel:
        kernels = [k.strip() for k in args.kernel.split(",")]

    results = _load_tuning(kernels)

    if not results:
        print("No tuning results found. Run: python autotune.py --kernel=<name>")
        return

    if args.ascii or not args.output:
        print_terminal_dashboard(results)

    if args.output:
        generate_html_dashboard(results, output_path=Path(args.output))


if __name__ == "__main__":
    main()

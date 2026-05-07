"""
streamlit_dashboard.py — Premium Live Dashboard for CUDA Auto-Tuner

Usage:
    streamlit run streamlit_dashboard.py

Features:
    - Premium dark glassmorphism UI
    - Live tuning panel with real-time stdout streaming
    - GPU stats monitoring (util, mem, temp, power)
    - Per-kernel analytics: speedup, latency, GFLOP/s, occupancy
    - Convergence + speedup + roofline charts
    - Top-5 config leaderboard per kernel
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Compiler Design Components (Unit I-VI)
try:
    from cdc.frontend import run_frontend
    from cdc.ir import emit_tac, partition_blocks, build_cfg, build_dag
    from cdc.ir.basic_block import format_blocks
    from cdc.ir.dag import format_dag
    from cdc.opt.dfa import LiveVariables, ReachingDefsSolver, AvailableExpressions, solve, format_dfa_result, collect_universe
    from cdc.opt import constant_propagation, common_subexpression_elimination, dead_code_elimination, loop_invariant_code_motion, strength_reduction
    from cdc.opt.register_pressure import estimate_register_pressure, format_report as format_rp_report
    from cdc.first_follow import build_example_grammar, compute_nullable, compute_first, compute_follow, build_ll1_table, format_first_follow, format_ll1_table
    CDC_AVAILABLE = True
except ImportError as e:
    CDC_AVAILABLE = False
    CDC_ERROR = str(e)

# ── Config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CUDA Auto-Tuner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
SRC_DIR     = ROOT / "src"
LOG_FILE    = ROOT / ".tuning_log.txt"
PID_FILE    = ROOT / ".tuning_pid"
CONFIG_FILE = ROOT / ".run_config.json"


# ── CSS ────────────────────────────────────────────────────────────────────

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
  * { font-family: 'Inter', sans-serif !important; }
  code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

  .stApp { background: #060b14; }
  .main .block-container { padding-top: 1rem; max-width: 100%; }

  /* Hide default streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── glass card ── */
  .glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(59,130,246,0.15);
  }

  /* ── hero metric card ── */
  .metric-hero {
    background: linear-gradient(135deg, rgba(17,24,39,0.9) 0%, rgba(10,14,23,0.95) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 22px 20px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
  }
  .metric-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  }
  .metric-hero:hover { transform: translateY(-3px); }
  .metric-hero .val {
    font-size: 2.4em;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -1px;
    font-family: 'JetBrains Mono', monospace !important;
  }
  .metric-hero .lbl {
    color: #64748b;
    font-size: 0.78em;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }
  .metric-hero .delta-up {
    display: inline-block;
    margin-top: 8px;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(16,185,129,0.12);
    color: #10b981;
    font-size: 0.78em;
    font-weight: 600;
  }
  .metric-hero .delta-down {
    display: inline-block;
    margin-top: 8px;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(239,68,68,0.12);
    color: #ef4444;
    font-size: 0.78em;
    font-weight: 600;
  }
  .metric-hero .delta-neutral {
    display: inline-block;
    margin-top: 8px;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(148,163,184,0.1);
    color: #94a3b8;
    font-size: 0.78em;
    font-weight: 600;
  }

  /* ── GPU stats strip ── */
  .gpu-strip {
    background: rgba(17,24,39,0.8);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 12px 20px;
    display: flex;
    justify-content: space-around;
    align-items: center;
    margin-bottom: 8px;
  }
  .gpu-stat { text-align: center; }
  .gpu-stat-val {
    font-size: 1.35em;
    font-weight: 700;
    color: #3b82f6;
    font-family: 'JetBrains Mono', monospace !important;
  }
  .gpu-stat-lbl {
    font-size: 0.72em;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #10b981;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
  }
  .live-dot.dead { background: #ef4444; animation: none; }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.6); }
    50%      { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
  }

  /* ── page title ── */
  .page-title {
    text-align: center;
    padding: 16px 0 8px;
  }
  .page-title h1 {
    font-size: 2.8em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
  }
  .page-title p {
    color: #64748b;
    font-size: 1em;
    margin-top: 6px;
  }
  .hw-badge {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    color: #3b82f6;
    padding: 5px 18px;
    border-radius: 50px;
    font-size: 0.82em;
    font-family: 'JetBrains Mono', monospace !important;
    display: inline-block;
    margin-top: 10px;
  }

  /* ── live tuning panel ── */
  .live-panel {
    background: rgba(10,14,23,0.95);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0;
  }
  .live-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    font-size: 1.05em;
    font-weight: 700;
    color: #f1f5f9;
  }
  .log-box {
    background: #020409;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78em;
    color: #94a3b8;
    max-height: 260px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.6;
  }
  .variant-stat-row {
    display: flex;
    gap: 12px;
    margin-top: 12px;
    flex-wrap: wrap;
  }
  .variant-stat {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 10px;
    padding: 8px 16px;
    text-align: center;
    flex: 1;
    min-width: 100px;
  }
  .variant-stat .vs-val {
    font-size: 1.3em;
    font-weight: 700;
    color: #3b82f6;
    font-family: 'JetBrains Mono', monospace !important;
  }
  .variant-stat .vs-lbl {
    font-size: 0.7em;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
  }

  /* ── table ── */
  .stDataFrame { border-radius: 12px; overflow: hidden; }
  thead th { background: #111827 !important; color: #94a3b8 !important; }
  tbody tr:hover td { background: rgba(59,130,246,0.06) !important; }

  /* ── misc ── */
  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: #64748b;
    padding: 10px 20px;
  }
  .stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
  }
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
  }
  div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
  }
  .section-head {
    font-size: 1em;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 20px 0 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
  }
  .no-data-card {
    background: rgba(17,24,39,0.6);
    border: 1px dashed rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    color: #475569;
  }
  .no-data-card .icon { font-size: 2.5em; margin-bottom: 10px; }
  .no-data-card code {
    background: rgba(59,130,246,0.1);
    color: #3b82f6;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.85em;
  }
  
  /* rank badges */
  .rank-gold   { color: #fbbf24; font-weight: 700; font-size: 1.1em; }
  .rank-silver { color: #94a3b8; font-weight: 700; }
  .rank-bronze { color: #d97706; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Chart theme ─────────────────────────────────────────────────────────────

_AXIS_STYLE = dict(gridcolor="rgba(255,255,255,0.04)", showline=False, zeroline=False)


def chart_layout(**extra) -> dict:
    """Base Plotly layout dict merged with any extra keys (no xaxis/yaxis collision)."""
    base = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b", family="Inter"),
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=_AXIS_STYLE.copy(),
        yaxis=_AXIS_STYLE.copy(),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    )
    # Deep-merge xaxis / yaxis overrides instead of replacing
    for k in ("xaxis", "yaxis"):
        if k in extra:
            base[k] = {**base[k], **extra.pop(k)}
    base.update(extra)
    return base

# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=2)
def get_gpu_stats() -> dict | None:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,"
             "temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            vals = [v.strip() for v in r.stdout.strip().split(",")]
            return {
                "gpu_util":  float(vals[0]),
                "mem_util":  float(vals[1]),
                "mem_used":  float(vals[2]),
                "mem_total": float(vals[3]),
                "temp":      float(vals[4]),
                "power":     float(vals[5]) if len(vals) > 5 else 0,
                "sm_clock":  int(vals[6]) if len(vals) > 6 else 0,
            }
    except Exception:
        pass
    return None


def _parse_ptx_registers(ptx_info: str) -> int | None:
    """Extract register count from ptxas -v stderr."""
    import re
    m = re.search(r"Used (\d+) registers", ptx_info or "")
    return int(m.group(1)) if m else None


def _parse_ptx_smem(ptx_info: str) -> int | None:
    """Extract shared memory bytes from ptxas -v stderr."""
    import re
    m = re.search(r"(\d+) bytes smem", ptx_info or "")
    return int(m.group(1)) if m else None


def _enrich(kernel: str, data: dict) -> dict:
    """Fill in computed roofline / occupancy fields missing from the JSON."""
    from src.roofline import RooflineAnalyzer, KERNEL_DIMS
    best = data.get("best", {})
    if not best:
        return data

    try:
        ra   = RooflineAnalyzer()
        dims = KERNEL_DIMS.get(kernel, (1024,))

        ai = best.get("arithmetic_intensity") or ra.arithmetic_intensity(kernel, dims)
        best.setdefault("arithmetic_intensity", ai)

        best_ms = best.get("mean_ms", 0)
        if best_ms and not best.get("achieved_gflops"):
            best["achieved_gflops"] = ra.achieved_gflops(kernel, dims, best_ms)

        if not best.get("bound_type"):
            best["bound_type"] = "compute" if ai >= ra.ridge_point else "memory"

        eff = best.get("roofline_efficiency_pct")
        if not eff and best.get("achieved_gflops"):
            peak = ra.peak_performance(kernel, dims)
            eff  = best["achieved_gflops"] / peak * 100 if peak else 0
            best["roofline_efficiency_pct"] = eff

        ptx = best.get("ptx_info", "")
        if ptx and not best.get("registers_per_thread"):
            reg = _parse_ptx_registers(ptx)
            if reg:
                best["registers_per_thread"] = reg
        if ptx and not best.get("shared_mem_bytes"):
            smem = _parse_ptx_smem(ptx)
            if smem is not None:
                best["shared_mem_bytes"] = smem

        # Compute speedup from baseline.json when missing (e.g. attention)
        if best.get("speedup") is None and best_ms:
            baseline_file = RESULTS_DIR / "baseline.json"
            if baseline_file.exists():
                bl = json.loads(baseline_file.read_text())
                bl_entry = next(
                    (v for tag, v in bl.items() if tag.startswith(kernel) and isinstance(v, dict)),
                    None,
                )
                bl_ms = (bl_entry or {}).get("mean_ms")
                if bl_ms:
                    best["speedup"] = bl_ms / best_ms
                    if not data.get("baseline_ms"):
                        data["baseline_ms"] = bl_ms
    except Exception:
        pass

    data["best"] = best
    return data


@st.cache_data(ttl=5)
def get_kernel_results(kernel: str) -> dict:
    f = RESULTS_DIR / f"{kernel}_tuning.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text())
    return _enrich(kernel, data)


@st.cache_data(ttl=5)
def get_baseline() -> dict:
    f = RESULTS_DIR / "baseline.json"
    return json.loads(f.read_text()) if f.exists() else {}


def is_tuning_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        PID_FILE.unlink(missing_ok=True)
        return False


def read_log_tail(n: int = 25) -> list[str]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(errors="replace").splitlines()
    return lines[-n:]


def parse_progress(lines: list[str]) -> tuple[int, int]:
    """Return (done, total) from the latest [N/M] line, or (0, 0)."""
    for line in reversed(lines):
        if "[" in line and "/" in line and "]" in line:
            try:
                inner = line[line.index("[") + 1: line.index("]")]
                if "/" in inner:
                    d, t = inner.strip().split("/")
                    return int(d.strip()), int(t.strip())
            except Exception:
                pass
    return 0, 0


def parse_latest_ms(lines: list[str]) -> float | None:
    """Return the most recently reported latency in ms."""
    for line in reversed(lines):
        for token in line.split():
            if token.endswith("ms") and "." in token:
                try:
                    return float(token.replace("ms", ""))
                except ValueError:
                    pass
    return None


def parse_best_speedup(lines: list[str]) -> float | None:
    """Return speedup if a BEST line is found."""
    for line in reversed(lines):
        if "speedup:" in line.lower():
            for part in line.split():
                if "x" in part.lower():
                    try:
                        return float(part.lower().replace("x", ""))
                    except ValueError:
                        pass
    return None


# ── Plotting helpers ─────────────────────────────────────────────────────────

KERNEL_COLORS = {
    "matmul":    "#3b82f6",
    "softmax":   "#f97316",
    "reduction": "#eab308",
    "layernorm": "#a855f7",
    "attention": "#ec4899",
}

def roofline_fig(kernel: str, results: dict) -> go.Figure:
    from src.roofline import RooflineAnalyzer, KERNEL_DIMS

    best        = results.get("best", {})
    baseline_ms = results.get("baseline_ms", 0)
    ra          = RooflineAnalyzer()
    dims        = KERNEL_DIMS.get(kernel, (1024,))

    peak_compute_gflops = 7500
    peak_mem_bw         = 448
    ridge               = peak_compute_gflops / peak_mem_bw

    import math
    ai_vals = [10 ** (x * 0.08) for x in range(-12, 52)]
    roof    = [min(ai * peak_mem_bw, peak_compute_gflops) for ai in ai_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ai_vals, y=roof, mode="lines", name="Roofline boundary",
        line=dict(color="#8b5cf6", width=3),
    ))
    # Ridge point vertical line
    fig.add_shape(type="line",
                  x0=ridge, x1=ridge, y0=0, y1=peak_compute_gflops * 1.2,
                  line=dict(color="#475569", width=1, dash="dash"))
    fig.add_annotation(x=math.log10(ridge), y=math.log10(peak_compute_gflops * 0.6),
                       text=f"Ridge: {ridge:.1f}", showarrow=False,
                       font=dict(color="#475569", size=10))

    b_ai = ra.arithmetic_intensity(kernel, dims)
    # Baseline point
    if baseline_ms:
        try:
            b_gf = ra.achieved_gflops(kernel, dims, baseline_ms)
            fig.add_trace(go.Scatter(
                x=[b_ai], y=[b_gf], mode="markers+text", name="Baseline",
                text=["Baseline"], textposition="bottom center",
                textfont=dict(color="#ef4444", size=10),
                marker=dict(size=13, color="#ef4444",
                            line=dict(width=2, color="rgba(239,68,68,0.4)")),
            ))
        except Exception:
            pass

    # Best variant point
    ai   = best.get("arithmetic_intensity") or b_ai
    best_gf = best.get("achieved_gflops", 0)
    if not best_gf and best.get("mean_ms"):
        try:
            best_gf = ra.achieved_gflops(kernel, dims, best["mean_ms"])
        except Exception:
            pass
    if best_gf:
        fig.add_trace(go.Scatter(
            x=[ai], y=[best_gf], mode="markers+text", name="Best variant",
            text=["Best"], textposition="top center",
            textfont=dict(color="#10b981", size=10),
            marker=dict(size=16, color="#10b981",
                        symbol="star",
                        line=dict(width=2, color="rgba(16,185,129,0.4)")),
        ))

    fig.update_layout(**chart_layout(
        title="Roofline Model (RTX 2070)",
        xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log"),
        yaxis=dict(title="Performance (GFLOP/s)", type="log"),
    ))
    return fig


def convergence_fig(variants: list[dict], color: str = "#3b82f6") -> go.Figure:
    df = pd.DataFrame([
        {"variant": i + 1, "latency_ms": v.get("mean_ms", 0)}
        for i, v in enumerate(variants[:50])
    ])
    fig = px.line(df, x="variant", y="latency_ms",
                  title="Latency over Variants",
                  color_discrete_sequence=[color])
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(**chart_layout())
    return fig


def speedup_fig(variants: list[dict]) -> go.Figure:
    df = pd.DataFrame([
        {"variant": i + 1, "speedup": v.get("speedup", 0)}
        for i, v in enumerate(variants[:30])
    ])
    fig = px.bar(df, x="variant", y="speedup",
                 title="Speedup per Variant",
                 color="speedup",
                 color_continuous_scale=["#1e40af", "#3b82f6", "#10b981"])
    fig.update_layout(**chart_layout(coloraxis_showscale=False))
    return fig


def gpu_history_fig(util_history: list, mem_history: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=util_history, name="GPU Util%",
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        y=mem_history, name="Mem Util%",
        mode="lines+markers",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(**chart_layout(
        title="GPU Activity (live)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
    ))
    return fig


# ── Session state init ───────────────────────────────────────────────────────

for _k, _v in [
    ("gpu_util_hist",  []),   # GPU utilisation %
    ("gpu_mem_hist",   []),   # VRAM utilisation %
    ("gpu_temp_hist",  []),   # temperature °C
    ("gpu_power_hist", []),   # power draw W
    ("best_ms_hist",   []),   # best latency seen so far (convergence)
    ("done_hist",      []),   # variant index for convergence x-axis
    ("gpu_seeded",     False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Seed GPU history with 8 samples on first load so chart isn't blank
if not st.session_state.gpu_seeded:
    _g0 = get_gpu_stats()
    if _g0:
        for _ in range(8):
            st.session_state.gpu_util_hist.append(_g0["gpu_util"])
            st.session_state.gpu_mem_hist.append(_g0["mem_util"])
            st.session_state.gpu_temp_hist.append(_g0["temp"])
            st.session_state.gpu_power_hist.append(_g0["power"])
    st.session_state.gpu_seeded = True

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🚀 CUDA Auto-Tuner")
    st.markdown("---")

    running = is_tuning_running()

    # ── START / STOP button at the TOP ──────────────────────────────────────
    if not running:
        _start_clicked = st.button("⚡ Start Tuning", use_container_width=True, type="primary")
    else:
        st.markdown(
            '<div style="background:rgba(16,185,129,0.12);border:1px solid #10b981;'
            'border-radius:10px;padding:8px 14px;color:#10b981;font-weight:700;'
            'font-size:0.9em;text-align:center;margin-bottom:8px;">🔥 TUNING RUNNING</div>',
            unsafe_allow_html=True,
        )
        _start_clicked = False

    st.markdown("---")
    st.markdown("##### ⚙️ Tuning Config")
    seed = st.selectbox("Kernel", ["matmul", "softmax", "reduction", "layernorm", "attention", "all"])
    kernel_sel   = seed
    strategy_sel = st.selectbox("Strategy", ["grid", "bayesian", "sha"])
    workers      = st.slider("Workers", 1, 8, 2)
    warmup       = st.slider("Warmup iters", 2, 20, 5)
    samples      = st.slider("Samples", 10, 100, 30)
    skip_verify  = st.checkbox("Skip verification (faster)", value=False)
    ptx          = st.checkbox("PTX Analysis", value=False)
    cuda_graphs  = st.checkbox("CUDA Graphs", value=False)
    matrix_size  = st.radio("Matrix Size (N)", [512, 1024, 2048], index=1, horizontal=True)

    st.markdown("---")

    if running:
        if st.button("🛑 Stop Tuning", use_container_width=True):
            try:
                pid = int(PID_FILE.read_text().strip())
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            PID_FILE.unlink(missing_ok=True)
            CONFIG_FILE.unlink(missing_ok=True)
            st.rerun()

    if not running and _start_clicked:
            # Write config
            CONFIG_FILE.write_text(json.dumps({
                "kernel":   kernel_sel,
                "strategy": strategy_sel,
                "workers":  workers,
                "warmup":   warmup,
                "samples":  samples,
                "skip_verification": skip_verify,
                "ptx_analysis":      ptx,
                "cuda_graphs":       cuda_graphs,
            }))
            # Clear old log and reset convergence history
            LOG_FILE.write_text("")
            st.session_state.best_ms_hist = []
            st.session_state.done_hist    = []
            # Build command
            cmd = [
                sys.executable,
                str(ROOT / "autotune.py"),
                f"--kernel={kernel_sel}",
                f"--strategy={strategy_sel}",
                f"--workers={workers}",
                f"--warmup={warmup}",
                f"--samples={samples}",
            ]
            if skip_verify:
                cmd.append("--skip-verification")
            if ptx:
                cmd.append("--ptx-analysis")
            if cuda_graphs:
                cmd.append("--cuda-graphs")
            cmd.append(f"--matrix-size={matrix_size}")

            with open(LOG_FILE, "w") as lf:
                proc = subprocess.Popen(
                    cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
                )
            PID_FILE.write_text(str(proc.pid))
            st.rerun()

    st.markdown("---")
    st.markdown("##### 📊 Results Cache")
    baseline = get_baseline()
    if baseline:
        baseline_items = baseline.get("kernels") if isinstance(baseline.get("kernels"), dict) else baseline
        for tag, data in baseline_items.items():
            if isinstance(data, dict):
                st.caption(f"• {tag}: {data.get('mean_ms', 0):.3f} ms")
    st.markdown("---")
    st.markdown("""
    <div style='color:#475569; font-size:0.78em; line-height:1.8;'>
    RTX 2070 (sm_75)<br>
    CUDA 11.0+ &nbsp;|&nbsp; Python 3.10+<br>
    Streamlit Dashboard
    </div>
    """, unsafe_allow_html=True)


# ── Page header ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-title">
  <h1>🚀 CUDA Auto-Tuner</h1>
  <p>Systematic GPU Kernel Optimization via Parameter Space Exploration</p>
  <span class="hw-badge">RTX 2070 (sm_75) &nbsp;•&nbsp; 36 SMs &nbsp;•&nbsp; 8 GB VRAM</span>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ── GPU Stats strip ──────────────────────────────────────────────────────────

@st.fragment(run_every=1)
def _live_gpu():
    """
    Runs every 1 s independently. Appends to session_state history arrays
    so the chart grows in real-time without re-rendering the whole page.
    """
    g = get_gpu_stats()
    r = is_tuning_running()

    if not g:
        st.warning("⚠️ GPU not detected — running in demo / results-only mode")
        return

    # ── Append to rolling history (2-minute window at 1 s ticks) ──────────
    st.session_state.gpu_util_hist.append(g["gpu_util"])
    st.session_state.gpu_mem_hist.append(g["mem_util"])
    st.session_state.gpu_temp_hist.append(g["temp"])
    st.session_state.gpu_power_hist.append(g["power"])
    for _k in ("gpu_util_hist", "gpu_mem_hist", "gpu_temp_hist", "gpu_power_hist"):
        st.session_state[_k] = st.session_state[_k][-120:]

    dot_cls    = "live-dot" if r else "live-dot dead"
    status_txt = "TUNING 🔥" if r else "IDLE"

    # ── Stats strip ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="gpu-strip">
      <div class="gpu-stat">
        <div class="gpu-stat-val">{g['gpu_util']:.0f}%</div>
        <div class="gpu-stat-lbl">GPU Util</div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-val">{g['mem_util']:.0f}%</div>
        <div class="gpu-stat-lbl">Mem Util</div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-val">{g['mem_used']:.0f} / {g['mem_total']:.0f} MB</div>
        <div class="gpu-stat-lbl">VRAM Used</div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-val">{g['temp']:.0f} °C</div>
        <div class="gpu-stat-lbl">Temperature</div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-val">{g['power']:.0f} W</div>
        <div class="gpu-stat-lbl">Power Draw</div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-val">{g.get('sm_clock', '—')} MHz</div>
        <div class="gpu-stat-lbl">SM Clock</div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-val">36</div>
        <div class="gpu-stat-lbl">SMs</div>
      </div>
      <div class="gpu-stat">
        <div style="display:flex;align-items:center;justify-content:center;gap:6px;">
          <span class="{dot_cls}"></span>
          <span style="color:#94a3b8;font-size:0.9em;font-weight:600;">{status_txt}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live GPU Activity chart — always visible ───────────────────────────
    n   = len(st.session_state.gpu_util_hist)
    xs  = list(range(-n + 1, 1))   # negative = seconds ago → 0 = now

    # Normalise temp and power to 0-100 for overlay on the same axis
    temp_max  = max(max(st.session_state.gpu_temp_hist, default=100), 100)
    pwr_max   = max(max(st.session_state.gpu_power_hist, default=200), 1)
    temp_norm = [t / temp_max * 100 for t in st.session_state.gpu_temp_hist]
    pwr_norm  = [p / pwr_max * 100  for p in st.session_state.gpu_power_hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=st.session_state.gpu_util_hist,
        name="GPU Util %", mode="lines",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=st.session_state.gpu_mem_hist,
        name="Mem Util %", mode="lines",
        line=dict(color="#8b5cf6", width=2),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.10)",
    ))
    if n > 1:
        fig.add_trace(go.Scatter(
            x=xs, y=temp_norm,
            name=f"Temp (÷{temp_max:.0f}°C×100)", mode="lines",
            line=dict(color="#f97316", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=pwr_norm,
            name=f"Power (÷{pwr_max:.0f}W×100)", mode="lines",
            line=dict(color="#10b981", width=1.5, dash="dot"),
        ))
    title_txt = "🔥 Live GPU Activity — TUNING IN PROGRESS" if r else "Live GPU Activity  (last 2 min · 1 s ticks)"
    fig.update_layout(**chart_layout(
        title=title_txt,
        height=240,
        xaxis=dict(title="Seconds ago", ticksuffix="s"),
        yaxis=dict(title="%", range=[0, 105], ticksuffix="%"),
        legend=dict(orientation="h", y=-0.35),
        title_font=dict(color="#f97316" if r else "#64748b", size=14),
    ))
    if r:
        fig.add_annotation(
            x=xs[-1] if xs else 0, y=105,
            text="● LIVE",
            showarrow=False,
            font=dict(color="#10b981", size=12, family="JetBrains Mono"),
            xanchor="right",
        )
    st.plotly_chart(fig, use_container_width=True, key="gpu_live_chart")


@st.fragment(run_every=2)
def _live_tuning():
    """
    Full live analytics dashboard — runs every 2 s while tuning is active.
    Computes GFLOP/s, roofline position, speedup, convergence all live from
    live_progress.json. Disappears when idle.
    """
    from src.roofline import RooflineAnalyzer, KERNEL_DIMS
    import math

    r   = is_tuning_running()
    cfg: dict = {}
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass

    if not r and not cfg:
        return   # nothing running — render nothing

    # ── Read live_progress.json ────────────────────────────────────────────
    live: dict = {}
    lp = RESULTS_DIR / "live_progress.json"
    if lp.exists():
        try:
            live = json.loads(lp.read_text())
        except Exception:
            pass

    done      = live.get("done", 0)
    total     = live.get("total", 0)
    best_ms   = live.get("best_ms")
    best_sp   = live.get("best_speedup")
    baseline  = live.get("baseline_ms")
    kernel    = cfg.get("kernel", live.get("kernel", "?"))
    strategy  = cfg.get("strategy", live.get("strategy", "?"))
    recent    = live.get("recent", [])
    log_lines = read_log_tail(30)

    # ── Compute live analytics from best_ms ────────────────────────────────
    ra          = RooflineAnalyzer()
    dims        = KERNEL_DIMS.get(kernel, (1024,))
    ai          = None
    best_gflops = None
    bl_gflops   = None
    eff_pct     = None
    bound_type  = None
    ridge       = 7500 / 448

    try:
        ai = ra.arithmetic_intensity(kernel, dims)
        if best_ms:
            best_gflops = ra.achieved_gflops(kernel, dims, best_ms)
            peak        = ra.peak_performance(kernel, dims)
            eff_pct     = best_gflops / peak * 100 if peak else 0
            bound_type  = "Compute" if ai >= ridge else "Memory"
        if baseline:
            bl_gflops = ra.achieved_gflops(kernel, dims, baseline)
    except Exception:
        pass

    # ── Track convergence history ──────────────────────────────────────────
    prev_done = len(st.session_state.done_hist)
    if done > prev_done and best_ms:
        prev_best = st.session_state.best_ms_hist[-1] if st.session_state.best_ms_hist else best_ms
        step_ms   = (prev_best - best_ms) / max(done - prev_done, 1)
        for i in range(prev_done, done):
            frac = prev_best - step_ms * (i - prev_done + 1)
            st.session_state.done_hist.append(i + 1)
            st.session_state.best_ms_hist.append(round(max(frac, best_ms), 4))

    # ══════════════════════════════════════════════════════════════════════
    # LIVE DASHBOARD HEADER
    # ══════════════════════════════════════════════════════════════════════
    color = KERNEL_COLORS.get(kernel, "#3b82f6")
    pct   = done / total if total else 0

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(17,24,39,0.95),rgba(10,14,23,0.98));
                border:1px solid {color}44; border-radius:20px; padding:20px 28px; margin-bottom:16px;">
      <div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
        <span class="live-dot"></span>
        <span style="font-size:1.4em;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;">
          Live Tuning — <span style="color:{color}">{kernel.upper()}</span>
        </span>
        <span style="margin-left:auto;background:rgba(16,185,129,0.12);color:#10b981;
                     padding:4px 14px;border-radius:20px;font-size:0.8em;font-weight:600;">
          {strategy.upper()} · {cfg.get('workers','?')} workers · {cfg.get('samples','?')} samples
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(pct, text=f"Variants: {done} / {total}  ({pct*100:.1f}%)" if total else "Compiling first variant…")

    if done == 0 and r:
        st.info("⚙️ Compiling + warming up first variant — GPU utilisation will spike shortly.", icon="🔧")

    st.markdown("")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1 — 6 LIVE HERO METRICS (computed right now)
    # ══════════════════════════════════════════════════════════════════════
    mc = st.columns(6)
    def _hero(col, val, label, sub="", color_sub="#64748b"):
        with col:
            st.markdown(f"""
            <div class="metric-hero">
              <div class="val">{val}</div>
              <div class="lbl">{label}</div>
              <div style="color:{color_sub};font-size:0.75em;margin-top:6px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    _hero(mc[0],
          f"{best_sp:.2f}×"          if best_sp   else "—",   "Speedup vs Baseline",
          f"baseline {baseline:.2f} ms" if baseline else "",    "#10b981" if best_sp and best_sp > 1 else "#64748b")
    _hero(mc[1],
          f"{best_ms:.3f} ms"        if best_ms   else "—",   "Best Latency")
    _hero(mc[2],
          f"{best_gflops:.0f}"       if best_gflops else "—", "GFLOP/s  (live)",
          f"baseline {bl_gflops:.0f} GFLOP/s" if bl_gflops else "", "#3b82f6")
    _hero(mc[3],
          f"{eff_pct:.1f}%"          if eff_pct   else "—",   "Roofline Efficiency",
          "of theoretical peak", "#8b5cf6")
    _hero(mc[4],
          bound_type                 if bound_type else "—",   "Bound Type",
          f"AI = {ai:.1f} FLOP/B"   if ai else "",
          "#ef4444" if bound_type == "Memory" else "#10b981")
    _hero(mc[5],
          f"{done}",                                           "Variants Done",
          f"of {total}" if total else "")

    st.markdown("")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2 — CONVERGENCE  +  LIVE ROOFLINE  +  GPU ACTIVITY
    # ══════════════════════════════════════════════════════════════════════
    col_conv, col_roof, col_gpu = st.columns([2, 2, 2])

    # ── Convergence chart ──────────────────────────────────────────────────
    with col_conv:
        st.markdown('<div class="section-head">📉 Convergence</div>', unsafe_allow_html=True)
        cfig = go.Figure()
        if st.session_state.done_hist:
            cfig.add_trace(go.Scatter(
                x=st.session_state.done_hist,
                y=st.session_state.best_ms_hist,
                mode="lines+markers",
                name="Best ms",
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
                fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
            ))
            if baseline:
                cfig.add_hline(y=baseline, line_dash="dash", line_color="#ef4444",
                               annotation_text=f"Baseline {baseline:.2f} ms",
                               annotation_position="bottom right",
                               annotation_font_color="#ef4444")
        else:
            cfig.add_annotation(text="Waiting for first variant…",
                                xref="paper", yref="paper", x=0.5, y=0.5,
                                showarrow=False, font=dict(color="#475569", size=13))
        cfig.update_layout(**chart_layout(
            height=280,
            xaxis=dict(title="Variant #"),
            yaxis=dict(title="Best latency (ms)"),
        ))
        st.plotly_chart(cfig, use_container_width=True, key="live_convergence")

    # ── Live Roofline chart (dot moves as ms improves) ─────────────────────
    with col_roof:
        st.markdown('<div class="section-head">📐 Live Roofline</div>', unsafe_allow_html=True)
        ai_vals = [10 ** (x * 0.08) for x in range(-12, 52)]
        roof_y  = [min(a * 448, 7500) for a in ai_vals]
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(
            x=ai_vals, y=roof_y, mode="lines", name="Roofline",
            line=dict(color="#8b5cf6", width=2.5),
        ))
        rfig.add_shape(type="line",
                       x0=ridge, x1=ridge, y0=0, y1=7500 * 1.2,
                       line=dict(color="#475569", width=1, dash="dash"))
        rfig.add_annotation(x=math.log10(ridge), y=math.log10(7500 * 0.5),
                            text=f"Ridge {ridge:.1f}", showarrow=False,
                            font=dict(color="#475569", size=9))
        if ai and bl_gflops:
            rfig.add_trace(go.Scatter(
                x=[ai], y=[bl_gflops], mode="markers+text",
                name="Baseline", text=["Baseline"],
                textposition="bottom center",
                textfont=dict(color="#ef4444", size=9),
                marker=dict(size=12, color="#ef4444"),
            ))
        if ai and best_gflops:
            rfig.add_trace(go.Scatter(
                x=[ai], y=[best_gflops], mode="markers+text",
                name="Best now", text=["Best now"],
                textposition="top center",
                textfont=dict(color="#10b981", size=9),
                marker=dict(size=16, color="#10b981", symbol="star"),
            ))
        rfig.update_layout(**chart_layout(
            height=280,
            xaxis=dict(title="AI (FLOP/B)", type="log"),
            yaxis=dict(title="GFLOP/s",     type="log"),
        ))
        st.plotly_chart(rfig, use_container_width=True, key="live_roofline")

    # ── GPU activity (shared session_state from _live_gpu) ─────────────────
    with col_gpu:
        st.markdown('<div class="section-head">⚡ GPU Activity</div>', unsafe_allow_html=True)
        n  = len(st.session_state.gpu_util_hist)
        xs = list(range(-n + 1, 1))
        gfig = go.Figure()
        if n:
            gfig.add_trace(go.Scatter(
                x=xs, y=st.session_state.gpu_util_hist,
                name="GPU Util %", mode="lines",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
            ))
            gfig.add_trace(go.Scatter(
                x=xs, y=st.session_state.gpu_mem_hist,
                name="Mem Util %", mode="lines",
                line=dict(color="#8b5cf6", width=2),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.10)",
            ))
            pwr_max  = max(max(st.session_state.gpu_power_hist, default=1), 1)
            pwr_norm = [p / pwr_max * 100 for p in st.session_state.gpu_power_hist]
            gfig.add_trace(go.Scatter(
                x=xs, y=pwr_norm,
                name=f"Power (÷{pwr_max:.0f}W×100)", mode="lines",
                line=dict(color="#10b981", width=1.5, dash="dot"),
            ))
        gfig.update_layout(**chart_layout(
            height=280,
            xaxis=dict(title="Seconds ago", ticksuffix="s"),
            yaxis=dict(title="%", range=[0, 105], ticksuffix="%"),
            legend=dict(orientation="h", y=-0.4),
        ))
        st.plotly_chart(gfig, use_container_width=True, key="live_gpu_analytics")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 3 — TOP VARIANTS TABLE (with live GFLOP/s column)
    # ══════════════════════════════════════════════════════════════════════
    if recent:
        st.markdown('<div class="section-head">🏅 Top Variants — Live Leaderboard</div>',
                    unsafe_allow_html=True)
        rows_r = []
        for rank, v in enumerate(recent, 1):
            v_ms  = v.get("mean_ms") or 0
            v_gf  = 0
            try:
                if v_ms:
                    v_gf = ra.achieved_gflops(kernel, dims, v_ms)
            except Exception:
                pass
            p     = v.get("params", v.get("variant", {}))
            cfg_s = (", ".join(f"{k}={val}" for k, val in list(p.items())[:4])
                     if isinstance(p, dict) else str(p))
            rows_r.append({
                "Rank":    f"#{rank}",
                "ms":      f"{v_ms:.3f}",
                "GFLOP/s": f"{v_gf:.0f}" if v_gf else "—",
                "Speedup": f"{v.get('speedup') or 0:.2f}×",
                "Config":  cfg_s,
            })
        st.dataframe(pd.DataFrame(rows_r), hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 4 — LIVE LOG
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">📋 Live Log</div>', unsafe_allow_html=True)
    log_text = "\n".join(log_lines) if log_lines else "Waiting for output…"
    st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

    # ── Handle tuning-finished state ───────────────────────────────────────
    if not r and cfg:
        CONFIG_FILE.unlink(missing_ok=True)
        get_kernel_results.clear()
        st.success("✅ Tuning complete! Results are updated in the tabs below.")
        st.rerun(scope="app")

    st.markdown("---")


# ── Render both live fragments ───────────────────────────────────────────────
_live_gpu()
running = is_tuning_running()
_live_tuning()

# ── Per-Kernel Tabs ──────────────────────────────────────────────────────────

KERNEL_META = {
    "matmul":    {"icon": "📊", "label": "Matrix Multiply", "color": "#3b82f6"},
    "softmax":   {"icon": "🔥", "label": "Softmax",         "color": "#f97316"},
    "reduction": {"icon": "⚡",  "label": "Reduction",       "color": "#eab308"},
    "layernorm": {"icon": "📐", "label": "Layer Norm",       "color": "#a855f7"},
    "attention": {"icon": "🧠", "label": "Attention",        "color": "#ec4899"},
}

tab_labels = ["🔨 Compiler Pipeline"] + [f"{m['icon']} {m['label']}" for m in KERNEL_META.values()] + ["📈 Compare All"]
tabs       = st.tabs(tab_labels)

# ── Compiler Pipeline Tab (Unit I-VI) ──────────────────────────────────────
with tabs[0]:
    st.markdown("## 🔨 Compiler Design Pipeline: CUDA C Subset")
    st.markdown("**Interactive demonstration of all compiler phases**: Lexical Analysis → Syntax Analysis → Semantic Analysis → Intermediate Code Generation → Code Generation → Optimization")

    if not CDC_AVAILABLE:
        st.error(f"CDC modules not available: {CDC_ERROR}")
    else:
        # Kernel selector
        kernel_name = st.selectbox("Select kernel to compile", list(KERNEL_META.keys()), key="cd_kernel_selector")

        # Create subtabs for each phase
        phase_tabs = st.tabs(["📝 Frontend (Phase 1)", "🔀 IR (Phase 2)", "⚙️  Optimization (Phase 3)", "📊 Parse Table"])

        # Load source file
        src_file = ROOT / "src" / "kernels" / "baseline_kernels.cu"
        if not src_file.exists():
            st.error(f"Source file not found: {src_file}")
        else:
            # Run frontend to extract kernels
            try:
                frontend_result = run_frontend(src_file)
                kernel_obj = frontend_result.by_name(kernel_name)

                if not kernel_obj:
                    st.error(f"Kernel '{kernel_name}' not found in source")
                elif not kernel_obj.ok():
                    st.error(f"Frontend errors in {kernel_name}: {[d.message for d in kernel_obj.diagnostics]}")
                else:
                    # ── Phase 1: Frontend ──────────────────────────────────────
                    with phase_tabs[0]:
                        st.write("### Lexical Analysis (Tokens)")
                        tokens_list = []
                        from cdc.preprocessor import preprocess
                        from cdc.lexer import tokenize
                        cleaned = preprocess(kernel_obj.source)
                        for tok in tokenize(cleaned):
                            tokens_list.append({"Line": tok.lineno, "Type": tok.type, "Value": repr(tok.value)})
                        if tokens_list:
                            st.dataframe(pd.DataFrame(tokens_list), use_container_width=True)

                        st.divider()
                        st.write("### Syntax Analysis (AST)")
                        from cdc.ast_nodes import pretty
                        st.code(pretty(kernel_obj.ast), language="text")

                        st.divider()
                        st.write("### Symbol Table")
                        symtab_rows = []
                        for scope in kernel_obj.scopes:
                            for sym in scope.symbols.values():
                                symtab_rows.append({
                                    "Name": sym.name,
                                    "Type": str(sym.type),
                                    "Kind": sym.kind,
                                    "Scope": scope.level,
                                })
                        if symtab_rows:
                            st.dataframe(pd.DataFrame(symtab_rows), use_container_width=True)

                        st.divider()
                        st.write("### Type Diagnostics")
                        if kernel_obj.diagnostics:
                            for d in kernel_obj.diagnostics:
                                if "error" in d.severity.lower():
                                    st.error(f"Line {d.line}: {d.message}")
                                elif "warning" in d.severity.lower():
                                    st.warning(f"Line {d.line}: {d.message}")
                        else:
                            st.success("✓ No type errors or warnings")

                    # ── Phase 2: Intermediate Representation ──────────────────
                    with phase_tabs[1]:
                        prog = emit_tac(kernel_obj.ast)
                        blocks = partition_blocks(prog)
                        cfg = build_cfg(blocks)

                        st.write("### Three-Address Code (TAC / Quadruples)")
                        tac_rows = []
                        for i, quad in enumerate(prog.quads):
                            tac_rows.append({
                                "#": i,
                                "Op": quad.op,
                                "Arg1": quad.arg1 or "—",
                                "Arg2": quad.arg2 or "—",
                                "Result": quad.result or "—",
                            })
                        st.dataframe(pd.DataFrame(tac_rows), use_container_width=True)

                        st.divider()
                        st.write("### Basic Blocks")
                        for bb in blocks:
                            with st.expander(f"Block {bb.id} ({len(bb.quads)} quads)"):
                                bb_rows = []
                                for i, quad in enumerate(bb.quads):
                                    bb_rows.append({"#": i, "Quad": str(quad)})
                                st.dataframe(pd.DataFrame(bb_rows), use_container_width=True)

                        st.divider()
                        st.write("### Control Flow Graph (CFG)")
                        st.text(cfg.format_edges())
                        st.divider()
                        st.text(cfg.format_dominators())

                    # ── Phase 3: Optimization ──────────────────────────────────
                    with phase_tabs[2]:
                        st.write("### Data Flow Analysis")

                        # Live Variables
                        lv = LiveVariables()
                        in_l, out_l = solve(cfg, lv)
                        st.text(format_dfa_result("Live Variables (backward, union)", in_l, out_l, cfg))

                        st.divider()

                        st.write("### Optimization Passes")
                        cp_stats = constant_propagation(prog, blocks)
                        cse_stats = common_subexpression_elimination(blocks)
                        sr_stats = strength_reduction(blocks)
                        licm_stats = loop_invariant_code_motion(blocks, cfg)
                        dce_stats = dead_code_elimination(blocks, cfg)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Const Propagation", f"{cp_stats['folded']} folded")
                        with col2:
                            st.metric("CSE", f"{cse_stats['eliminated']} eliminated")
                        with col3:
                            st.metric("Strength Reduction", f"{sr_stats['rewritten']} rewritten")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("LICM", f"{licm_stats['hoisted']} hoisted")
                        with col2:
                            st.metric("DCE", f"{dce_stats['removed']} removed")

                        st.divider()
                        st.write("### Register Pressure")
                        try:
                            rp = estimate_register_pressure(kernel_name, blocks, cfg)
                            st.text(format_rp_report(rp))
                        except Exception as e:
                            st.warning(f"Could not estimate register pressure: {e}")

                    # ── LL(1) Parse Table ──────────────────────────────────────
                    with phase_tabs[3]:
                        st.write("### FIRST/FOLLOW Sets and LL(1) Parse Table")
                        st.markdown("**Tutorial 5 / Lab Practical 7**: Compute FIRST/FOLLOW sets from grammar and construct LL(1) predictive parse table.")

                        grammar = build_example_grammar()
                        nullable = compute_nullable(grammar)
                        first = compute_first(grammar, nullable)
                        follow = compute_follow(grammar, first, nullable)

                        st.text(format_first_follow(first, follow, grammar.non_terminals))

                        st.divider()
                        table = build_ll1_table(grammar, first, follow, nullable)
                        st.text(format_ll1_table(table, grammar))

                        st.info("ℹ️ This grammar is LL(1)-friendly for educational purposes. The actual LALR(1) parser resolves ambiguities using shift/reduce decisions.")

            except Exception as e:
                st.error(f"Error in compiler pipeline: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── Kernel-Specific Tabs ────────────────────────────────────────────────────
for tab_widget, (kernel, meta) in zip(tabs[1:], KERNEL_META.items()):
    with tab_widget:
        results  = get_kernel_results(kernel)
        baseline = get_baseline()
        color    = meta["color"]

        if not results:
            st.markdown(f"""
            <div class="no-data-card">
              <div class="icon">{meta["icon"]}</div>
              <div style="font-size:1.1em;font-weight:600;color:#64748b;margin-bottom:8px;">
                No tuning results yet
              </div>
              <div>Run: <code>python autotune.py --kernel={kernel}</code></div>
              <div style="margin-top:8px;color:#334155;font-size:0.85em;">
                Or use the sidebar controls and click ⚡ Start Tuning
              </div>
            </div>
            """, unsafe_allow_html=True)
            continue

        best       = results.get("best", {})
        variants   = results.get("variants", [])
        baseline_ms = results.get("baseline_ms", 0)
        best_ms    = best.get("mean_ms", 0)

        speedup    = best.get("speedup") or 0
        gflops     = best.get("achieved_gflops") or 0
        occupancy  = (best.get("occupancy") or 0) * 100
        eff_pct    = best.get("roofline_efficiency_pct") or 0
        lat_delta  = f"-{((baseline_ms - best_ms) / baseline_ms * 100):.0f}%" if baseline_ms and best_ms else "—"
        sp_delta   = f"↑ {(speedup-1)*100:.0f}% faster" if speedup > 1 else ("—" if speedup == 0 else f"↓ {(1-speedup)*100:.0f}%")

        # ── Hero metrics ─────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, f"{speedup:.2f}×" if speedup else "—", "Speedup",    sp_delta, "delta-up" if speedup > 1 else "delta-neutral"),
            (c2, f"{best_ms:.3f} ms" if best_ms else "—", "Latency",  lat_delta, "delta-down"),
            (c3, f"{gflops:.0f}" if gflops else "—", "GFLOP/s",      f"~{eff_pct:.1f}% of peak", "delta-neutral"),
            (c4, f"{occupancy:.0f}%", "Occupancy", "SM utilisation", "delta-neutral"),
        ]
        for col, val, lbl, delta, delta_cls in cards:
            with col:
                st.markdown(f"""
                <div class="metric-hero">
                  <div class="val">{val}</div>
                  <div class="lbl">{lbl}</div>
                  <span class="{delta_cls}">{delta}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Charts row 1 ─────────────────────────────────────────────
        col_l, col_r = st.columns(2)
        with col_l:
            if variants:
                st.plotly_chart(convergence_fig(variants, color), use_container_width=True, key=f"{kernel}_convergence")
        with col_r:
            if variants:
                st.plotly_chart(speedup_fig(variants), use_container_width=True, key=f"{kernel}_speedup")

        # ── Roofline + stats ─────────────────────────────────────────
        col_roof, col_stats = st.columns([3, 1])
        with col_roof:
            try:
                st.plotly_chart(roofline_fig(kernel, results), use_container_width=True, key=f"{kernel}_roofline")
            except Exception:
                pass
        with col_stats:
            ai   = best.get("arithmetic_intensity", 0) or 0
            bound = best.get("bound_type", "—") or "—"
            st.markdown(f"""
            <div class="glass-card" style="margin-top:30px;">
              <div style="color:#94a3b8;font-size:0.75em;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">
                Roofline Stats
              </div>
              <table style="width:100%;color:#94a3b8;font-size:0.82em;border-collapse:collapse;">
                <tr><td style="padding:4px 0;color:#64748b;">Peak Compute</td><td style="text-align:right;color:#f1f5f9;font-weight:600;">7,500 GFLOP/s</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;">Peak Memory</td><td style="text-align:right;color:#f1f5f9;font-weight:600;">448 GB/s</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;">Ridge Point</td><td style="text-align:right;color:#f1f5f9;font-weight:600;">16.74 FLOP/B</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;">AI (kernel)</td><td style="text-align:right;color:#3b82f6;font-weight:600;">{ai:.1f} FLOP/B</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;">Bound</td><td style="text-align:right;color:{'#ef4444' if bound.lower()=='memory' else '#10b981'};font-weight:700;">{bound.capitalize()}</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;">Registers</td><td style="text-align:right;color:#f1f5f9;font-weight:600;">{best.get('registers_per_thread','—')}</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;">Shared Mem</td><td style="text-align:right;color:#f1f5f9;font-weight:600;">{best.get('shared_mem_bytes','—')} B</td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        # ── CSV Export ────────────────────────────────────────────────────────
        if variants:
            import io, csv as _csv
            buf = io.StringIO()
            w   = _csv.DictWriter(buf, fieldnames=[
                "rank", "variant", "mean_ms", "min_ms", "max_ms",
                "speedup", "p_value", "is_significant",
                "occupancy", "registers_per_thread",
                "achieved_gflops", "roofline_efficiency_pct", "bound_type"
            ], extrasaction="ignore")
            w.writeheader()
            for i, v in enumerate(variants[:50]):
                row = dict(v)
                row["rank"] = i + 1
                w.writerow(row)
            st.download_button(
                f"⬇️ Export Top-50 CSV",
                data=buf.getvalue(),
                file_name=f"{kernel}_top50.csv",
                mime="text/csv",
                key=f"{kernel}_csv_dl",
            )

        # ── Top-5 leaderboard ─────────────────────────────────────────
        st.markdown('<div class="section-head">🏆 Top Configurations</div>', unsafe_allow_html=True)
        if variants:
            top5 = variants[:5]
            ranks = ["🥇", "🥈", "🥉", "#4", "#5"]
            rows  = []
            for i, v in enumerate(top5):
                params  = v.get("params", {})
                cfg_str = ", ".join(f"{k}={val}" for k, val in list(params.items())[:3])
                sig     = "✅" if v.get("is_significant") else "—"
                rows.append({
                    "Rank":          ranks[i],
                    "Configuration": cfg_str,
                    "Latency (ms)":  f"{v.get('mean_ms') or 0:.3f}",
                    "Speedup":       f"{v.get('speedup') or 0:.2f}×",
                    "p-value":       f"{v.get('p_value') or 1:.2e}",
                    "Sig":           sig,
                })
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
            )


# ── Compare All tab (last tab) ───────────────────────────────────────────────

with tabs[-1]:
    all_results = {k: get_kernel_results(k) for k in KERNEL_META if get_kernel_results(k)}

    if not all_results:
        st.markdown("""
        <div class="no-data-card">
          <div class="icon">📈</div>
          <div style="font-size:1.1em;font-weight:600;color:#64748b;margin-bottom:8px;">
            No results yet
          </div>
          <div>Run tuning for at least one kernel first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Cross-kernel summary metrics ──────────────────────────────────
        cols = st.columns(len(all_results))
        for col, (kernel, res) in zip(cols, all_results.items()):
            meta = KERNEL_META[kernel]
            best = res.get("best", {})
            sp   = best.get("speedup") or 0
            ms   = best.get("mean_ms") or 0
            with col:
                st.markdown(f"""
                <div class="metric-hero">
                  <div style="font-size:1.5em;margin-bottom:4px;">{meta["icon"]}</div>
                  <div class="val">{sp:.2f}×</div>
                  <div class="lbl">{meta["label"]}</div>
                  <span class="delta-up">{ms:.3f} ms</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Combined roofline scatter ─────────────────────────────────────
        from src.roofline import RooflineAnalyzer, KERNEL_DIMS
        import math
        ra = RooflineAnalyzer()
        peak_c, peak_m = 7500, 448
        ridge = peak_c / peak_m
        ai_vals = [10 ** (x * 0.08) for x in range(-12, 52)]
        roof    = [min(ai * peak_m, peak_c) for ai in ai_vals]

        st.markdown('<div class="section-head">🔭 Combined Roofline</div>', unsafe_allow_html=True)
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=ai_vals, y=roof, mode="lines", name="Roofline",
                                  line=dict(color="#8b5cf6", width=3)))
        rfig.add_shape(type="line", x0=ridge, x1=ridge, y0=0, y1=peak_c * 1.2,
                       line=dict(color="#475569", width=1, dash="dash"))

        for kernel, res in all_results.items():
            best   = res.get("best", {})
            meta   = KERNEL_META[kernel]
            dims   = KERNEL_DIMS.get(kernel, (1024,))
            ai     = best.get("arithmetic_intensity") or ra.arithmetic_intensity(kernel, dims)
            gf     = best.get("achieved_gflops", 0)
            if not gf and best.get("mean_ms"):
                try:
                    gf = ra.achieved_gflops(kernel, dims, best["mean_ms"])
                except Exception:
                    pass
            if gf:
                rfig.add_trace(go.Scatter(
                    x=[ai], y=[gf], mode="markers+text",
                    name=kernel, text=[kernel],
                    textposition="top center",
                    textfont=dict(color=meta["color"], size=11),
                    marker=dict(size=16, color=meta["color"], symbol="star",
                                line=dict(width=2, color="white")),
                    hovertemplate=f"<b>{kernel}</b><br>AI: %{{x:.2f}} FLOP/byte<br>GFLOP/s: %{{y:.1f}}<extra></extra>",
                ))

        rfig.update_layout(**chart_layout(
            title="All Kernels — Roofline (RTX 2070)",
            xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log"),
            yaxis=dict(title="Performance (GFLOP/s)", type="log"),
        ))
        st.plotly_chart(rfig, use_container_width=True, key="compare_roofline")

        # ── Speedup comparison bar chart ──────────────────────────────────
        st.markdown('<div class="section-head">⚡ Speedup Comparison</div>', unsafe_allow_html=True)
        sp_fig = go.Figure()
        k_names = list(all_results.keys())
        baselines_ms = [all_results[k].get("baseline_ms") or 0 for k in k_names]
        bests_ms     = [all_results[k].get("best", {}).get("mean_ms") or 0 for k in k_names]
        speedups_bar = [b / t if t else 0 for b, t in zip(baselines_ms, bests_ms)]

        sp_fig.add_trace(go.Bar(
            name="Baseline",
            x=k_names, y=baselines_ms,
            marker_color="#374151",
            text=[f"{v:.3f} ms" for v in baselines_ms],
            textposition="outside",
        ))
        sp_fig.add_trace(go.Bar(
            name="Best Variant",
            x=k_names, y=bests_ms,
            marker_color=[KERNEL_META[k]["color"] for k in k_names],
            text=[f"{v:.3f} ms" for v in bests_ms],
            textposition="outside",
        ))
        for i, (k, sp) in enumerate(zip(k_names, speedups_bar)):
            if sp:
                sp_fig.add_annotation(
                    x=k, y=max(baselines_ms[i], bests_ms[i]) * 1.2,
                    text=f"{sp:.2f}×",
                    showarrow=False,
                    font=dict(color="#10b981" if sp >= 1 else "#ef4444", size=13, family="JetBrains Mono"),
                )
        sp_fig.update_layout(**chart_layout(
            title="Baseline vs Best Variant — Latency (ms)",
            barmode="group",
            xaxis=dict(title="Kernel"),
            yaxis=dict(title="Latency (ms)"),
        ))
        st.plotly_chart(sp_fig, use_container_width=True, key="compare_speedup")

        # ── Summary table ─────────────────────────────────────────────────
        st.markdown('<div class="section-head">📋 Summary Table</div>', unsafe_allow_html=True)
        rows = []
        for kernel, res in all_results.items():
            best = res.get("best", {})
            rows.append({
                "Kernel":        KERNEL_META[kernel]["icon"] + " " + kernel,
                "Baseline (ms)": f"{res.get('baseline_ms') or 0:.3f}",
                "Best (ms)":     f"{best.get('mean_ms') or 0:.3f}",
                "Speedup":       f"{best.get('speedup') or 0:.2f}×",
                "p-value":       f"{best.get('p_value') or 1:.2e}",
                "GFLOP/s":       f"{best.get('achieved_gflops') or 0:.1f}",
                "AI":            f"{best.get('arithmetic_intensity') or 0:.2f}",
                "Bound":         (best.get("bound_type") or "—").capitalize(),
                "Occupancy":     f"{(best.get('occupancy') or 0)*100:.0f}%",
                "Variants":      str(res.get("n_variants") or "—"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#1e293b;padding:20px 0 10px;font-size:0.82em;">
  🚀 CUDA Auto-Tuner &nbsp;•&nbsp; Built for GPU Optimization<br>
  RTX 2070 (sm_75) &nbsp;•&nbsp; CUDA 11.0+ &nbsp;•&nbsp; Python 3.10+ &nbsp;•&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
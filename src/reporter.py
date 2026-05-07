"""
reporter.py — Markdown report generation and terminal summary tables.

Generates results/{kernel}_report.md after each tuning run, containing:
  1. Header (kernel, date, hardware, strategy)
  2. Roofline summary (baseline vs best)
  3. Statistical results table (top 10 configs)
  4. Occupancy table (top 10 configs)
  5. PTX analysis section (if --ptx-analysis was run)
  6. Correctness summary
  7. Convergence note (for Bayesian / SHA strategies)
  8. Recommended config

Also provides print_terminal_summary() for clean stdlib-only terminal output
(no rich / tabulate dependency).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark import BenchmarkResult
from roofline import RooflineAnalyzer, get_dims

RESULTS_DIR = Path(__file__).parent.parent / "results"

_HW_LABEL = "RTX 2070 (sm_75) — peak 7.5 TFLOP/s FP32 — 448 GB/s DRAM"


# ── Markdown helpers ───────────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    )
    return "\n".join([head, sep, body])


def _fmt_float(v: Optional[float], decimals: int = 3, suffix: str = "") -> str:
    """Format an optional float, showing '—' for None."""
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{suffix}"


def _fmt_bool(v: bool) -> str:
    return "✓" if v else "✗"


# ── Report generator ───────────────────────────────────────────────────────

class ReportGenerator:
    """
    Build a full Markdown tuning report for one kernel.

    Usage::

        rg = ReportGenerator(kernel="matmul", strategy="bayesian")
        rg.set_results(sorted_results, baseline_ms=4.84)
        rg.set_roofline_points(baseline_pt, best_pt)
        rg.set_correctness(n_passed=158, n_failed=2)
        rg.generate()   # writes results/matmul_report.md
    """

    def __init__(
        self,
        kernel: str,
        strategy: str = "grid",
        results_dir: Path = RESULTS_DIR,
    ) -> None:
        """
        Args:
            kernel:      Kernel name (matmul | softmax | reduction | layernorm).
            strategy:    Search strategy used (grid | bayesian | sha).
            results_dir: Output directory for the report file.
        """
        self._kernel      = kernel
        self._strategy    = strategy
        self._results_dir = results_dir
        self._report_path = results_dir / f"{kernel}_report.md"

        # Populated by set_* methods
        self._results:   list[BenchmarkResult] = []
        self._baseline_ms: Optional[float] = None
        self._roofline_baseline: Optional[dict] = None
        self._roofline_best:     Optional[dict] = None
        self._n_passed  = 0
        self._n_failed  = 0
        self._n_checked = 0
        self._ptx_data: Optional[dict] = None
        self._convergence: Optional[dict] = None

    def set_results(
        self,
        results: list[BenchmarkResult],
        baseline_ms: Optional[float] = None,
    ) -> None:
        """
        Provide the full list of benchmark results (sorted best-first).

        Args:
            results:     All BenchmarkResult objects for this kernel.
            baseline_ms: Baseline mean latency for speedup column.
        """
        self._results    = results
        self._baseline_ms = baseline_ms

    def set_roofline_points(
        self,
        baseline_pt: Optional[dict],
        best_pt: Optional[dict],
    ) -> None:
        """
        Set roofline analysis dicts for baseline and best variant.

        Each dict should have keys: arithmetic_intensity, achieved_gflops,
        bound_type, efficiency_pct.
        """
        self._roofline_baseline = baseline_pt
        self._roofline_best     = best_pt

    def set_correctness(
        self, n_passed: int, n_failed: int, n_checked: int = 0
    ) -> None:
        """Record correctness verification summary counts."""
        self._n_passed  = n_passed
        self._n_failed  = n_failed
        self._n_checked = n_checked or (n_passed + n_failed)

    def set_ptx_data(self, ptx_data: Optional[dict]) -> None:
        """Provide PTX analysis dict (from ptx_analysis.save output)."""
        self._ptx_data = ptx_data

    def set_convergence(self, convergence: Optional[dict]) -> None:
        """Provide convergence curve dict (from ConvergenceLogger)."""
        self._convergence = convergence

    def generate(self) -> Path:
        """
        Write the Markdown report to results/{kernel}_report.md.

        Returns:
            Path to the generated report file.
        """
        lines: list[str] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 1. Header ──────────────────────────────────────────────────
        lines += [
            f"# CUDA Auto-Tuner Report — `{self._kernel}`",
            "",
            f"| Field     | Value |",
            f"|-----------|-------|",
            f"| Kernel    | `{self._kernel}` |",
            f"| Date      | {now} |",
            f"| Hardware  | {_HW_LABEL} |",
            f"| Strategy  | `{self._strategy}` |",
            "",
        ]

        # ── 2. Roofline summary ────────────────────────────────────────
        lines.append("## Roofline Analysis\n")
        lines.append(f"> Ridge point: **16.74 FLOP/byte** "
                     f"(7.5 TFLOP/s ÷ 448 GB/s)\n")
        if self._roofline_baseline or self._roofline_best:
            rf_headers = ["Config", "AI (FLOP/byte)", "Bound", "Achieved (GFLOP/s)", "Eff%"]
            rf_rows = []
            if self._roofline_baseline:
                b = self._roofline_baseline
                rf_rows.append([
                    "Baseline",
                    f"{b.get('arithmetic_intensity', 0):.2f}",
                    b.get("bound_type", "—"),
                    f"{b.get('achieved_gflops', 0):.1f}",
                    f"{b.get('efficiency_pct', 0):.1f}%",
                ])
            if self._roofline_best:
                o = self._roofline_best
                rf_rows.append([
                    "Best variant",
                    f"{o.get('arithmetic_intensity', 0):.2f}",
                    o.get("bound_type", "—"),
                    f"{o.get('achieved_gflops', 0):.1f}",
                    f"{o.get('efficiency_pct', 0):.1f}%",
                ])
            lines.append(_md_table(rf_headers, rf_rows))
            lines.append("")
        else:
            lines.append("*(Roofline data not collected in this run)*\n")

        # ── 3. Statistical results table (top 10) ─────────────────────
        lines.append("## Top 10 Configurations (Statistical)\n")
        top10 = self._results[:10]
        if top10:
            st_headers = ["Config", "mean_ms", "±CI", "speedup", "p-value", "sig?"]
            st_rows = []
            for r in top10:
                config = self._short_params(r.params)
                st_rows.append([
                    config,
                    f"{r.mean_ms:.3f}ms",
                    f"±{r.ci_95_ms:.3f}ms",
                    _fmt_float(r.speedup, 2, "x"),
                    _fmt_float(r.p_value, 4) if r.p_value is not None else "—",
                    "✓" if r.is_significant else "✗",
                ])
            lines.append(_md_table(st_headers, st_rows))
            lines.append("")
        else:
            lines.append("*(No results)*\n")

        # ── 4. Occupancy table (top 10) ───────────────────────────────
        occ_results = [r for r in top10 if r.occupancy is not None]
        if occ_results:
            lines.append("## Top 10 Occupancy Analysis\n")
            oc_headers = ["Config", "Occupancy%", "Regs", "SMEM (bytes)", "Spill", "Speedup"]
            oc_rows = []
            for r in occ_results:
                oc_rows.append([
                    self._short_params(r.params),
                    f"{(r.occupancy or 0) * 100:.1f}%",
                    str(r.registers_per_thread or "—"),
                    str(r.shared_mem_bytes or "—"),
                    "YES" if r.has_register_spill else "no",
                    _fmt_float(r.speedup, 2, "x"),
                ])
            lines.append(_md_table(oc_headers, oc_rows))
            lines.append("")

        # ── 5. PTX analysis ────────────────────────────────────────────
        if self._ptx_data and self._ptx_data.get("status") != "analysis_failed":
            lines.append("## PTX Analysis\n")
            pd = self._ptx_data
            lines.append(f"- Instruction delta: `{pd.get('instruction_delta', '—')}`")
            lines.append(f"- Memory op delta: `{pd.get('memory_op_delta', '—')}`")
            lines.append(f"- Compute ratio delta: `{pd.get('compute_ratio_delta', '—'):.3f}`")
            lines.append("")

        # ── 6. Correctness summary ─────────────────────────────────────
        lines.append("## Correctness Verification\n")
        if self._n_checked > 0:
            lines.append(f"| Metric | Count |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Variants compiled & checked | {self._n_checked} |")
            lines.append(f"| Passed  | {self._n_passed} |")
            lines.append(f"| Failed  | {self._n_failed} |")
        else:
            lines.append("*(Verification not run — use --skip-verification to suppress)*")
        lines.append("")

        # ── 7. Convergence note ───────────────────────────────────────
        if self._convergence and self._strategy in ("bayesian", "sha"):
            lines.append("## Convergence\n")
            n = self._convergence.get("n_evals", "—")
            best_ms = self._convergence.get("best_ms", "—")
            lines.append(f"Strategy **{self._strategy}** reached best config "
                         f"(`{best_ms:.3f}ms`) after **{n}** evaluations.\n")

        # ── 8. Recommended config ──────────────────────────────────────
        lines.append("## Recommended Configuration\n")
        if self._results:
            best = self._results[0]
            lines.append(f"**Kernel:** `{self._kernel}`\n")
            lines.append("**Parameters:**\n")
            for k, v in best.params.items():
                lines.append(f"  - `{k}` = `{v}`")
            lines.append("")
            lines.append(f"**Mean latency:** {best.mean_ms:.3f}ms "
                         f"±{best.ci_95_ms:.3f}ms (95% CI, n={best.n_samples})")
            if best.speedup:
                lines.append(f"\n**Speedup:** {best.speedup:.2f}x over baseline "
                             f"({'statistically significant' if best.is_significant else 'not significant'})")
            lines.append("")
        else:
            lines.append("*(No successful variants)*")

        # ── Write ──────────────────────────────────────────────────────
        text = "\n".join(lines) + "\n"
        self._report_path.write_text(text, encoding="utf-8")
        print(f"[REPORT] Markdown report → {self._report_path}")
        return self._report_path

    @staticmethod
    def _short_params(params: dict) -> str:
        """Compact one-line param string for table cells."""
        parts = []
        for k, v in sorted(params.items()):
            short_k = k.replace("block_size", "blk").replace("transpose_b", "T").replace(
                "warp_shuffle", "wshfl").replace("tile_x", "tx").replace("tile_y", "ty")
            parts.append(f"{short_k}={v}")
        return " ".join(parts)[:55]


# ── Terminal summary table (stdlib only) ───────────────────────────────────

def print_terminal_summary(
    kernel: str,
    results: list[BenchmarkResult],
    baseline_ms: Optional[float],
    n_total: int,
    strategy: str = "grid",
) -> None:
    """
    Print a clean terminal summary after tuning completes.

    Uses only stdlib string formatting (no rich / tabulate).

    Args:
        kernel:      Kernel name.
        results:     Benchmark results sorted best-first.
        baseline_ms: Baseline latency for speedup computation.
        n_total:     Total variants evaluated.
        strategy:    Search strategy name.
    """
    W = 74
    print("\n" + "=" * W)
    print(f"  TUNING COMPLETE — {kernel.upper()}  |  strategy={strategy}  "
          f"|  {n_total} variants")
    print("=" * W)

    # Top results
    top = results[:10]
    if not top:
        print("  No successful results.")
        print("=" * W)
        return

    header = (
        f"  {'#':<3}  {'mean_ms':>8}  {'±CI':>8}  "
        f"{'speedup':>8}  {'p-val':>7}  {'sig':>4}  params"
    )
    print(header)
    print("  " + "-" * (W - 2))

    for rank, r in enumerate(top, 1):
        speedup_str = f"{r.speedup:.2f}x" if r.speedup else "  —  "
        pval_str    = f"{r.p_value:.3f}" if r.p_value is not None else "  —  "
        sig_str     = "✓" if r.is_significant else "✗"
        params_str  = " ".join(
            f"{k}={v}" for k, v in sorted(r.params.items())
        )[:40]
        print(
            f"  {rank:<3}  {r.mean_ms:>7.3f}ms  "
            f"±{r.ci_95_ms:>6.3f}ms  "
            f"{speedup_str:>8}  "
            f"{pval_str:>7}  "
            f"{sig_str:>4}  "
            f"{params_str}"
        )

    print("=" * W)

    best = results[0]
    if baseline_ms:
        speedup = baseline_ms / best.mean_ms
        sig_tag = " (statistically significant)" if best.is_significant else " (not significant)"
        print(f"\n  BEST: {best.mean_ms:.3f}ms  |  baseline: {baseline_ms:.3f}ms  "
              f"|  speedup: {speedup:.2f}x{sig_tag}")

    # Roofline line if present
    if best.bound_type:
        print(f"  BOUND: {best.bound_type}-bound  "
              f"| efficiency: {best.roofline_efficiency_pct:.1f}%  "
              f"| AI: {best.arithmetic_intensity:.2f} FLOP/byte")

    # Occupancy line if present
    if best.occupancy is not None:
        spill = "  SPILL!" if best.has_register_spill else ""
        print(f"  OCCUPANCY: {best.occupancy*100:.1f}%  "
              f"| regs: {best.registers_per_thread}  "
              f"| smem: {best.shared_mem_bytes}B{spill}")

    print("=" * W + "\n")


def print_final_summary(all_results: dict[str, list[BenchmarkResult]],
                        baselines: dict[str, dict]) -> None:
    """
    Print a multi-kernel summary after tuning all requested kernels.

    Args:
        all_results: {kernel: sorted_results_list}
        baselines:   {tag: {"mean_ms": float, ...}}
    """
    W = 60
    print("\n" + "=" * W)
    print("  FULL TUNING SUMMARY")
    print("=" * W)
    print(f"  {'Kernel':<14} {'Baseline':>10} {'Best':>10} {'Speedup':>10}")
    print("  " + "-" * (W - 2))
    for kernel, results in all_results.items():
        if not results:
            continue
        base_ms = next((v["mean_ms"] for tag, v in baselines.items()
                        if tag.startswith(kernel)), None)
        best_ms = results[0].mean_ms
        speedup = f"{base_ms / best_ms:.2f}x" if base_ms else "—"
        base_str = f"{base_ms:.3f}ms" if base_ms else "—"
        print(f"  {kernel:<14} {base_str:>10} {best_ms:>8.3f}ms {speedup:>10}")
    print("=" * W + "\n")

"""
ptx_analysis.py — PTX and SASS static analysis for CUDA kernel variants.

Workflow:
  1. Compile variant .cu → PTX  (nvcc -ptx)
  2. Extract SASS from binary    (cuobjdump --dump-sass, skipped gracefully if missing)
  3. Parse PTX for instruction-level metrics
  4. Compare baseline vs best variant, save to results/{kernel}_ptx_analysis.json

PTX metrics extracted per kernel function:
  total_instructions  — total instruction count
  ld_global           — ld.global (DRAM load) count
  st_global           — st.global (DRAM store) count
  fma_count           — fma / mad instruction count
  branch_count        — bra / brx instruction count
  compute_ratio       — fma_count / total_instructions
  memory_ratio        — (ld+st) / total_instructions

Enabled via:  python autotune.py --ptx-analysis
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# ── Instruction patterns (PTX) ─────────────────────────────────────────────

_RE_FUNC     = re.compile(r"\.visible\s+\.entry\s+(\w+)|\.func\s+(\w+)")
_RE_LD_GLOB  = re.compile(r"\bld\.global\b")
_RE_ST_GLOB  = re.compile(r"\bst\.global\b")
_RE_FMA      = re.compile(r"\b(?:fma|mad)\.(?:rn|rz|rm|rp)?\b")
_RE_BRANCH   = re.compile(r"\b(?:bra|brx|call|ret)\b")
_RE_INSTR    = re.compile(r"^\s+[a-z]", re.MULTILINE)  # any instruction line


@dataclass
class PTXMetrics:
    """Static instruction-level metrics for one kernel function."""
    kernel_function: str
    total_instructions: int
    ld_global: int
    st_global: int
    fma_count: int
    branch_count: int
    compute_ratio: float    # fma / total
    memory_ratio: float     # (ld+st) / total


@dataclass
class PTXComparison:
    """Delta between baseline and optimised variant PTX metrics."""
    kernel: str
    baseline: PTXMetrics
    optimised: PTXMetrics
    instruction_delta: int      # optimised - baseline (negative = fewer = better)
    memory_op_delta: int
    compute_ratio_delta: float


# ── PTX analysis ───────────────────────────────────────────────────────────

def _parse_ptx_file(ptx_text: str) -> dict[str, PTXMetrics]:
    """
    Parse a PTX file and return {function_name: PTXMetrics}.

    Splits on .entry / .func boundaries and analyses each section.
    """
    # Split into per-function sections
    sections: list[tuple[str, str]] = []   # (name, body)
    parts = re.split(r"(\.visible\s+\.entry\s+\w+|\.func\s+\w+)", ptx_text)
    i = 0
    while i < len(parts):
        m = re.match(r"\.(?:visible\s+)?(?:entry|func)\s+(\w+)", parts[i].strip())
        if m and i + 1 < len(parts):
            sections.append((m.group(1), parts[i] + parts[i + 1]))
            i += 2
        else:
            i += 1

    metrics: dict[str, PTXMetrics] = {}
    for name, body in sections:
        total = len(_RE_INSTR.findall(body))
        ld    = len(_RE_LD_GLOB.findall(body))
        st    = len(_RE_ST_GLOB.findall(body))
        fma   = len(_RE_FMA.findall(body))
        br    = len(_RE_BRANCH.findall(body))
        comp_ratio = fma / total if total > 0 else 0.0
        mem_ratio  = (ld + st) / total if total > 0 else 0.0
        metrics[name] = PTXMetrics(
            kernel_function=name,
            total_instructions=total,
            ld_global=ld,
            st_global=st,
            fma_count=fma,
            branch_count=br,
            compute_ratio=round(comp_ratio, 4),
            memory_ratio=round(mem_ratio, 4),
        )
    return metrics


# ── SASS extraction ────────────────────────────────────────────────────────

def _dump_sass(binary: Path) -> Optional[str]:
    """
    Run cuobjdump --dump-sass and return the output text.

    Returns None if cuobjdump is not on PATH (skip gracefully).
    """
    if not shutil.which("cuobjdump"):
        return None
    try:
        result = subprocess.run(
            ["cuobjdump", "--dump-sass", str(binary)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# ── Main analyzer class ────────────────────────────────────────────────────

class PTXAnalyzer:
    """
    Generate PTX, extract SASS, and compute instruction-level metrics.

    Enabled when the user passes --ptx-analysis.  Disabled by default
    because it adds one extra nvcc compilation per variant.

    Usage::

        pa = PTXAnalyzer()
        baseline_m = pa.analyze_source(baseline_src)
        best_m     = pa.analyze_source(best_src)
        cmp        = pa.compare(kernel, baseline_m, best_m)
        pa.save(kernel, cmp)
    """

    def __init__(
        self,
        arch: str = "sm_75",
        results_dir: Path = RESULTS_DIR,
    ) -> None:
        """
        Args:
            arch:        Compilation target for PTX generation.
            results_dir: Output directory for JSON reports.
        """
        self._arch        = arch
        self._results_dir = results_dir

    def analyze_source(
        self,
        src_path: Path,
        binary_path: Optional[Path] = None,
    ) -> dict[str, PTXMetrics]:
        """
        Compile *src_path* to PTX and return per-function metrics.

        Optionally also runs cuobjdump on *binary_path* for SASS (if
        cuobjdump is available on PATH).

        Args:
            src_path:    .cu source file to analyse.
            binary_path: Pre-compiled binary for SASS extraction (optional).

        Returns:
            {function_name: PTXMetrics}
        """
        ptx_path = src_path.with_suffix(".ptx")
        cmd = [
            "nvcc", "-O3", f"-arch={self._arch}",
            "--use_fast_math",
            "-ptx", "-o", str(ptx_path),
            str(src_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            warnings.warn(f"PTX compilation failed for {src_path.name}: {result.stderr[:200]}")
            return {}

        try:
            ptx_text = ptx_path.read_text(encoding="utf-8")
        finally:
            try:
                ptx_path.unlink()
            except FileNotFoundError:
                pass

        metrics = _parse_ptx_file(ptx_text)

        if binary_path is not None:
            sass = _dump_sass(binary_path)
            if sass:
                # Attach raw SASS summary (line count as proxy for complexity)
                for name, m in metrics.items():
                    sass_lines = sass.count("\n")
                    # We could parse SASS more deeply; for now attach count
                    # to a note field if we extended PTXMetrics
                    pass   # SASS integration reserved for future extension

        return metrics

    def compare(
        self,
        kernel: str,
        baseline: dict[str, PTXMetrics],
        optimised: dict[str, PTXMetrics],
    ) -> Optional[PTXComparison]:
        """
        Build a PTXComparison between baseline and optimised metrics.

        Matches the first __global__ kernel function in each dict.

        Args:
            kernel:    Kernel name (for labelling).
            baseline:  Metrics from baseline source.
            optimised: Metrics from best optimised source.
        """
        if not baseline or not optimised:
            return None

        # Pick the first (and usually only) non-device function
        def _pick(m: dict[str, PTXMetrics]) -> PTXMetrics:
            for k, v in m.items():
                if "opt" in k or "naive" in k or kernel in k:
                    return v
            return next(iter(m.values()))

        b = _pick(baseline)
        o = _pick(optimised)

        return PTXComparison(
            kernel=kernel,
            baseline=b,
            optimised=o,
            instruction_delta=o.total_instructions - b.total_instructions,
            memory_op_delta=(o.ld_global + o.st_global) - (b.ld_global + b.st_global),
            compute_ratio_delta=round(o.compute_ratio - b.compute_ratio, 4),
        )

    def save(self, kernel: str, comparison: Optional[PTXComparison]) -> None:
        """
        Save PTX comparison to results/{kernel}_ptx_analysis.json.

        Args:
            kernel:     Kernel name (determines output filename).
            comparison: Result of compare(), or None if analysis failed.
        """
        path = self._results_dir / f"{kernel}_ptx_analysis.json"
        if comparison is None:
            data = {"kernel": kernel, "status": "analysis_failed"}
        else:
            data = {
                "kernel":             comparison.kernel,
                "instruction_delta":  comparison.instruction_delta,
                "memory_op_delta":    comparison.memory_op_delta,
                "compute_ratio_delta": comparison.compute_ratio_delta,
                "baseline":           asdict(comparison.baseline),
                "optimised":          asdict(comparison.optimised),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[PTX] Analysis saved → {path}")

    def print_comparison(self, comparison: Optional[PTXComparison]) -> None:
        """Print a human-readable PTX delta table."""
        if comparison is None:
            print("[PTX] No comparison available.")
            return
        b, o = comparison.baseline, comparison.optimised
        print(f"\n{'='*60}")
        print(f"  PTX Analysis — {comparison.kernel}")
        print(f"{'='*60}")
        print(f"  {'Metric':<24} {'Baseline':>10} {'Optimised':>10} {'Delta':>8}")
        print(f"  {'-'*54}")
        def row(label, bv, ov):
            delta = ov - bv
            sign  = "+" if delta > 0 else ""
            print(f"  {label:<24} {bv:>10} {ov:>10} {sign}{delta:>7}")
        row("Total instructions",  b.total_instructions, o.total_instructions)
        row("ld.global",           b.ld_global,          o.ld_global)
        row("st.global",           b.st_global,          o.st_global)
        row("fma/mad",             b.fma_count,          o.fma_count)
        row("branches",            b.branch_count,       o.branch_count)
        print(f"  {'compute_ratio':<24} {b.compute_ratio:>10.3f} {o.compute_ratio:>10.3f}"
              f" {comparison.compute_ratio_delta:>+8.3f}")
        print("=" * 60 + "\n")

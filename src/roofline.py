"""
roofline.py — Roofline model integration for RTX 2070 (sm_75).

The Roofline model (Williams et al., 2009) characterises whether a kernel is
memory-bandwidth-bound or compute-bound by comparing its arithmetic intensity
(FLOP/byte) against the hardware ridge point.

RTX 2070 constants:
  peak_tflops  = 7.5  TFLOP/s  (FP32)
  mem_bw_gbps  = 448  GB/s     (GDDR6 peak)
  ridge_point  = peak_tflops * 1e12 / (mem_bw_gbps * 1e9)  ≈ 16.74 FLOP/byte

This module exposes two APIs:

  1. RooflineAnalyzer (class) — primary API used by reporter, visualization,
     streamlit_dashboard, search.
  2. compute_roofline / format_table (legacy functions) — used by autotune,
     plots and benchmark.  Supports an optional ncu_metrics dict to supply
     measured DRAM bytes (dram__bytes.sum) for measured-AI roofline points.

Reference:
  Williams, S., Waterman, A., & Patterson, D. (2009).
  Roofline: An insightful visual performance model for multicore architectures.
  Communications of the ACM, 52(4), 65–76.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# ── Hardware constants (RTX 2070) ──────────────────────────────────────────

PEAK_TFLOPS    = 7.5          # FP32 tensor TFLOP/s
MEM_BW_GBPS    = 448.0        # peak DRAM bandwidth GB/s
RIDGE_POINT    = PEAK_TFLOPS * 1e3 / MEM_BW_GBPS   # FLOP/byte  ≈ 16.74

# Hardware descriptor used by legacy callers (plots.py, benchmark.py).
RTX2070 = {
    "name":            "NVIDIA GeForce RTX 2070",
    "arch":            "sm_75",
    "cores":           2304,
    "boost_clock_ghz": 1.62,
    "peak_fp32_tflops": PEAK_TFLOPS,
    "peak_bw_gbs":     MEM_BW_GBPS,
}

BoundType = Literal["memory", "compute"]

# ── Per-kernel analytical FLOP and byte models ─────────────────────────────
# All sizes are element counts (not bytes); element size = sizeof(float) = 4.

def _matmul_flops(N: int) -> int:
    return 2 * N * N * N


def _matmul_bytes(N: int) -> int:
    return 3 * N * N * 4


def _softmax_flops(R: int, C: int) -> int:
    return 5 * R * C


def _softmax_bytes(R: int, C: int) -> int:
    return 2 * R * C * 4


def _reduction_flops(N: int) -> int:
    return 2 * N


def _reduction_bytes(N: int) -> int:
    return N * 4


def _layernorm_flops(R: int, C: int) -> int:
    return 8 * R * C


def _layernorm_bytes(R: int, C: int) -> int:
    return 3 * R * C * 4


def _attention_flops(S: int, D: int) -> int:
    return 4 * S * S * D + 5 * S * S


def _attention_bytes_flash(S: int, D: int) -> int:
    return 4 * S * D * 4


def _attention_bytes_naive(S: int, D: int) -> int:
    return (3 * S * D + 2 * S * S + S * D) * 4


# ── Default problem dimensions per kernel ──────────────────────────────────

KERNEL_DIMS: dict[str, tuple] = {
    "matmul":    (1024,),
    "softmax":   (1024, 4096),
    "reduction": (1 << 20,),
    "layernorm": (512, 2048),
    "attention": (512, 64),
}


def get_dims(kernel: str) -> tuple:
    """Return the canonical problem dimensions for a kernel."""
    base = kernel.split("_")[0]
    return KERNEL_DIMS.get(base, KERNEL_DIMS.get(kernel, (1024,)))


@dataclass
class RooflinePoint:
    """A single kernel's position on the roofline chart."""
    kernel: str
    dims: tuple
    arithmetic_intensity: float
    achieved_gflops: float
    peak_gflops: float
    bound_type: BoundType
    efficiency_pct: float


class RooflineAnalyzer:
    """
    Primary roofline API.  Compute roofline metrics for the auto-tuned kernels
    on RTX 2070.

        ra = RooflineAnalyzer()
        ai = ra.arithmetic_intensity("matmul", (1024,))
        bt = ra.bound_type("matmul", (1024,))
        pt = ra.analyze("matmul", (1024,), elapsed_ms=4.8)
    """

    peak_tflops: float = PEAK_TFLOPS
    mem_bw_gbps: float = MEM_BW_GBPS
    ridge_point: float = RIDGE_POINT

    # ── FLOP / byte analytics ──────────────────────────────────────────────

    def total_flops(self, kernel: str, dims: tuple) -> int:
        k = kernel.split("_")[0]
        if k == "matmul":    return _matmul_flops(*dims)
        if k == "softmax":   return _softmax_flops(*dims)
        if k == "reduction": return _reduction_flops(*dims)
        if k == "layernorm": return _layernorm_flops(*dims)
        if k == "attention": return _attention_flops(*dims)
        raise ValueError(f"Unknown kernel: {kernel!r}")

    def total_bytes(self, kernel: str, dims: tuple) -> int:
        k = kernel.split("_")[0]
        if k == "matmul":    return _matmul_bytes(*dims)
        if k == "softmax":   return _softmax_bytes(*dims)
        if k == "reduction": return _reduction_bytes(*dims)
        if k == "layernorm": return _layernorm_bytes(*dims)
        if k == "attention": return _attention_bytes_flash(*dims)
        raise ValueError(f"Unknown kernel: {kernel!r}")

    def arithmetic_intensity(self, kernel: str, dims: tuple) -> float:
        return self.total_flops(kernel, dims) / self.total_bytes(kernel, dims)

    def achieved_gflops(self, kernel: str, dims: tuple, elapsed_ms: float) -> float:
        if elapsed_ms <= 0:
            return 0.0
        return self.total_flops(kernel, dims) / (elapsed_ms * 1e6)

    def bound_type(self, kernel: str, dims: tuple) -> BoundType:
        return "memory" if self.arithmetic_intensity(kernel, dims) < self.ridge_point else "compute"

    def peak_performance(self, kernel: str, dims: tuple) -> float:
        ai = self.arithmetic_intensity(kernel, dims)
        if ai < self.ridge_point:
            return ai * self.mem_bw_gbps
        return self.peak_tflops * 1000.0

    def efficiency_pct(self, kernel: str, dims: tuple, elapsed_ms: float) -> float:
        achieved = self.achieved_gflops(kernel, dims, elapsed_ms)
        ceiling  = self.peak_performance(kernel, dims)
        if ceiling <= 0:
            return 0.0
        return min(achieved / ceiling * 100.0, 100.0)

    def analyze(self, kernel: str, dims: tuple, elapsed_ms: float) -> RooflinePoint:
        return RooflinePoint(
            kernel=kernel,
            dims=dims,
            arithmetic_intensity=self.arithmetic_intensity(kernel, dims),
            achieved_gflops=self.achieved_gflops(kernel, dims, elapsed_ms),
            peak_gflops=self.peak_performance(kernel, dims),
            bound_type=self.bound_type(kernel, dims),
            efficiency_pct=self.efficiency_pct(kernel, dims, elapsed_ms),
        )

    def print_summary_table(
        self,
        entries: list[tuple[str, tuple, float]],
        title: str = "Roofline Summary",
    ) -> None:
        print(f"\n{'='*74}")
        print(f"  {title}  |  RTX 2070  |  ridge={self.ridge_point:.1f} FLOP/byte")
        print(f"{'='*74}")
        print(f"{'Kernel':<22} {'AI':>8} {'Bound':>8} "
              f"{'Achieved':>10} {'Ceiling':>10} {'Eff%':>6}")
        print("-" * 74)
        for kernel, dims, elapsed_ms in entries:
            pt = self.analyze(kernel, dims, elapsed_ms)
            print(
                f"{kernel:<22} "
                f"{pt.arithmetic_intensity:>7.2f}  "
                f"{pt.bound_type:>8}  "
                f"{pt.achieved_gflops:>8.1f}G  "
                f"{pt.peak_gflops:>8.1f}G  "
                f"{pt.efficiency_pct:>5.1f}%"
            )
        print("=" * 74 + "\n")


# ── Legacy API (compute_roofline, RooflineResult, format_table) ────────────
# Kept for autotune.py, plots.py, benchmark.py.  Adds ncu-aware measured-bytes
# support that the class API doesn't expose.

KERNEL_COUNTS = {
    "matmul":    {"flops": _matmul_flops(1024), "bytes": _matmul_bytes(1024), "shape": "1024x1024"},
    "softmax":   {"flops": _softmax_flops(1024, 4096), "bytes": _softmax_bytes(1024, 4096), "shape": "1024x4096"},
    "reduction": {"flops": _reduction_flops(1 << 20), "bytes": _reduction_bytes(1 << 20), "shape": f"N={1<<20}"},
    "layernorm": {"flops": _layernorm_flops(512, 2048), "bytes": _layernorm_bytes(512, 2048), "shape": "512x2048"},
    "attention": {"flops": _attention_flops(512, 64), "bytes": _attention_bytes_flash(512, 64), "shape": "S=512 D=64"},
}


@dataclass
class RooflineResult:
    kernel:           str
    shape:            str
    mean_ms:          float
    flops_analytic:   float
    bytes_analytic:   float
    ai_analytic:      float
    gflops_analytic:  float
    bw_analytic_gbs:  float
    bound_gflops:     float
    pct_of_roof:      float
    bytes_measured:   Optional[float] = None
    bw_measured_gbs:  Optional[float] = None
    ai_measured:      Optional[float] = None


def compute_roofline(kernel: str, mean_ms: float,
                     ncu_metrics: Optional[dict] = None) -> RooflineResult:
    """Compute roofline stats for a kernel at a given measured runtime."""
    counts = KERNEL_COUNTS[kernel]
    flops  = counts["flops"]
    bytes_ = counts["bytes"]

    time_s = mean_ms * 1e-3
    gflops = flops  / time_s / 1e9
    bw_gbs = bytes_ / time_s / 1e9
    ai     = flops  / bytes_

    peak_flops_gflops = PEAK_TFLOPS * 1000
    bound = min(peak_flops_gflops, ai * MEM_BW_GBPS)
    pct   = 100.0 * gflops / bound if bound > 0 else 0.0

    bytes_meas = bw_meas = ai_meas = None
    if ncu_metrics:
        dram = ncu_metrics.get("dram__bytes.sum")
        if dram is not None and dram > 0:
            bytes_meas = float(dram)
            bw_meas    = bytes_meas / time_s / 1e9
            ai_meas    = flops / bytes_meas

    return RooflineResult(
        kernel=kernel, shape=counts["shape"], mean_ms=mean_ms,
        flops_analytic=flops, bytes_analytic=bytes_, ai_analytic=ai,
        gflops_analytic=gflops, bw_analytic_gbs=bw_gbs,
        bound_gflops=bound, pct_of_roof=pct,
        bytes_measured=bytes_meas, bw_measured_gbs=bw_meas, ai_measured=ai_meas,
    )


def format_table(results: list[RooflineResult]) -> str:
    lines = []
    lines.append(f"{'='*92}")
    lines.append(f"Roofline summary — {RTX2070['name']} "
                 f"(peak {RTX2070['peak_fp32_tflops']:.2f} TFLOPS / "
                 f"{RTX2070['peak_bw_gbs']:.0f} GB/s)")
    lines.append(f"{'-'*92}")
    lines.append(f"{'Kernel':<10} {'Shape':<14} {'ms':>8} "
                 f"{'GFLOPS':>9} {'BW':>8} {'AI':>7} {'% roof':>8}")
    lines.append(f"{'-'*92}")
    for r in results:
        lines.append(
            f"{r.kernel:<10} {r.shape:<14} {r.mean_ms:>8.3f} "
            f"{r.gflops_analytic:>9.1f} {r.bw_analytic_gbs:>8.1f} "
            f"{r.ai_analytic:>7.2f} {r.pct_of_roof:>7.1f}%"
        )
        if r.bw_measured_gbs is not None:
            lines.append(
                f"{'  (ncu)':<10} {'':<14} {'':<8} "
                f"{'':<9} {r.bw_measured_gbs:>8.1f} "
                f"{r.ai_measured:>7.2f}"
            )
    lines.append(f"{'='*92}")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    ra = RooflineAnalyzer()
    print(f"RTX 2070 ridge point: {ra.ridge_point:.2f} FLOP/byte\n")

    if len(sys.argv) >= 2:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path(__file__).parent.parent / "results"

    rr: list[RooflineResult] = []
    for kernel in ("matmul", "softmax", "reduction", "layernorm", "attention"):
        f = results_dir / f"{kernel}_tuning.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        best = data.get("best")
        if not best:
            continue
        ncu = data.get("ncu_metrics") or {}
        rr.append(compute_roofline(kernel, best["mean_ms"], ncu))

    if rr:
        print(format_table(rr))
    else:
        sample = [
            ("matmul",    (1024,),        4.84),
            ("softmax",   (1024, 4096),   1.44),
            ("reduction", (1 << 20,),     0.089),
            ("layernorm", (512, 2048),    0.379),
        ]
        ra.print_summary_table(sample, "Baseline Roofline (sample)")

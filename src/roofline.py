"""
roofline.py — Roofline model integration for RTX 2070 (sm_75).

The Roofline model (Williams et al., 2009) characterises whether a kernel is
memory-bandwidth-bound or compute-bound by comparing its arithmetic intensity
(FLOP/byte) against the hardware ridge point.

RTX 2070 constants:
  peak_tflops  = 7.5  TFLOP/s  (FP32)
  mem_bw_gbps  = 448  GB/s     (GDDR6 peak)
  ridge_point  = peak_tflops * 1e12 / (mem_bw_gbps * 1e9)  ≈ 16.74 FLOP/byte

Reference:
  Williams, S., Waterman, A., & Patterson, D. (2009).
  Roofline: An insightful visual performance model for multicore architectures.
  Communications of the ACM, 52(4), 65–76.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ── Hardware constants (RTX 2070) ──────────────────────────────────────────

PEAK_TFLOPS    = 7.5          # FP32 tensor TFLOP/s
MEM_BW_GBPS    = 448.0        # peak DRAM bandwidth GB/s
RIDGE_POINT    = PEAK_TFLOPS * 1e3 / MEM_BW_GBPS   # FLOP/byte  ≈ 16.74

BoundType = Literal["memory", "compute"]

# ── Per-kernel analytical FLOP and byte models ─────────────────────────────
# All sizes are element counts (not bytes); element size = sizeof(float) = 4.

def _matmul_flops(N: int) -> int:
    """2*N^3 FLOPs for N×N FP32 matmul (N multiply-adds)."""
    return 2 * N * N * N


def _matmul_bytes(N: int) -> int:
    """3*N*N*4 bytes: load A (N²), load B (N²), store C (N²)."""
    return 3 * N * N * 4


def _softmax_flops(R: int, C: int) -> int:
    """5*R*C FLOPs: max(1*RC) + exp+sub(2*RC) + sum(1*RC) + div(1*RC)."""
    return 5 * R * C


def _softmax_bytes(R: int, C: int) -> int:
    """2*R*C*4 bytes: read input once, write output once."""
    return 2 * R * C * 4


def _reduction_flops(N: int) -> int:
    """2*N FLOPs: N adds in tree reduction (binary tree ≈ N-1 ≈ N adds)."""
    return 2 * N


def _reduction_bytes(N: int) -> int:
    """N*4 bytes: read input once."""
    return N * 4


def _layernorm_flops(R: int, C: int) -> int:
    """8*R*C FLOPs: mean(1*RC), var(2*RC), inv_std(0≈0), normalize+scale(3*RC),
    bias add(1*RC), rsqrt(1*RC per row ≈ 0 vs RC)."""
    return 8 * R * C


def _layernorm_bytes(R: int, C: int) -> int:
    """3*R*C*4 bytes: read input, read gamma+beta (per-col, ≈ C), write output."""
    return 3 * R * C * 4


def _attention_flops(S: int, D: int) -> int:
    """4*S*S*D + 5*S*S FLOPs: QK dot products, softmax ops, AV accumulation."""
    return 4 * S * S * D + 5 * S * S


def _attention_bytes_flash(S: int, D: int) -> int:
    """4*S*D*4 bytes (flash variant): Q,K,V,O each S*D floats."""
    return 4 * S * D * 4


def _attention_bytes_naive(S: int, D: int) -> int:
    """(3*S*D + 2*S*S + S*D)*4 bytes (naive): Q,K,V reads, S matrix write/read, O write."""
    return (3 * S * D + 2 * S * S + S * D) * 4


# ── Dimensions helper ──────────────────────────────────────────────────────

_KERNEL_DEFAULTS: dict[str, tuple] = {
    "matmul":    (1024,),
    "softmax":   (1024, 4096),
    "reduction": (1 << 20,),
    "layernorm": (512, 2048),
    "attention": (512, 64),
}


@dataclass
class RooflinePoint:
    """A single kernel's position on the roofline chart."""
    kernel: str
    dims: tuple
    arithmetic_intensity: float   # FLOP / byte
    achieved_gflops: float        # measured
    peak_gflops: float            # applicable roof (memory or compute)
    bound_type: BoundType
    efficiency_pct: float         # achieved / peak * 100


class RooflineAnalyzer:
    """
    Compute roofline metrics for the four auto-tuned kernels on RTX 2070.

    Usage::

        ra = RooflineAnalyzer()
        ai = ra.arithmetic_intensity("matmul", (1024,))   # → float FLOP/byte
        bt = ra.bound_type("matmul", (1024,))             # → "memory" | "compute"
        pt = ra.analyze("matmul", (1024,), elapsed_ms=4.8)
    """

    peak_tflops: float = PEAK_TFLOPS
    mem_bw_gbps: float = MEM_BW_GBPS
    ridge_point:  float = RIDGE_POINT

    # ── FLOP / byte analytics ──────────────────────────────────────────────

    def total_flops(self, kernel: str, dims: tuple) -> int:
        """Total floating-point operations for kernel with given dims."""
        k = kernel.split("_")[0]   # strip suffix like "_1024"
        if k == "matmul":
            return _matmul_flops(*dims)
        if k == "softmax":
            return _softmax_flops(*dims)
        if k == "reduction":
            return _reduction_flops(*dims)
        if k == "layernorm":
            return _layernorm_flops(*dims)
        if k == "attention":
            return _attention_flops(*dims)
        raise ValueError(f"Unknown kernel: {kernel!r}")

    def total_bytes(self, kernel: str, dims: tuple) -> int:
        """Total bytes transferred between DRAM and SM for the kernel."""
        k = kernel.split("_")[0]
        if k == "matmul":
            return _matmul_bytes(*dims)
        if k == "softmax":
            return _softmax_bytes(*dims)
        if k == "reduction":
            return _reduction_bytes(*dims)
        if k == "layernorm":
            return _layernorm_bytes(*dims)
        if k == "attention":
            return _attention_bytes_flash(*dims)
        raise ValueError(f"Unknown kernel: {kernel!r}")

    def arithmetic_intensity(self, kernel: str, dims: tuple) -> float:
        """
        Arithmetic intensity in FLOP/byte.

        AI < ridge_point → memory-bound; AI > ridge_point → compute-bound.
        """
        return self.total_flops(kernel, dims) / self.total_bytes(kernel, dims)

    # ── Performance metrics ────────────────────────────────────────────────

    def achieved_gflops(self, kernel: str, dims: tuple, elapsed_ms: float) -> float:
        """
        Achieved compute throughput in GFLOP/s.

        Args:
            kernel:     Kernel name (matmul | softmax | reduction | layernorm).
            dims:       Problem dimensions tuple, e.g. (1024,) for N=1024 matmul.
            elapsed_ms: Measured kernel time in milliseconds.
        """
        if elapsed_ms <= 0:
            return 0.0
        flops = self.total_flops(kernel, dims)
        return flops / (elapsed_ms * 1e6)   # GFLOP/s

    def bound_type(self, kernel: str, dims: tuple) -> BoundType:
        """
        Classify whether the kernel is memory-bound or compute-bound.

        Returns "memory" if arithmetic_intensity < ridge_point, else "compute".
        """
        return "memory" if self.arithmetic_intensity(kernel, dims) < self.ridge_point else "compute"

    def peak_performance(self, kernel: str, dims: tuple) -> float:
        """
        Applicable roofline ceiling in GFLOP/s.

        For memory-bound kernels: attainable = AI * mem_bw (GFLOP/s).
        For compute-bound kernels: attainable = peak_tflops * 1000 (GFLOP/s).
        """
        ai = self.arithmetic_intensity(kernel, dims)
        if ai < self.ridge_point:
            return ai * self.mem_bw_gbps   # memory-bound roof (GFLOP/s)
        return self.peak_tflops * 1000.0   # compute-bound roof (GFLOP/s)

    def efficiency_pct(self, kernel: str, dims: tuple, elapsed_ms: float) -> float:
        """
        Roofline efficiency as a percentage of the applicable ceiling.

        100% = perfectly hitting the memory-bandwidth or compute roof.
        """
        achieved = self.achieved_gflops(kernel, dims, elapsed_ms)
        ceiling  = self.peak_performance(kernel, dims)
        if ceiling <= 0:
            return 0.0
        return min(achieved / ceiling * 100.0, 100.0)

    def analyze(self, kernel: str, dims: tuple, elapsed_ms: float) -> RooflinePoint:
        """
        Full roofline analysis for one timing measurement.

        Args:
            kernel:     Kernel name.
            dims:       Problem size tuple.
            elapsed_ms: Measured kernel latency in ms.

        Returns:
            RooflinePoint with all metrics populated.
        """
        ai       = self.arithmetic_intensity(kernel, dims)
        achieved = self.achieved_gflops(kernel, dims, elapsed_ms)
        ceiling  = self.peak_performance(kernel, dims)
        bound    = self.bound_type(kernel, dims)
        eff      = self.efficiency_pct(kernel, dims, elapsed_ms)
        return RooflinePoint(
            kernel=kernel,
            dims=dims,
            arithmetic_intensity=ai,
            achieved_gflops=achieved,
            peak_gflops=ceiling,
            bound_type=bound,
            efficiency_pct=eff,
        )

    # ── Reporting ──────────────────────────────────────────────────────────

    def print_summary_table(
        self,
        entries: list[tuple[str, tuple, float]],   # (kernel, dims, elapsed_ms)
        title: str = "Roofline Summary",
    ) -> None:
        """
        Print a roofline summary table to stdout.

        Args:
            entries: List of (kernel_name, dims, elapsed_ms) tuples.
            title:   Table title.
        """
        print(f"\n{'='*74}")
        print(f"  {title}  |  RTX 2070  |  ridge={self.ridge_point:.1f} FLOP/byte")
        print(f"{'='*74}")
        hdr = (f"{'Kernel':<22} {'AI':>8} {'Bound':>8} "
               f"{'Achieved':>10} {'Ceiling':>10} {'Eff%':>6}")
        print(hdr)
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


# ── Default dimensions per kernel ──────────────────────────────────────────

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


if __name__ == "__main__":
    ra = RooflineAnalyzer()
    print(f"RTX 2070 ridge point: {ra.ridge_point:.2f} FLOP/byte\n")
    sample = [
        ("matmul",    (1024,),        4.84),
        ("softmax",   (1024, 4096),   1.44),
        ("reduction", (1 << 20,),     0.089),
        ("layernorm", (512, 2048),    0.379),
    ]
    ra.print_summary_table(sample, "Baseline Roofline Analysis")

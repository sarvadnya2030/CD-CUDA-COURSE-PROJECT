"""
benchmark.py — compile, run, and profile CUDA kernels with statistical rigor.

Wraps nvcc compilation and CUDA event timing. Provides the BenchmarkResult
dataclass consumed by all other analysis modules.

Statistical measurement protocol:
  - 5 warmup runs (GPU thermal stabilization)
  - 30 timed runs via CUDA events (per-sample SAMPLE lines in binary output)
  - mean, std, 95% CI (1.96 * std / sqrt(n))
  - Welch t-test vs baseline (scipy.stats.ttest_ind, equal_var=False)
  - A variant is "winning" only if speedup is statistically significant (p < 0.05)

Phase A-D additions:
  - collect_ncu_metrics() supports kernel_regex / launch_count / env_overrides
    so the autotune driver can profile only the optimized kernel of the best
    binary instead of every launch.
  - parse_timing_output() also returns a flat per-tag dict (mean_ms/min_ms/...)
    via parse_timing_output_simple() for legacy callers.
"""

import math
import os
import subprocess
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

try:
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _scipy_stats = None  # type: ignore
    _SCIPY_OK = False
    warnings.warn(
        "scipy not installed — Welch t-test significance will be skipped. "
        "Install with: pip install scipy",
        RuntimeWarning,
        stacklevel=2,
    )

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_WARMUP_DEFAULT = 5
N_SAMPLES_DEFAULT = 30


# ── Result dataclass (shared across all modules) ───────────────────────────

@dataclass
class BenchmarkResult:
    """
    Complete benchmark result for a single kernel variant.

    Fields are populated incrementally: timing fields are always present;
    roofline / occupancy / graph / correctness fields are filled by their
    respective analyzers and default to None when that analysis was not run.
    """
    # ── Identity ─────────────────────────────────────────────────────────
    kernel: str
    variant: str
    params: dict = field(default_factory=dict)

    # ── Timing (statistical) ──────────────────────────────────────────────
    mean_ms: float = 0.0
    std_ms: float = 0.0
    ci_95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    n_samples: int = 0
    raw_samples: List[float] = field(default_factory=list)

    # ── Statistical significance vs baseline ──────────────────────────────
    speedup: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: bool = False

    # ── Compilation ───────────────────────────────────────────────────────
    compile_ok: bool = True
    ptx_info: str = ""

    # ── Roofline model (filled by RooflineAnalyzer) ───────────────────────
    arithmetic_intensity: Optional[float] = None
    achieved_gflops: Optional[float] = None
    bound_type: Optional[str] = None          # "memory" | "compute"
    roofline_efficiency_pct: Optional[float] = None

    # ── Occupancy (filled by OccupancyAnalyzer) ───────────────────────────
    occupancy: Optional[float] = None
    registers_per_thread: Optional[int] = None
    shared_mem_bytes: Optional[int] = None
    has_register_spill: bool = False
    spill_stores: int = 0
    spill_loads: int = 0

    # ── CUDA Graph (filled by CUDAGraphBenchmark) ─────────────────────────
    graph_launch_ms: Optional[float] = None

    # ── Correctness (filled by CorrectnessVerifier or in-process CHECK) ──
    correctness_checked: bool = False
    is_correct: bool = True
    max_diff: Optional[float] = None

    # ── Nsight Compute hardware counters (Phase C) ────────────────────────
    ncu_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage (raw_samples excluded to save space)."""
        d = asdict(self)
        d.pop("raw_samples", None)  # don't bloat JSON with 30 floats per variant
        return d


# ── Compilation ────────────────────────────────────────────────────────────

def compile_kernel(
    src_path: Path,
    out_path: Path,
    arch: str = "sm_75",
    extra_flags: Optional[List[str]] = None,
) -> tuple[bool, str]:
    """
    Compile a .cu file with nvcc.

    Always passes -Xptxas -v so occupancy.py can parse register / smem stats.
    Returns (success, stderr).
    """
    flags = [
        "nvcc", "-O3", f"-arch={arch}",
        "--use_fast_math",
        "-Xptxas", "-v",
        str(src_path), "-o", str(out_path),
    ]
    if extra_flags:
        flags.extend(extra_flags)
    result = subprocess.run(flags, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


# ── Binary execution ───────────────────────────────────────────────────────

def run_binary(
    binary: Path,
    env_overrides: Optional[dict] = None,
    timeout: int = 120,
) -> tuple[bool, str, str]:
    """
    Run a compiled binary.

    Returns (success, stdout, stderr).
    """
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True,
        env=env, timeout=timeout,
    )
    return result.returncode == 0, result.stdout, result.stderr


# ── Output parsing ─────────────────────────────────────────────────────────

def parse_timing_output(stdout: str) -> dict[str, dict]:
    """
    Parse TIMING and SAMPLE lines emitted by benchmark binaries.

    Expected line formats::

        SAMPLE  <tag> <ms>           — one per timed iteration
        TIMING  <tag> <mean> <min> <max> <iters>  — aggregate summary

    Returns::

        {tag: {"mean_ms": float, "min_ms": float, "max_ms": float,
               "iters": int, "samples": [float, ...]}}
    """
    timings: dict[str, dict] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "SAMPLE" and len(parts) >= 3:
            tag = parts[1]
            timings.setdefault(tag, {"samples": []})
            timings[tag]["samples"].append(float(parts[2]))
        elif parts[0] == "TIMING" and len(parts) >= 5:
            tag = parts[1]
            timings.setdefault(tag, {"samples": []})
            timings[tag]["mean_ms"] = float(parts[2])
            timings[tag]["min_ms"]  = float(parts[3])
            timings[tag]["max_ms"]  = float(parts[4])
            timings[tag]["iters"]   = int(parts[5]) if len(parts) > 5 else 0
    return timings


def parse_timing_output_simple(stdout: str) -> dict[str, float]:
    """
    Legacy flat-dict parser (Phase A-D autotune.run_baseline path).

    Returns ``{tag: mean_ms}`` for each TIMING line found.
    """
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        if line.startswith("TIMING"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    out[parts[1]] = float(parts[2])
                except ValueError:
                    pass
    return out


def parse_check_output(stdout: str) -> dict[str, dict]:
    """
    Parse Phase-B CHECK lines emitted by the with-correctness driver.

    Expected::

        CHECK <tag> <max_rel_err> <pass>     # pass = 0 or 1

    Returns ``{tag: {"max_rel_err": float, "pass": bool}}``.
    """
    out: dict[str, dict] = {}
    for line in stdout.splitlines():
        if line.startswith("CHECK"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    out[parts[1]] = {
                        "max_rel_err": float(parts[2]),
                        "pass":         parts[3] == "1",
                    }
                except ValueError:
                    pass
    return out


# ── Statistics ─────────────────────────────────────────────────────────────

def _compute_stats(samples: List[float]) -> tuple[float, float, float, float, float]:
    """
    Compute (mean, std, ci_95, min, max) from a list of timing samples.

    Uses Bessel's correction (n-1) for sample std.
    95% CI = 1.96 * std / sqrt(n).
    """
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean = sum(samples) / n
    if n == 1:
        return mean, 0.0, 0.0, samples[0], samples[0]
    variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
    std = math.sqrt(variance)
    ci_95 = 1.96 * std / math.sqrt(n)
    return mean, std, ci_95, min(samples), max(samples)


def compute_significance(
    variant_samples: List[float],
    baseline_samples: List[float],
) -> tuple[Optional[float], Optional[float], bool, Optional[float]]:
    """
    Compare variant vs baseline using Welch's t-test.

    Returns (speedup, p_value, is_significant, baseline_mean).
    speedup > 1 means variant is faster (lower latency).
    is_significant requires p < 0.05 AND speedup > 1.
    """
    if not baseline_samples:
        return None, None, False, None

    baseline_mean = sum(baseline_samples) / len(baseline_samples)
    variant_mean  = sum(variant_samples)  / len(variant_samples) if variant_samples else 0.0

    speedup = (baseline_mean / variant_mean) if variant_mean > 0 else None
    p_value: Optional[float] = None
    is_significant = False

    if _SCIPY_OK and len(variant_samples) >= 2 and len(baseline_samples) >= 2:
        try:
            _, pv = _scipy_stats.ttest_ind(  # type: ignore[union-attr]
                baseline_samples, variant_samples, equal_var=False
            )
            p_value = float(pv)
            is_significant = p_value < 0.05 and speedup is not None and speedup > 1.0
        except Exception:
            pass
    elif speedup is not None:
        is_significant = speedup > 1.0  # no t-test, use heuristic

    return speedup, p_value, is_significant, baseline_mean


# ── Statistical benchmark runner ───────────────────────────────────────────

def run_statistical_benchmark(
    binary: Path,
    kernel_tag: str,
    baseline_samples: Optional[List[float]] = None,
    warmup: int = N_WARMUP_DEFAULT,
    n_samples: int = N_SAMPLES_DEFAULT,
    timeout: int = 180,
) -> Optional[BenchmarkResult]:
    """
    Run a compiled benchmark binary with statistical rigor.

    The binary must emit SAMPLE lines (one per iteration) and a TIMING
    aggregate line.  The binary is invoked once with WARMUP=warmup and
    ITERS=n_samples; it performs warmup internally then prints n_samples
    SAMPLE lines.

    Args:
        binary: Path to compiled executable.
        kernel_tag: Tag prefix used to match TIMING/SAMPLE lines.
        baseline_samples: Per-run baseline timings for Welch t-test.
        warmup: Number of warmup iterations passed to the binary.
        n_samples: Number of measured iterations (statistical samples).
        timeout: Subprocess timeout in seconds.

    Returns:
        Populated BenchmarkResult, or None if binary failed.
    """
    env = {"WARMUP": warmup, "ITERS": n_samples}
    try:
        ok, stdout, stderr = run_binary(binary, env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None

    if not ok:
        return None

    timings = parse_timing_output(stdout)
    if not timings:
        return None

    # Match tag: exact first, then prefix, then first available
    entry = timings.get(kernel_tag)
    if entry is None:
        for tag, data in timings.items():
            if kernel_tag in tag or tag.startswith(kernel_tag.split("_")[0]):
                entry = data
                break
    if entry is None:
        entry = next(iter(timings.values()))

    samples = entry.get("samples", [])
    # Fallback: if binary doesn't emit SAMPLE lines, synthesise from aggregate
    if not samples:
        m = entry.get("mean_ms", 0.0)
        samples = [m] if m else []

    mean_ms, std_ms, ci_95_ms, min_ms, max_ms = _compute_stats(samples)
    speedup, p_value, is_sig, _ = compute_significance(samples, baseline_samples or [])

    # Parse any Phase-B CHECK lines for this tag
    checks = parse_check_output(stdout)
    chk = checks.get(kernel_tag)
    if chk is None:
        for tag, data in checks.items():
            if kernel_tag in tag or tag.startswith(kernel_tag.split("_")[0]):
                chk = data
                break

    # Extract kernel name from tag
    kernel_name = kernel_tag.split("_")[0] if "_" in kernel_tag else kernel_tag

    result = BenchmarkResult(
        kernel=kernel_name,
        variant=kernel_tag,
        params={},
        mean_ms=mean_ms,
        std_ms=std_ms,
        ci_95_ms=ci_95_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        n_samples=len(samples),
        raw_samples=samples,
        speedup=speedup,
        p_value=p_value,
        is_significant=is_sig,
    )
    if chk is not None:
        result.correctness_checked = True
        result.is_correct          = bool(chk["pass"])
        result.max_diff            = chk["max_rel_err"]
    return result


# ── Baseline benchmark ─────────────────────────────────────────────────────

def run_baseline_benchmark(
    warmup: int = N_WARMUP_DEFAULT,
    n_samples: int = N_SAMPLES_DEFAULT,
) -> dict[str, dict]:
    """
    Compile and run the baseline benchmark binary.

    Returns a dict of per-kernel statistics::

        {
          "matmul_1024": {
            "mean_ms": float, "std_ms": float, "ci_95_ms": float,
            "samples": [float, ...]
          }, ...
        }

    Also saves results/baseline.json.
    """
    src    = ROOT / "src" / "kernels" / "benchmark_runner.cu"
    binary = ROOT / "results" / "bins" / "baseline_runner"
    binary.parent.mkdir(parents=True, exist_ok=True)

    print(f"[COMPILE] {src.name} ...", end=" ", flush=True)
    ok, stderr = compile_kernel(src, binary)
    if not ok:
        print("FAILED\n" + stderr)
        return {}
    print("OK")

    ptx_info = "\n".join(l for l in stderr.splitlines() if "ptxas info" in l.lower())

    print(f"[RUN] baseline  warmup={warmup}  samples={n_samples} ...")
    env = {"WARMUP": warmup, "ITERS": n_samples}
    try:
        ok, stdout, run_stderr = run_binary(binary, env, timeout=600)
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {}

    if not ok:
        print("BINARY FAILED:", run_stderr)
        return {}

    print(stdout)
    timings = parse_timing_output(stdout)

    results: dict[str, dict] = {}
    for tag, data in timings.items():
        samples = data.get("samples", [])
        if not samples and "mean_ms" in data:
            samples = [data["mean_ms"]]
        mean_ms, std_ms, ci_95_ms, min_ms, max_ms = _compute_stats(samples)
        results[tag] = {
            "mean_ms":  mean_ms,
            "std_ms":   std_ms,
            "ci_95_ms": ci_95_ms,
            "min_ms":   min_ms,
            "max_ms":   max_ms,
            "samples":  samples,
            "ptx_info": ptx_info,
        }

    out_path = RESULTS_DIR / "baseline.json"
    # Serialise without raw samples for conciseness
    save_data = {tag: {k: v for k, v in d.items() if k != "samples"}
                 for tag, d in results.items()}
    with open(out_path, "w") as f:
        import json
        json.dump(save_data, f, indent=2)
    print(f"[SAVED] {out_path}\n")
    return results


# ── Nsight Compute integration (Phase C) ──────────────────────────────────

_NCU_DEFAULT_METRICS = [
    "dram__bytes.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__occupancy_max_warps_active.avg.pct_of_peak_sustained_active",
]


def collect_ncu_metrics(
    binary: Path,
    metrics: Optional[List[str]] = None,
    kernel_regex: Optional[str] = None,
    launch_count: Optional[int] = None,
    env_overrides: Optional[dict] = None,
    timeout: int = 180,
) -> dict:
    """
    Run Nsight Compute (ncu) to collect hardware counters for ``binary``.

    Returns an empty dict if ncu is not installed or fails gracefully.

    Args:
        binary:        Path to the compiled benchmark binary to profile.
        metrics:       List of ncu metric names; a sensible default set if None.
        kernel_regex:  Restrict profiling to kernels matching this regex
                       (passed as ``-k regex:<pattern>``).  Useful to skip the
                       naive reference kernel that the with-correctness driver
                       also launches.
        launch_count:  Only profile the first N matching launches; big speedup
                       when the binary loops 100+ iterations for timing.
        env_overrides: Extra environment variables (e.g. WARMUP=0, ITERS=3
                       to keep the profiling run short).
        timeout:       Subprocess timeout in seconds.
    """
    if not metrics:
        metrics = list(_NCU_DEFAULT_METRICS)

    cmd = ["ncu",
           "--metrics", ",".join(metrics),
           "--csv",
           "--target-processes", "all"]
    if kernel_regex:
        cmd += ["-k", f"regex:{kernel_regex}"]
    if launch_count:
        cmd += ["--launch-count", str(launch_count)]
    cmd.append(str(binary))

    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env, timeout=timeout)
        if result.returncode != 0:
            return {}
        return _parse_ncu_csv(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def _parse_ncu_csv(csv_text: str) -> dict:
    """Parse ncu --csv output into {metric_name: value}."""
    metrics: dict = {}
    lines = csv_text.strip().splitlines()
    if len(lines) < 2:
        return metrics
    headers = [h.strip('"') for h in lines[0].split(",")]
    for line in lines[1:]:
        parts = [p.strip('"') for p in line.split(",")]
        if len(parts) < len(headers):
            continue
        row = dict(zip(headers, parts))
        name = row.get("Metric Name", "")
        val  = row.get("Metric Value", "")
        if name:
            try:
                metrics[name] = float(val.replace(",", ""))
            except ValueError:
                metrics[name] = val
    return metrics


# ── CLI (standalone baseline runner) ──────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Baseline CUDA kernel benchmarker")
    parser.add_argument("--warmup",   type=int, default=N_WARMUP_DEFAULT)
    parser.add_argument("--samples",  type=int, default=N_SAMPLES_DEFAULT)
    args = parser.parse_args()

    results = run_baseline_benchmark(args.warmup, args.samples)
    if results:
        print(f"\n{'='*60}")
        print(f"{'Kernel':<26} {'Mean':>8} {'Std':>7} {'CI±':>7}")
        print("-" * 60)
        for tag, r in results.items():
            print(f"{tag:<26} {r['mean_ms']:>7.3f}ms "
                  f"{r['std_ms']:>6.3f}ms "
                  f"{r['ci_95_ms']:>6.3f}ms")
        print("=" * 60)

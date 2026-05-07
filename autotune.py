"""
autotune.py — main entry point for the CUDA kernel auto-tuner.

Usage:
    python autotune.py --kernel=matmul
    python autotune.py --kernel=all --strategy=bayesian
    python autotune.py --kernel=reduction --strategy=sha
    python autotune.py --baseline-only
    python autotune.py --kernel=softmax --ptx-analysis --cuda-graphs
    python autotune.py --kernel=matmul --skip-verification --workers=4
    python autotune.py --kernel=matmul --use-libclang --ncu --plots

Pipeline:
    1. Parse baseline kernel → extract profile + search space
    2. Generate variant .cu files (all or via search strategy)
    3. [Optional] Verify correctness of each variant vs NumPy reference
    4. Compile each variant with nvcc (parallel)
    5. Statistical benchmark: 5 warmup + 30 timed runs (CUDA events)
    6. Welch t-test vs baseline; flag statistically significant wins
    7. Roofline model: classify memory-bound vs compute-bound
    8. Occupancy analysis: parse ptxas register/smem output
    9. [Optional] PTX/SASS analysis (--ptx-analysis)
   10. [Optional] CUDA Graph overhead measurement (--cuda-graphs)
   11. [Optional] Nsight Compute hardware counters (--ncu)
   12. [Optional] Render Phase-D figures (--plots)
   13. Generate Markdown report + print terminal summary table
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

ROOT        = Path(__file__).parent
SRC_DIR     = ROOT / "src"
RESULTS_DIR = ROOT / "results"
GEN_DIR     = RESULTS_DIR / "generated"
BINS_DIR    = RESULTS_DIR / "bins"

for d in (GEN_DIR, BINS_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from parser    import (KernelProfile, MemoryAccessPattern, build_search_space,
                       find_kernel)
from generator import enumerate_variants, write_variant, ARCH
from benchmark import (
    BenchmarkResult, compile_kernel, run_binary, parse_timing_output,
    run_statistical_benchmark, run_baseline_benchmark, collect_ncu_metrics,
    N_WARMUP_DEFAULT, N_SAMPLES_DEFAULT,
)
from roofline  import RooflineAnalyzer, get_dims
from occupancy import OccupancyAnalyzer, parse_ptxas_stderr
from search    import make_strategy, ConvergenceLogger, GridSearchStrategy
from reporter  import ReportGenerator, print_terminal_summary, print_final_summary

# Optional: Phase A-D analytic roofline (lower-bound table) lives alongside
# the friend's RooflineAnalyzer.  Both can coexist; we use the analytic
# variant for the pretty CLI table when --plots is requested.
try:
    from roofline import compute_roofline, format_table
    _ROOFLINE_ANALYTIC_OK = True
except ImportError:
    _ROOFLINE_ANALYTIC_OK = False

# Optional Phase-D figure renderer
try:
    from plots import main as _render_plots_main
    _PLOTS_OK = True
except ImportError:
    _PLOTS_OK = False

# Optional modules (degrade gracefully)
try:
    from verifier import CorrectnessVerifier
    _VERIFIER_OK = True
except ImportError:
    _VERIFIER_OK = False

try:
    from ptx_analysis import PTXAnalyzer
    _PTX_OK = True
except ImportError:
    _PTX_OK = False

try:
    from cuda_graph import CUDAGraphBenchmark
    _GRAPH_OK = True
except ImportError:
    _GRAPH_OK = False

# Attention kernel (always-available — pure Python templates)
try:
    from attention import (
        enumerate_attention_variants,
        write_attention_variant,
        build_attention_search_space,
        attention_flops,
        attention_bytes_flash,
        SEQ_LEN as ATT_SEQ_LEN,
        D_HEAD  as ATT_D_HEAD,
    )
    _ATTENTION_OK = True
except ImportError:
    _ATTENTION_OK = False

SUPPORTED_KERNELS = ["matmul", "softmax", "reduction", "layernorm", "attention"]

_BASELINE_SRC = SRC_DIR / "kernels" / "baseline_kernels.cu"


# ── Kernel profile builder ─────────────────────────────────────────────────

def _hardcoded_profile(kernel: str) -> KernelProfile:
    """Kernel-name-based fallback used when source parsing fails."""
    memory = MemoryAccessPattern(
        has_global_load    = True,
        has_global_store   = True,
        has_strided_access = kernel == "matmul",
        has_reduction      = kernel in ("reduction", "softmax"),
        has_shared_mem     = False,
    )
    return KernelProfile(
        name          = kernel,
        src_path      = _BASELINE_SRC,
        block_dim     = 16,
        uses_shared   = False,
        loop_depth    = 2 if kernel == "matmul" else 1,
        reduction_ops = ["+="] if kernel in ("reduction", "softmax") else [],
        memory        = memory,
        backend       = "hardcoded",
    )


def build_kernel_profile(kernel: str, use_libclang: bool = False) -> KernelProfile:
    """
    Return a KernelProfile for the given kernel name.

    When use_libclang=True (Phase A), the libclang/regex parser is invoked
    on the baseline source so the search space is driven by the actual AST
    contents.  Otherwise a hardcoded heuristic profile is returned (this is
    the friend's fast path that doesn't require libclang to be installed).
    """
    if use_libclang:
        profile = find_kernel(_BASELINE_SRC, kernel, verbose=True)
        if profile is not None:
            print(f"[PARSE] {kernel} → {profile.name}  "
                  f"[backend={profile.backend}]  "
                  f"loop_depth={profile.loop_depth}  "
                  f"shared={profile.uses_shared}  "
                  f"strided={profile.memory.has_strided_access}  "
                  f"reduction={profile.memory.has_reduction}")
            return profile
        print(f"[PARSE] {kernel}: no match in {_BASELINE_SRC.name}, "
              f"using hardcoded fallback")

    return _hardcoded_profile(kernel)


# ── Compilation ────────────────────────────────────────────────────────────

def compile_variant(src: Path, binary: Path) -> tuple[bool, str]:
    """Compile a variant .cu with nvcc, capturing ptxas info."""
    cmd = [
        "nvcc", "-O3", f"-arch={ARCH}",
        "--use_fast_math",
        "-Xptxas", "-v",
        str(src), "-o", str(binary),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr


# ── Per-variant processing ─────────────────────────────────────────────────

def process_variant(
    kernel: str,
    params: dict,
    src_path: Path,
    baseline_samples: list[float],
    warmup: int,
    n_samples: int,
    roofline: RooflineAnalyzer,
    occ_analyzer: OccupancyAnalyzer,
    verifier: Optional[object],
    graph_bench: Optional[object],
    skip_verification: bool,
) -> Optional[BenchmarkResult]:
    """
    Compile, (verify), benchmark, and analyse one kernel variant.

    Returns a fully populated BenchmarkResult or None on compile failure.
    """
    binary = BINS_DIR / (src_path.stem + ".exe")

    # 1. Compile
    ok, compile_stderr = compile_variant(src_path, binary)
    if not ok:
        return None

    # 2. Correctness verification (always-on unless --skip-verification)
    if verifier is not None and not skip_verification and _VERIFIER_OK:
        vr = verifier.check(kernel, src_path, params)
        if not vr.is_correct:
            return None   # Skip benchmarking failed variants

    # 3. Statistical benchmark
    result = run_statistical_benchmark(
        binary, src_path.stem,
        baseline_samples=baseline_samples,
        warmup=warmup,
        n_samples=n_samples,
    )
    if result is None:
        return None

    # In-process Phase-B CHECK: drop variants that failed numerical comparison
    if result.correctness_checked and not result.is_correct:
        return None

    result.params    = params
    result.kernel    = kernel
    result.variant   = src_path.stem
    result.ptx_info  = compile_stderr

    # 4. Roofline analysis
    dims = get_dims(kernel)
    result.arithmetic_intensity    = roofline.arithmetic_intensity(kernel, dims)
    result.achieved_gflops         = roofline.achieved_gflops(kernel, dims, result.mean_ms)
    result.bound_type              = roofline.bound_type(kernel, dims)
    result.roofline_efficiency_pct = roofline.efficiency_pct(kernel, dims, result.mean_ms)

    # 5. Occupancy analysis (from ptxas stderr captured during compile)
    occ_info = occ_analyzer.analyze_from_stderr(
        compile_stderr,
        kernel_function=f"{kernel}_opt",
        block_size=params.get("block_size", 64),
    )
    if occ_info:
        result.occupancy             = occ_info.occupancy
        result.registers_per_thread  = occ_info.registers_per_thread
        result.shared_mem_bytes      = occ_info.shared_mem_bytes
        result.has_register_spill    = occ_info.has_register_spill
        result.spill_stores          = occ_info.spill_stores
        result.spill_loads           = occ_info.spill_loads

    # 6. CUDA Graph benchmark (optional)
    if graph_bench is not None and _GRAPH_OK and graph_bench.available:
        gr = graph_bench.benchmark(binary, kernel, result.variant, result.mean_ms)
        if gr is not None:
            result.graph_launch_ms = gr.graph_mean_ms

    return result


# ── Main auto-tuning loop ─────────────────────────────────────────────────

def autotune_kernel(
    kernel: str,
    baseline_data: dict,
    strategy_name: str = "grid",
    max_workers: int = 2,
    warmup: int = N_WARMUP_DEFAULT,
    n_samples: int = N_SAMPLES_DEFAULT,
    skip_verification: bool = False,
    run_ptx_analysis: bool = False,
    run_cuda_graphs: bool = False,
    matrix_size: int = 1024,
    use_libclang: bool = False,
    with_correctness: bool = False,
    run_ncu: bool = False,
) -> list[BenchmarkResult]:
    """
    Auto-tune one kernel: generate → verify → compile → benchmark → analyse.

    Args:
        kernel:           Kernel name.
        baseline_data:    Output of run_baseline_benchmark() for this kernel.
        strategy_name:    "grid" | "bayesian" | "sha"
        max_workers:      Parallel compile+benchmark workers.
        warmup:           Warmup iterations per variant.
        n_samples:        Statistical samples per variant.
        skip_verification: If True, skip NumPy correctness checking.
        run_ptx_analysis: If True, generate and analyse PTX.
        run_cuda_graphs:  If True, add CUDA Graph timing.
        matrix_size:      Problem size N for matmul; rows for softmax/layernorm.
        use_libclang:     Phase A — drive search space from libclang AST parse
                          of the baseline source instead of the hardcoded
                          profile.
        with_correctness: Phase B — emit the in-process CHECK driver in each
                          generated variant (compares against the naive
                          reference kernel embedded in the same binary).
        run_ncu:          Phase C — profile the best variant with Nsight
                          Compute and embed measured DRAM/cache counters in
                          the tuning JSON.
    """
    profile  = build_kernel_profile(kernel, use_libclang=use_libclang)
    space    = build_search_space(profile)
    variants = enumerate_variants(kernel, space)

    total = len(variants)
    print(f"\n[GENERATE] {total} variants for '{kernel}' ...")
    for params, src_path in variants:
        write_variant(kernel, params, src_path,
                      matrix_size=matrix_size,
                      with_correctness=with_correctness)

    # Extract baseline samples for Welch t-test
    baseline_samples: list[float] = baseline_data.get("samples", [])
    if not baseline_samples:
        bms = baseline_data.get("mean_ms")
        if bms:
            baseline_samples = [bms]

    # Instantiate analyzers
    roofline      = RooflineAnalyzer()
    occ_analyzer  = OccupancyAnalyzer(arch=ARCH)
    verifier      = CorrectnessVerifier(verbose=False) if _VERIFIER_OK else None
    graph_bench   = (CUDAGraphBenchmark() if (run_cuda_graphs and _GRAPH_OK) else None)
    ptx_analyzer  = (PTXAnalyzer(results_dir=RESULTS_DIR) if run_ptx_analysis else None)
    conv_logger   = ConvergenceLogger(kernel, RESULTS_DIR)

    # Build search strategy
    strategy = make_strategy(strategy_name, space, kernel=kernel)

    print(f"[STRATEGY] {strategy.name}  "
          f"warmup={warmup}  samples={n_samples}  workers={max_workers}")
    if not skip_verification:
        print(f"[VERIFY] Correctness checking enabled (numpy reference)")
    else:
        print(f"[VERIFY] WARNING: correctness verification SKIPPED (--skip-verification)")
    if with_correctness:
        print(f"[CHECK] In-process CHECK driver enabled (naive ref embedded in binary)")
    if use_libclang:
        print(f"[FRONTEND] libclang AST parse of {_BASELINE_SRC.name} "
              f"→ profile.backend={profile.backend}")

    results: list[BenchmarkResult] = []
    done  = 0
    t0    = time.time()

    # Live progress JSON path
    live_progress_path = RESULTS_DIR / "live_progress.json"
    live_progress_path.write_text(json.dumps({
        "kernel": kernel, "strategy": strategy_name,
        "done": 0, "total": total,
        "best_ms": None, "baseline_ms": baseline_data.get("mean_ms"),
        "best_speedup": None, "best_params": None, "recent": [],
    }, indent=2))

    def _process(item: tuple) -> Optional[BenchmarkResult]:
        params, src_path = item
        return process_variant(
            kernel, params, src_path,
            baseline_samples, warmup, n_samples,
            roofline, occ_analyzer, verifier, graph_bench, skip_verification,
        )

    # All strategies: iterate over all variants (strategy controls ordering)
    # For grid: enumerate_variants already returns all; no need to re-suggest
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, v): v for v in variants}
        for fut in as_completed(futures):
            params, src_path = futures[fut]
            done += 1
            elapsed = time.time() - t0
            eta     = elapsed / done * (total - done)
            try:
                result = fut.result()
            except Exception:
                result = None

            if result is not None:
                results.append(result)
                strategy.update(params, result.mean_ms)
                conv_logger.record(params, result.mean_ms)
                status = (f"{result.mean_ms:.3f}ms  "
                          f"{'OK' if result.is_significant else '.'}")
            else:
                status = "FAIL"

            print(f"  [{done:3d}/{total}] {status:<20} ETA {eta:.0f}s", end="\r")

            # Write live progress JSON after every variant
            best_so_far = min(results, key=lambda r: r.mean_ms) if results else None
            bms = baseline_data.get("mean_ms")
            recent = [
                {"variant": r.variant,
                 "mean_ms": round(r.mean_ms, 4),
                 "speedup": round(r.speedup, 3) if r.speedup else None,
                 "passed": r.is_significant}
                for r in sorted(results, key=lambda r: r.mean_ms)[:5]
            ]
            try:
                live_progress_path.write_text(json.dumps({
                    "kernel": kernel, "strategy": strategy_name,
                    "done": done, "total": total,
                    "best_ms": round(best_so_far.mean_ms, 4) if best_so_far else None,
                    "baseline_ms": bms,
                    "best_speedup": round(best_so_far.speedup, 3)
                                    if (best_so_far and best_so_far.speedup) else None,
                    "best_params": best_so_far.params if best_so_far else None,
                    "recent": recent,
                }, indent=2))
            except Exception:
                pass  # never let progress writes crash the main loop

    print()  # clear \r line

    # Verifier summary
    if verifier is not None and not skip_verification:
        verifier.print_summary()
        verifier.save_failures(kernel)

    # Sort results best-first
    results.sort(key=lambda r: r.mean_ms)

    if not results:
        print(f"[ERROR] No variants compiled+verified+benchmarked successfully.")
        return []

    # ── Phase C: Nsight Compute on the best variant ──────────────────────
    ncu_metrics: dict = {}
    if run_ncu and results:
        best_bin = BINS_DIR / (results[0].variant + ".exe")
        if best_bin.exists():
            print(f"[NCU] profiling {best_bin.name} "
                  f"(kernel filter: .*_opt, launches=3) ...")
            ncu_metrics = collect_ncu_metrics(
                best_bin,
                kernel_regex=".*_opt",
                launch_count=3,
                env_overrides={"WARMUP": 0, "ITERS": 3},
            )
            if ncu_metrics:
                print(f"[NCU] collected {len(ncu_metrics)} metric(s)")
                results[0].ncu_metrics = ncu_metrics
            else:
                print(f"[NCU] no metrics collected (ncu may be unavailable)")

    # Save tuning JSON
    baseline_ms = baseline_data.get("mean_ms")
    out_path = RESULTS_DIR / f"{kernel}_tuning.json"
    with open(out_path, "w") as f:
        json.dump({
            "kernel":      kernel,
            "strategy":    strategy_name,
            "n_variants":  total,
            "n_results":   len(results),
            "baseline_ms": baseline_ms,
            "best":        results[0].to_dict() if results else None,
            "variants":    [r.to_dict() for r in results[:50]],  # top 50
            "ncu_metrics": ncu_metrics,
        }, f, indent=2)

    # Save convergence
    conv_logger.save()

    # PTX analysis (optional)
    if ptx_analyzer and results:
        best_src = GEN_DIR / (results[0].variant + ".cu")
        baseline_src = SRC_DIR / "kernels" / "baseline_kernels.cu"
        if best_src.exists() and baseline_src.exists():
            b_metrics = ptx_analyzer.analyze_source(baseline_src)
            o_metrics = ptx_analyzer.analyze_source(best_src)
            cmp = ptx_analyzer.compare(kernel, b_metrics, o_metrics)
            ptx_analyzer.save(kernel, cmp)
            ptx_analyzer.print_comparison(cmp)

    # Roofline summary table
    if baseline_ms and results:
        dims = get_dims(kernel)
        roofline.print_summary_table([
            (kernel, dims, baseline_ms),
            (kernel, dims, results[0].mean_ms),
        ], title=f"Roofline — {kernel} (baseline vs best)")

    # Phase A-D analytic roofline (compute_roofline) table — uses ncu metrics
    # when present so the user sees both analytic and measured BW/AI.
    if _ROOFLINE_ANALYTIC_OK and results:
        try:
            ar = compute_roofline(kernel, results[0].mean_ms, ncu_metrics)
            print(format_table([ar]))
        except Exception:
            pass

    # Occupancy table
    occ_rows = [r for r in results[:10] if r.occupancy is not None]
    if occ_rows:
        occ_analyzer.print_table(occ_rows, title=f"Occupancy — {kernel} top 10")

    # Terminal summary
    print_terminal_summary(
        kernel=kernel,
        results=results,
        baseline_ms=baseline_ms,
        n_total=total,
        strategy=strategy_name,
    )

    # Markdown report
    rg = ReportGenerator(kernel, strategy=strategy_name)
    rg.set_results(results, baseline_ms)
    if baseline_ms and results:
        dims = get_dims(kernel)
        b_pt = {
            "arithmetic_intensity": roofline.arithmetic_intensity(kernel, dims),
            "achieved_gflops":      roofline.achieved_gflops(kernel, dims, baseline_ms),
            "bound_type":           roofline.bound_type(kernel, dims),
            "efficiency_pct":       roofline.efficiency_pct(kernel, dims, baseline_ms),
        }
        o_pt = {
            "arithmetic_intensity": results[0].arithmetic_intensity,
            "achieved_gflops":      results[0].achieved_gflops,
            "bound_type":           results[0].bound_type,
            "efficiency_pct":       results[0].roofline_efficiency_pct,
        }
        rg.set_roofline_points(b_pt, o_pt)
    if verifier:
        rg.set_correctness(
            n_passed=verifier._n_passed,
            n_failed=verifier._n_failed,
            n_checked=verifier._n_checked,
        )
    if ptx_analyzer:
        ptx_path = RESULTS_DIR / f"{kernel}_ptx_analysis.json"
        if ptx_path.exists():
            with open(ptx_path) as f:
                rg.set_ptx_data(json.load(f))
    conv_path = RESULTS_DIR / f"{kernel}_convergence.json"
    if conv_path.exists():
        with open(conv_path) as f:
            rg.set_convergence(json.load(f))
    rg.generate()

    return results


# ── Attention auto-tuner ───────────────────────────────────────────────────

def autotune_attention(
    warmup: int = N_WARMUP_DEFAULT,
    n_samples: int = N_SAMPLES_DEFAULT,
) -> list[BenchmarkResult]:
    """
    Auto-tune all attention variants (Tiled + Flash × seq_tile × unroll).
    Writes results/attention_tuning.json and updates live_progress.json.
    """
    if not _ATTENTION_OK:
        print("[ATTENTION] attention.py not importable — skipping.")
        return []

    variants = enumerate_attention_variants()
    total    = len(variants)
    print(f"\n[ATTENTION] {total} variants (Tiled + Flash variants) ...")

    for params, src_path in variants:
        write_attention_variant(params, src_path)

    live_progress_path = RESULTS_DIR / "live_progress.json"
    live_progress_path.write_text(json.dumps({
        "kernel": "attention", "strategy": "grid",
        "done": 0, "total": total,
        "best_ms": None, "baseline_ms": None,
        "best_speedup": None, "best_params": None, "recent": [],
    }, indent=2))

    roofline     = RooflineAnalyzer()
    occ_analyzer = OccupancyAnalyzer(arch=ARCH)

    results: list[BenchmarkResult] = []
    done = 0
    t0   = time.time()

    for params, src_path in variants:
        binary = BINS_DIR / (src_path.stem + ".exe")
        ok, compile_stderr = compile_variant(src_path, binary)
        done += 1
        if not ok:
            print(f"  [{done:3d}/{total}] COMPILE FAIL", end="\r")
            continue

        result = run_statistical_benchmark(
            binary, src_path.stem,
            baseline_samples=[],
            warmup=warmup,
            n_samples=n_samples,
        )
        if result is None:
            continue

        result.params  = params
        result.kernel  = "attention"
        result.variant = src_path.stem
        result.ptx_info = compile_stderr

        # Roofline for attention (no get_dims integration — use attention model)
        s, d = ATT_SEQ_LEN, ATT_D_HEAD
        flops = attention_flops(s, d)
        byt   = attention_bytes_flash(s, d) if params.get("flash") else int((3*s*d + 2*s*s + s*d)*4)
        result.arithmetic_intensity    = flops / byt if byt else 0
        result.achieved_gflops         = (flops / 1e9) / (result.mean_ms / 1000) if result.mean_ms else 0
        peak = 7500  # RTX 2070 FP32 GFLOP/s
        result.bound_type              = "compute" if result.arithmetic_intensity > 16.74 else "memory"
        result.roofline_efficiency_pct = (result.achieved_gflops / peak) * 100

        # Occupancy
        occ_info = occ_analyzer.analyze_from_stderr(
            compile_stderr,
            kernel_function="attention_flash" if params.get("flash") else "attention_tiled",
            block_size=ATT_D_HEAD,
        )
        if occ_info:
            result.occupancy            = occ_info.occupancy
            result.registers_per_thread = occ_info.registers_per_thread
            result.shared_mem_bytes     = occ_info.shared_mem_bytes

        results.append(result)
        elapsed = time.time() - t0
        print(f"  [{done:3d}/{total}] {result.mean_ms:.3f}ms  ETA {elapsed/done*(total-done):.0f}s", end="\r")

        # Update live progress JSON
        best_so_far = min(results, key=lambda r: r.mean_ms) if results else None
        recent = [
            {"variant": r.variant, "mean_ms": round(r.mean_ms, 4),
             "passed": True, "flash": r.params.get("flash", False)}
            for r in sorted(results, key=lambda r: r.mean_ms)[:5]
        ]
        try:
            live_progress_path.write_text(json.dumps({
                "kernel": "attention", "strategy": "grid",
                "done": done, "total": total,
                "best_ms": round(best_so_far.mean_ms, 4) if best_so_far else None,
                "baseline_ms": None,
                "best_speedup": round(best_so_far.speedup, 3)
                                if (best_so_far and best_so_far.speedup) else None,
                "best_params": best_so_far.params if best_so_far else None,
                "recent": recent,
            }, indent=2))
        except Exception:
            pass

    print()  # clear \r line
    results.sort(key=lambda r: r.mean_ms)

    if not results:
        print("[ATTENTION] No variants compiled successfully.")
        return []

    out_path = RESULTS_DIR / "attention_tuning.json"
    with open(out_path, "w") as f:
        json.dump({
            "kernel": "attention",
            "strategy": "grid",
            "n_variants": total,
            "n_results": len(results),
            "baseline_ms": None,
            "best": results[0].to_dict() if results else None,
            "variants": [r.to_dict() for r in results[:50]],
        }, f, indent=2)

    best = results[0]
    variant_type = "Flash" if best.params.get("flash") else "Tiled"
    print(f"\n[ATTENTION] Best: {best.mean_ms:.3f}ms — {variant_type} "
          f"seq_tile={best.params.get('seq_tile')} unroll={best.params.get('unroll')}")
    print(f"[ATTENTION] {best.achieved_gflops:.1f} GFLOP/s  "
          f"({best.roofline_efficiency_pct:.1f}% of peak)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUDA Kernel Auto-Tuner — RTX 2070 (sm_75)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autotune.py --kernel=matmul
  python autotune.py --kernel=all --strategy=bayesian
  python autotune.py --kernel=reduction --strategy=sha --workers=4
  python autotune.py --baseline-only
  python autotune.py --kernel=softmax --ptx-analysis --cuda-graphs
  python autotune.py --kernel=matmul --skip-verification
  python autotune.py --kernel=matmul --use-libclang --ncu --plots
""",
    )
    parser.add_argument(
        "--kernel", choices=SUPPORTED_KERNELS + ["all"], default="matmul",
        help="Kernel to tune (default: matmul)"
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only run baseline benchmarks, skip tuning"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Parallel compile+benchmark workers (default: 2)"
    )
    parser.add_argument(
        "--warmup", type=int, default=N_WARMUP_DEFAULT,
        help=f"GPU warmup iterations per variant (default: {N_WARMUP_DEFAULT})"
    )
    parser.add_argument(
        "--samples", type=int, default=N_SAMPLES_DEFAULT,
        help=f"Statistical samples per variant (default: {N_SAMPLES_DEFAULT})"
    )
    # Legacy --iters alias
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Alias for --samples (backward compatibility)"
    )
    parser.add_argument(
        "--strategy", choices=["grid", "bayesian", "sha"], default="grid",
        help="Search strategy: grid | bayesian | sha (default: grid)"
    )
    parser.add_argument(
        "--skip-verification", action="store_true",
        help="Skip correctness verification (faster, but unsafe)"
    )
    parser.add_argument(
        "--ptx-analysis", action="store_true",
        help="Enable PTX/SASS instruction analysis (adds compile time)"
    )
    parser.add_argument(
        "--cuda-graphs", action="store_true",
        help="Measure CUDA Graph launch overhead vs event-based timing"
    )
    parser.add_argument(
        "--matrix-size", type=int, default=1024, choices=[512, 1024, 2048],
        help="Problem size N for matmul (N×N), rows for softmax/layernorm (default: 1024)"
    )
    # ── Phase A-D flags ────────────────────────────────────────────────
    parser.add_argument(
        "--use-libclang", action="store_true",
        help="Phase A: drive search space from libclang AST parse of "
             "baseline_kernels.cu (falls back to regex / hardcoded profile)"
    )
    parser.add_argument(
        "--with-correctness", action="store_true",
        help="Phase B: emit in-process CHECK driver in each generated variant "
             "(compares against the naive reference kernel embedded in the binary)"
    )
    parser.add_argument(
        "--ncu", action="store_true",
        help="Phase C: profile the best variant with Nsight Compute and "
             "embed the hardware counters in the tuning JSON"
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Phase D: render figures (speedup bars, roofline plot) via plots.py "
             "after tuning completes"
    )
    # ── Phase 1 (CDC frontend) flags ───────────────────────────────────
    parser.add_argument(
        "--cdc-frontend", action="store_true",
        help="Phase 1: run the PLY-based compiler frontend (lex + parse + "
             "AST + symbol table + type check) over baseline_kernels.cu and "
             "print the report before tuning"
    )
    parser.add_argument(
        "--cdc-frontend-only", action="store_true",
        help="Run the CDC frontend and exit (no compilation, no tuning). "
             "Equivalent to: python -m cdc src/kernels/baseline_kernels.cu"
    )
    parser.add_argument(
        "--cdc-ir", action="store_true",
        help="Phase 2: emit TAC + basic blocks + CFG + DAG for every kernel "
             "after the frontend pass.  Equivalent to: python -m cdc <file> --ir"
    )
    parser.add_argument(
        "--cdc-opt", action="store_true",
        help="Phase 3: run all classical optimisation passes (const-prop, "
             "CSE, DCE, LICM, strength reduction) and print stats per kernel"
    )
    parser.add_argument(
        "--cdc-regs", action="store_true",
        help="Phase 3: estimate per-kernel register pressure from live-vars "
             "and feed the auto-tuner's tile/unroll cost model"
    )

    args = parser.parse_args()

    # ── CDC frontend (Phase 1) ─────────────────────────────────────────
    if args.cdc_frontend or args.cdc_frontend_only or args.cdc_ir \
       or args.cdc_opt or args.cdc_regs:
        try:
            from cdc.frontend import run_frontend, format_report
            kernels_cu = Path(__file__).parent / "src" / "kernels" / "baseline_kernels.cu"
            print("[CDC] running PLY frontend (lex + parse + AST + symtab + typecheck) ...")
            result = run_frontend(kernels_cu)
            if args.cdc_frontend or args.cdc_frontend_only:
                print(format_report(result, show_ast=False, show_symbols=True))
                print()
            if not result.ok():
                print("[CDC] frontend reported errors; exiting.", flush=True)
                return
        except Exception as e:
            print(f"[CDC] frontend failed: {e}")
            if args.cdc_frontend_only:
                return

        # Phase 2: TAC + basic blocks + CFG + DAG
        if args.cdc_ir or args.cdc_opt or args.cdc_regs:
            try:
                from cdc.ir import emit_tac, partition_blocks, build_cfg, build_dag
                if args.cdc_ir:
                    print("[CDC] running IR pipeline (TAC -> BB -> CFG -> DAG) ...")
                    for k in result.kernels:
                        prog = emit_tac(k.ast)
                        blocks = partition_blocks(prog)
                        cfg = build_cfg(blocks)
                        n_dag = sum(len(build_dag(b)) for b in blocks)
                        n_edges = sum(len(s) for s in cfg.succ.values())
                        print(f"  {k.name:<20} "
                              f"quads={len(prog.quads):>3}  "
                              f"BBs={len(blocks):>2}  "
                              f"edges={n_edges:>2}  "
                              f"DAG-nodes={n_dag:>3}")
                    print()
            except Exception as e:
                print(f"[CDC] IR pipeline failed: {e}")

        # Phase 3: optimisation passes + register-pressure cost model
        if args.cdc_opt or args.cdc_regs:
            try:
                from cdc.ir import emit_tac, partition_blocks, build_cfg
                from cdc.opt import (
                    constant_propagation, common_subexpression_elimination,
                    dead_code_elimination, loop_invariant_code_motion,
                    strength_reduction, estimate_register_pressure,
                )
                if args.cdc_opt:
                    print("[CDC] running optimisation passes "
                          "(const-prop / CSE / strength / LICM / DCE) ...")
                    print(f"  {'kernel':<20} "
                          f"{'fold':>4} {'prop':>4} {'CSE':>4} {'DCE':>4} "
                          f"{'sr':>4} {'loops':>5} {'hoist':>5}")
                    for k in result.kernels:
                        prog = emit_tac(k.ast)
                        blocks = partition_blocks(prog)
                        cfg = build_cfg(blocks)
                        cp = constant_propagation(prog, blocks)
                        cs = common_subexpression_elimination(blocks)
                        sr = strength_reduction(blocks)
                        lc = loop_invariant_code_motion(blocks, cfg)
                        dc = dead_code_elimination(blocks, cfg)
                        print(f"  {k.name:<20} "
                              f"{cp['folded']:>4} {cp['propagated']:>4} "
                              f"{cs['eliminated']:>4} {dc['removed']:>4} "
                              f"{sr['rewritten']:>4} "
                              f"{lc['loops_found']:>5} {lc['hoisted']:>5}")
                    print()

                if args.cdc_regs:
                    print("[CDC] register-pressure cost model "
                          "(live-vars -> tile/unroll budget) ...")
                    print(f"  {'kernel':<20} {'max_live':>9} {'avg_live':>9} "
                          f"{'sugg_unroll':>11} {'sugg_tile':>10}")
                    for k in result.kernels:
                        prog = emit_tac(k.ast)
                        blocks = partition_blocks(prog)
                        cfg = build_cfg(blocks)
                        rp = estimate_register_pressure(k.name, blocks, cfg)
                        print(f"  {rp.kernel:<20} "
                              f"{rp.max_live:>9} "
                              f"{rp.avg_live:>9.1f} "
                              f"{rp.suggested_max_unroll:>11} "
                              f"{rp.suggested_max_tile:>10}")
                    print()

                print("[CDC] use 'python -m cdc <file.cu> --opt --kernel <name>' "
                      "for full TAC dumps.")
                print()
            except Exception as e:
                import traceback
                print(f"[CDC] opt pipeline failed: {e}")
                traceback.print_exc()

        if args.cdc_frontend_only:
            return

    # --iters is a backward-compat alias for --samples
    n_samples = args.samples
    if args.iters is not None:
        n_samples = args.iters

    print("CUDA Kernel Auto-Tuner  |  RTX 2070  |  sm_75\n")

    # Run baseline
    print("[BASELINE] Compiling and benchmarking naive kernels ...\n")
    baselines = run_baseline_benchmark(
        warmup=args.warmup,
        n_samples=n_samples,
    )

    if not baselines:
        print("[ERROR] Baseline benchmark failed. Check CUDA installation.")
        sys.exit(1)

    if args.baseline_only:
        print("\n[DONE] Baseline only — exiting.")
        return

    kernels = ([k for k in SUPPORTED_KERNELS if k != "attention"]
               if args.kernel == "all"
               else ([] if args.kernel == "attention"
                     else [args.kernel]))

    all_results: dict[str, list[BenchmarkResult]] = {}

    for k in kernels:
        # Find baseline data for this kernel (match by prefix)
        baseline_data = next(
            (v for tag, v in baselines.items() if tag.startswith(k)),
            {}
        )

        results = autotune_kernel(
            kernel=k,
            baseline_data=baseline_data,
            strategy_name=args.strategy,
            max_workers=args.workers,
            warmup=args.warmup,
            n_samples=n_samples,
            skip_verification=args.skip_verification,
            run_ptx_analysis=args.ptx_analysis,
            run_cuda_graphs=args.cuda_graphs,
            matrix_size=args.matrix_size,
            use_libclang=args.use_libclang,
            with_correctness=args.with_correctness,
            run_ncu=args.ncu,
        )
        all_results[k] = results

    # Attention kernel (separate pipeline, no baseline needed)
    if args.kernel in ("attention", "all"):
        att_results = autotune_attention(
            warmup=args.warmup,
            n_samples=n_samples,
        )
        if att_results:
            all_results["attention"] = att_results

    # Final multi-kernel summary
    if len(all_results) > 1:
        print_final_summary(all_results, baselines)

    # ── Phase D: render figures ────────────────────────────────────────
    if args.plots:
        if _PLOTS_OK:
            try:
                print("\n[PLOTS] Rendering figures via plots.py ...")
                _render_plots_main(RESULTS_DIR)
                print("[PLOTS] Saved to results/figures/")
            except Exception as e:
                print(f"[PLOTS] render failed: {e}")
        else:
            print("[PLOTS] plots.py not importable — skipping figure render.")

    print("[DONE] Results saved to results/")


if __name__ == "__main__":
    main()

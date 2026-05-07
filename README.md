# CUDA Kernel Auto-Tuner — Systematic GPU Optimization via Compiler-Guided Parameter Space Exploration

Automatically tunes naive CUDA kernels by generating, compiling, and statistically benchmarking parameter variants to find the fastest configuration. Targets the NVIDIA RTX 2070 (sm_75, Turing).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          autotune.py                                │
│                       (orchestration)                               │
└───┬─────────┬──────────┬──────────┬─────────┬──────────────────────┘
    │         │          │          │         │
    ▼         ▼          ▼          ▼         ▼
parser.py  generator.py  verifier.py  search.py  benchmark.py
(AST/regex) (templates) (NumPy ref) (grid/BO/SHA) (stats+CUDA)
    │                                              │
    │              ┌────────────────────────────────┤
    │              │                               │
    ▼              ▼                               ▼
KernelProfile   .cu files              BenchmarkResult
(search space)  (generated)            (mean, std, CI, p-value)
                                               │
                         ┌─────────────────────┤
                         │          │          │
                         ▼          ▼          ▼
                    roofline.py  occupancy.py  ptx_analysis.py
                    (AI, GFLOP/s) (occ%, regs)  (instr counts)
                                               │
                                               ▼
                                          reporter.py
                                     (Markdown + terminal table)
                                               │
                                               ▼
                                        cuda_graph.py
                                     (graph launch overhead)
                                               │
                                               ▼
                                          plots.py
                                  (Phase D: speedup bars,
                                  roofline figures)
```

---

## Hardware Requirements & Setup

- **GPU**: NVIDIA RTX 2070 (sm_75) or compatible Turing/Ampere GPU
- **CUDA**: 11.0 or later (`nvcc` must be on `PATH`)
- **Python**: 3.10 or later

```bash
# Clone
git clone https://github.com/parag050701/Cuda-Optimization
cd Cuda-Optimization

# Install Python dependencies
pip install -r requirements.txt

# Optional: libclang for AST-based parser (Phase A / Upgrade 7)
pip install libclang

# Optional: cupy for cuBLAS correctness reference (Upgrade 5)
pip install cupy-cuda12x   # or cupy-cuda11x

# Optional: ncu (Nsight Compute) on PATH for Phase C measured DRAM bytes
ncu --version
```

---

## Quick Start

```bash
# Baseline benchmark only
python autotune.py --baseline-only

# Tune matmul with default grid search
python autotune.py --kernel=matmul

# Tune all kernels with Bayesian optimisation
python autotune.py --kernel=all --strategy=bayesian

# Enable all analysis features
python autotune.py --kernel=softmax --strategy=sha --ptx-analysis --cuda-graphs

# Skip verification for speed (warns loudly)
python autotune.py --kernel=reduction --skip-verification --workers=4

# Phase A-D: libclang-driven search space + in-process CHECK + ncu + plots
python autotune.py --kernel=matmul --use-libclang --with-correctness --ncu --plots
```

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel` | `matmul` | Kernel to tune: `matmul` \| `softmax` \| `reduction` \| `layernorm` \| `attention` \| `all` |
| `--baseline-only` | off | Run baseline benchmarks only, skip tuning |
| `--strategy` | `grid` | Search strategy: `grid` \| `bayesian` \| `sha` |
| `--workers` | `2` | Parallel compile+benchmark workers |
| `--warmup` | `5` | GPU warmup iterations per variant |
| `--samples` | `30` | Statistical samples per variant (CUDA events) |
| `--skip-verification` | off | Skip NumPy correctness check (faster) |
| `--ptx-analysis` | off | Generate PTX and extract instruction metrics |
| `--cuda-graphs` | off | Measure CUDA Graph vs event-based launch overhead |
| `--iters` | — | Alias for `--samples` (backward compatibility) |
| `--use-libclang` | off | **Phase A** — drive search space from libclang AST parse of `baseline_kernels.cu` |
| `--with-correctness` | off | **Phase B** — emit in-process `CHECK` driver in each variant (compares against the naive reference embedded in the same binary) |
| `--ncu` | off | **Phase C** — profile the best variant with Nsight Compute, embed DRAM/cache counters in the tuning JSON |
| `--plots` | off | **Phase D** — render figures (speedup bars, roofline plot) via `plots.py` after tuning |

---

## Optimization Dimensions

| Parameter | Values | Applies to |
|-----------|--------|-----------|
| `block_size` | 64, 128, 192, 256 | all kernels |
| `tile_x`, `tile_y` | 16, 32 (forced equal for matmul) | matmul |
| `unroll` | 1, 2, 4, 8 | all kernels |
| `transpose_b` | True, False | matmul |
| `warp_shuffle` | True, False | reduction, softmax |
| `reg_tile` | 1, 2, 4 | matmul (Phase B: per-thread register blocking — RT outputs per thread) |

After validity pruning: ~96 valid configs across the four core kernels.

---

## Statistical Rigour (Upgrade 1)

Every variant timing uses:
- **5 warmup** runs before measuring (GPU thermal stabilisation)
- **30 CUDA-event timing samples** per variant
- **mean, std, 95% CI** (1.96 × std / √30)
- **Welch's t-test** (scipy, `equal_var=False`) vs baseline
- A variant is flagged as "winning" **only if** `p < 0.05` AND `speedup > 1.0`

---

## Roofline Model (Upgrade 2)

The [Roofline model](https://en.wikipedia.org/wiki/Roofline_model) characterises whether a kernel is memory-bandwidth-limited or compute-limited.

**RTX 2070 constants:**

| Constant | Value |
|----------|-------|
| Peak compute | 7.5 TFLOP/s (FP32) |
| Peak memory bandwidth | 448 GB/s (GDDR6) |
| Ridge point | 7500 / 448 ≈ **16.74 FLOP/byte** |

A kernel with arithmetic intensity < 16.74 FLOP/byte is **memory-bound**; above it is **compute-bound**.

**Per-kernel models:**

| Kernel | FLOP count | Bytes moved | Typical AI |
|--------|-----------|-------------|-----------|
| matmul N×N | 2N³ | 3N²×4 | 2N/12 ≈ 170 for N=1024 |
| softmax R×C | 5RC | 2RC×4 | 0.625 — memory-bound |
| reduction N | 2N | N×4 | 0.5 — memory-bound |
| layernorm R×C | 8RC | 3RC×4 | 0.667 — memory-bound |

When `ncu` is available (`--ncu`), DRAM bytes from `dram__bytes.sum` give a second set of measured numbers (typically higher bandwidth and lower AI than the analytic lower bound, reflecting actual cache-miss traffic).

---

## Search Strategy Comparison (Upgrade 3)

| Strategy | Evals to optimum | Exploration quality | Recommended use case |
|----------|-----------------|--------------------|--------------------|
| **Grid** | All (~160) | Exhaustive | Small spaces, reproducibility |
| **Bayesian** | ~40 | High (GP surrogate + EI) | Large spaces, expensive evals |
| **SHA** | ~160 → halved each round | Adaptive elimination | Medium spaces, quick screening |

- **Grid** (`--strategy=grid`): Exhaustive Cartesian product. Guaranteed to find the global optimum within the defined space.
- **Bayesian** (`--strategy=bayesian`): Gaussian Process with Expected Improvement acquisition (scikit-optimize). 40 evaluations typically reach ≥95% of the grid optimum.
- **SHA** (`--strategy=sha`): Starts with 80 random configs, runs 5 iterations each, halves survivors each round: 80 → 40 → 20 → 10 → 5 → 1.

---

## Occupancy Analysis (Upgrade 4)

Parses `ptxas -v` stderr to extract per-kernel resource usage and compute theoretical occupancy for sm_75:

| Limit | sm_75 hardware cap |
|-------|-------------------|
| Registers per block | 65,536 |
| Shared memory per block | 49,152 bytes |
| Max warps per SM | 32 |
| Max blocks per SM | 16 |

**Occupancy** = `min(active_warps_from_regs, active_warps_from_smem, max_warps_per_sm) / 32`

Register spilling (spill_stores > 0) is flagged as a performance penalty.

---

## Correctness Verification (Upgrade 5)

Every variant is verified before benchmarking (always-on by default):

| Kernel | Reference | Tolerance |
|--------|-----------|-----------|
| matmul | `np.matmul` (or `cp.matmul` if cupy available) | 1e-3 max abs diff |
| softmax | `np.exp` + normalise | 1e-5 |
| reduction | `np.sum` (partial sums) | 1e-3 |
| layernorm | manual mean/var/normalize | 1e-4 |

Failed variants are skipped and logged to `results/{kernel}_failures.json`.
Use `--skip-verification` to bypass (warns loudly).

---

## PTX / SASS Analysis (Upgrade 6)

Enabled via `--ptx-analysis`. Adds one extra `nvcc -ptx` compilation per analysed source:

- **PTX metrics**: total instructions, `ld.global`, `st.global`, `fma/mad`, branches
- **Derived**: `compute_ratio = fma / total`, `memory_ratio = (ld+st) / total`
- **SASS** via `cuobjdump --dump-sass` (skipped gracefully if not on PATH)
- Baseline vs best delta saved to `results/{kernel}_ptx_analysis.json`

---

## CUDA Graph Support (Upgrade 9)

Enabled via `--cuda-graphs`. Measures kernel graph-launch overhead vs standard event-based launch:

```
event_launch_ms  — standard cudaEventRecord timing
graph_launch_ms  — CUDA Graph replay timing
overhead_reduction_pct = (event - graph) / event * 100
```

CUDA Graphs eliminate per-launch CPU overhead, important for production inference serving (e.g. TensorRT, PyTorch `torch.cuda.CUDAGraph`). Uses `ctypes` to call `libcudart.so` directly — no PyCUDA required.

---

## Phase A–D Additions

These build on the Upgrade 1-9 stack and ship as opt-in CLI flags:

### Phase A — libclang compiler frontend (`--use-libclang`)

`parser.py` ships two backends:

- **`LibclangKernelParser`** (primary) — AST traversal via `clang.cindex`. Walks `FUNCTION_DECL` cursors with the `__global__` attribute and accumulates `MemoryAccessPattern` / loop-depth / shared-mem usage from the AST itself, not from regex over text. Detects strided indexing, warp-shuffle calls, register-resident vs shared-memory accumulators.
- **`RegexKernelParser`** (fallback) — pattern-matching heuristics; takes over when `clang` isn't importable, when `CUDA_AUTOTUNER_NO_LIBCLANG` is set, or when the libclang TU yields zero kernels.

Both produce the same `KernelProfile` shape with a `backend` tag (`"libclang"` | `"regex"` | `"hardcoded"`) so downstream code is agnostic.

### Phase B — Register-blocked matmul + in-process correctness (`--with-correctness`)

Three additions in `generator.py`:

- **`MATMUL_REGBLOCK_TEMPLATE`** — each thread computes RT outputs in the column direction, reusing one A-row load across RT B-column loads. Selected when `params['reg_tile'] > 1`.
- **`LAYERNORM_SINGLEPASS_TEMPLATE`** — single-pass layernorm accumulating sum(x) and sum(x²) in parallel, faster than the three-pass variant.
- **`BENCHMARK_MAIN_WITH_CORRECTNESS`** — driver variant that embeds the naive reference kernel from `naive_kernels.cuh`, runs it on the same inputs, and emits a `CHECK <tag> <max_rel_err> <pass>` line for the autotune driver. Variants exceeding the 1% tolerance are dropped automatically — a faster, in-process alternative to the NumPy `verifier.py` path.

### Phase C — Nsight Compute integration (`--ncu`)

`benchmark.collect_ncu_metrics()` runs `ncu --metrics dram__bytes.sum,...` on the best variant after tuning, parses the CSV, and embeds the counters in `<kernel>_tuning.json`. Supports `kernel_regex=".*_opt"` and `launch_count=3` so profiling only the optimized kernel is fast.

### Phase D — Figure generation (`--plots`)

`src/plots.py` reads the per-kernel tuning JSONs and renders three figures into `results/figures/`:
- `ms_bars.png` — baseline vs best mean-ms per kernel
- `speedup_factor.png` — speedup factor with the 3× target line
- `roofline.png` — RTX 2070 roofline with each kernel's analytic and ncu-measured points

---

## Output Files

| File | Description |
|------|-------------|
| `results/baseline.json` | Baseline timing statistics (mean, std, CI per kernel) |
| `results/{kernel}_tuning.json` | All variant results (top 50), best config, `ncu_metrics`, roofline |
| `results/{kernel}_report.md` | Full Markdown tuning report |
| `results/{kernel}_convergence.json` | Eval-number vs best-ms curve (Bayesian/SHA) |
| `results/{kernel}_failures.json` | Correctness failures (if any) |
| `results/{kernel}_ptx_analysis.json` | PTX instruction delta (if `--ptx-analysis`) |
| `results/live_progress.json` | Live tuning progress (used by streamlit dashboard) |
| `results/generated/*.cu` | Generated kernel variant sources |
| `results/bins/*.exe` | Compiled variant binaries |
| `results/figures/*.png` | Phase-D rendered figures (if `--plots`) |

---

## Sample Terminal Output

```
CUDA Kernel Auto-Tuner  |  RTX 2070  |  sm_75

[BASELINE] Compiling and benchmarking naive kernels ...
[COMPILE] benchmark_runner.cu ... OK
[RUN] baseline  warmup=5  samples=30 ...

TIMING matmul_1024       4.8374 4.7901 4.8891 30
TIMING softmax_1024x4096 1.4395 1.4211 1.4601 30
TIMING reduction_1M      0.0886 0.0879 0.0901 30
TIMING layernorm_512x2048 0.3788 0.3752 0.3811 30

[GENERATE] 64 variants for 'matmul' ...
[STRATEGY] GridSearchStrategy  warmup=5  samples=30  workers=2
[VERIFY] Correctness checking enabled (numpy reference)

  [ 64/ 64] 2.103ms OK            ETA 0s

[VERIFY] 64 checked — 63 passed — 1 failed

==========================================================================
  TUNING COMPLETE — MATMUL  |  strategy=grid  |  64 variants
==========================================================================
  #    mean_ms       ±CI   speedup   p-val  sig   params
  ----------------------------------------------------------------------
  1    2.103ms  ±0.012ms    2.30x   0.000   OK   blk=64 tx=32 ty=32 unroll=4
  2    2.187ms  ±0.015ms    2.21x   0.000   OK   blk=64 tx=32 ty=32 unroll=2
  3    2.314ms  ±0.018ms    2.09x   0.001   OK   blk=128 tx=32 ty=32 unroll=4
  ...
==========================================================================

  BEST: 2.103ms  |  baseline: 4.837ms  |  speedup: 2.30x (statistically significant)
  BOUND: memory-bound | efficiency: 3.2% | AI: 170.67 FLOP/byte
  OCCUPANCY: 75.0% | regs: 32 | smem: 8192B
==========================================================================
```

---

## Architecture decisions

- **libclang over a handwritten parser.** AST traversal accurately detects CUDA features (`__global__`, `__shared__`, `__shfl_*`, loop nesting) where regex heuristics over-count sequential loops as nested and miss subscripted reductions. A regex path is kept as a fallback so the tuner degrades gracefully when libclang isn't available.
- **In-process correctness.** Comparing against a naive reference kernel inside the same binary sidesteps any cross-process seed/state drift and keeps the search loop self-contained. The standalone NumPy `verifier.py` remains the default; the in-process `CHECK` driver is opt-in via `--with-correctness`.
- **Grid search for small spaces, BO/SHA for larger.** With 96 variants per kernel exhaustive grid dominates on wall-clock and simplicity; Bayesian/SHA are wired in for users who want to extend the search space.
- **No tensor-core / WMMA path.** Turing's sm_75 only supports fp16 WMMA, not fp32. A meaningful WMMA implementation would be a separate project.

---

## References

1. **Roofline Model**: Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Communications of the ACM, 52(4), 65–76.
2. **Bayesian Optimization**: Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. Advances in Neural Information Processing Systems (NeurIPS), 25.
3. **Successive Halving**: Jamieson, K., & Talwalkar, A. (2016). *Non-stochastic Best Arm Identification and Hyperparameter Optimization*. Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS).

---

## Project Structure

```
Cuda-Optimization/
├── autotune.py                    # Main entry point (Phase A-D flags wired here)
├── requirements.txt
├── README.md
└── src/
    ├── parser.py                  # Phase A / Upgrade 7: libclang + regex kernel parser
    ├── generator.py               # Template-based .cu code generator
    │                              #   (matmul tile / matmul regblock / softmax /
    │                              #    reduction / layernorm 3-pass /
    │                              #    layernorm single-pass / Phase-B CHECK driver)
    ├── benchmark.py               # Upgrade 1: statistical benchmarking + BenchmarkResult
    │                              #   (Phase C: collect_ncu_metrics with kernel_regex)
    ├── roofline.py                # Upgrade 2: roofline model
    ├── search.py                  # Upgrade 3: Grid / Bayesian / SHA strategies
    ├── occupancy.py               # Upgrade 4: ptxas-based occupancy analysis
    ├── verifier.py                # Upgrade 5: NumPy correctness verification
    ├── ptx_analysis.py            # Upgrade 6: PTX/SASS instruction analysis
    ├── reporter.py                # Upgrade 8: Markdown reports + terminal tables
    ├── cuda_graph.py              # Upgrade 9: CUDA Graph benchmark
    ├── plots.py                   # Phase D: speedup / roofline figure render
    └── kernels/
        ├── baseline_kernels.cu    # 4 intentionally naive kernels
        ├── naive_kernels.cuh      # Header form of the naive kernels (used by
        │                          # the Phase-B in-process CHECK driver)
        └── benchmark_runner.cu    # Standalone baseline profiler
```

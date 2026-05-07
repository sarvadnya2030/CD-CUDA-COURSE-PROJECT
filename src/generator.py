"""
generator.py — template-based CUDA code generator.

Takes a kernel profile + a concrete parameter configuration and emits a .cu
file containing the optimised kernel and a self-contained benchmark main().

Two driver flavours are supported:

  Default ("statistical" driver, friend's path)
    Emits SAMPLE <tag> <ms>     — one per measured iteration
    Emits TIMING <tag> <mean> <min> <max> <iters>

  Correctness driver (Phase B, with_correctness=True)
    Embeds the naive reference kernel from src/kernels/naive_kernels.cuh,
    runs it on the same inputs, and emits an extra
        CHECK <tag> <max_rel_err> <pass>
    line so the autotune driver can drop variants that exceed tolerance.

Kernel templates available:
  MATMUL_TEMPLATE          — shared-memory tiled matmul (square TILE_X×TILE_X).
  MATMUL_REGBLOCK_TEMPLATE — register-blocked matmul (1×RT per thread).
                             Selected when params['reg_tile'] > 1.
  REDUCTION_TEMPLATE       — block-stride reduction (with optional warp-shuffle tail).
  SOFTMAX_TEMPLATE         — row-wise softmax with shared-memory reduction.
  LAYERNORM_TEMPLATE       — three-pass layernorm (mean → var → normalise).
  LAYERNORM_SINGLEPASS_TEMPLATE — single-pass layernorm (sum + sum-of-squares
                             accumulated in parallel; faster but less stable).
"""

import itertools
from pathlib import Path
from typing import Any

from parser import KernelProfile, build_search_space

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
GEN_DIR     = RESULTS_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

ARCH = "sm_75"   # RTX 2070 (Turing)

# Absolute posix path to the naive-kernels header so nvcc can find it from
# the generated file's location regardless of working directory.  Used by
# the with-correctness driver path (mine, Phase B).
_NAIVE_HEADER = (ROOT / "src" / "kernels" / "naive_kernels.cuh").resolve().as_posix()

# Correctness tolerance used by the in-process comparison code.
CORRECTNESS_TOL = 1e-2


# ── Kernel templates ─────────────────────────────────────────────────────────

MATMUL_TEMPLATE = """\
/*
 * Generated matmul variant
 * block={block_size}x{block_size}  tile={tile_x}x{tile_y}  unroll={unroll}
 * transpose_B={transpose_b}
 */
#include <cuda_runtime.h>

#define TILE_X {tile_x}
#define TILE_Y {tile_y}
#define BLOCK  {block_size}

__global__ void matmul_opt(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int N)
{{
    __shared__ float As[TILE_Y][TILE_X];
    __shared__ float Bs[TILE_Y][TILE_X];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_Y + ty;
    int col = blockIdx.x * TILE_X + tx;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_X - 1) / TILE_X; ++t) {{
        if (row < N && t * TILE_X + tx < N)
            As[ty][tx] = A[row * N + t * TILE_X + tx];
        else
            As[ty][tx] = 0.0f;

{b_load}
        __syncthreads();

        #pragma unroll {unroll}
        for (int k = 0; k < TILE_X; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }}

    if (row < N && col < N)
        C[row * N + col] = sum;
}}
"""

# Register-blocked matmul (Phase B): each thread computes RT outputs in
# the column direction, reusing one A-row load across RT B-column loads.
MATMUL_REGBLOCK_TEMPLATE = """\
/*
 * Generated matmul variant (register-blocked, 1xRT per thread)
 * block={tile_x}x{tile_x}  reg_tile={reg_tile}  unroll={unroll}
 */
#include <cuda_runtime.h>

#define BLOCK {tile_x}
#define RT    {reg_tile}

__global__ void matmul_opt(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int N)
{{
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK * RT];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row      = blockIdx.y * BLOCK + ty;
    int col_base = blockIdx.x * (BLOCK * RT) + tx * RT;

    float sum[RT];
    #pragma unroll
    for (int r = 0; r < RT; ++r) sum[r] = 0.0f;

    for (int t = 0; t < (N + BLOCK - 1) / BLOCK; ++t) {{
        // Each thread loads 1 A element and RT B elements per tile step.
        int aK = t * BLOCK + tx;
        As[ty][tx] = (row < N && aK < N) ? A[row * N + aK] : 0.0f;

        int bRow = t * BLOCK + ty;
        #pragma unroll
        for (int r = 0; r < RT; ++r) {{
            int c = col_base + r;
            Bs[ty][tx * RT + r] =
                (bRow < N && c < N) ? B[bRow * N + c] : 0.0f;
        }}
        __syncthreads();

        #pragma unroll {unroll}
        for (int k = 0; k < BLOCK; ++k) {{
            float a = As[ty][k];
            #pragma unroll
            for (int r = 0; r < RT; ++r)
                sum[r] += a * Bs[k][tx * RT + r];
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int r = 0; r < RT; ++r) {{
        int c = col_base + r;
        if (row < N && c < N) C[row * N + c] = sum[r];
    }}
}}
"""

REDUCTION_TEMPLATE = """\
/*
 * Generated reduction variant
 * block={block_size}  unroll={unroll}  warp_shuffle={warp_shuffle}
 */
#include <cuda_runtime.h>

#define BLOCK {block_size}

__global__ void reduction_opt(const float* __restrict__ input,
                              float* __restrict__ output, int N)
{{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + tid;

    float val = 0.0f;
    if (gid < N)               val  = input[gid];
    if (gid + blockDim.x < N)  val += input[gid + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

{reduce_body}

    if (tid == 0) output[blockIdx.x] = sdata[0];
}}
"""

SOFTMAX_TEMPLATE = """\
/*
 * Generated softmax variant
 * block={block_size}  unroll={unroll}
 */
#include <cuda_runtime.h>

#define BLOCK {block_size}

__global__ void softmax_opt(const float* __restrict__ input,
                            float* __restrict__ output,
                            int rows, int cols)
{{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    float thread_max = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        thread_max = fmaxf(thread_max, in_row[i]);
    smem[threadIdx.x] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }}
    float row_max = smem[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {{
        float e = expf(in_row[i] - row_max);
        out_row[i] = e;
        thread_sum += e;
    }}
    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }}
    float total = smem[0];

    #pragma unroll {unroll}
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        out_row[i] /= total;
}}
"""

LAYERNORM_TEMPLATE = """\
/*
 * Generated layernorm variant (three-pass: mean → var → normalize)
 * block={block_size}  unroll={unroll}
 */
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK {block_size}

__global__ void layernorm_opt(const float* __restrict__ input,
                              float* __restrict__ output,
                              const float* __restrict__ gamma,
                              const float* __restrict__ beta,
                              int rows, int cols, float eps)
{{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    // Pass 1: parallel mean
    float thread_sum = 0.0f;
    #pragma unroll {unroll}
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        thread_sum += in_row[i];
    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }}
    float mean = smem[0] / cols;
    __syncthreads();

    // Pass 2: parallel variance
    float thread_var = 0.0f;
    #pragma unroll {unroll}
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {{
        float d = in_row[i] - mean;
        thread_var += d * d;
    }}
    smem[threadIdx.x] = thread_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }}
    float inv_std = rsqrtf(smem[0] / cols + eps);
    __syncthreads();

    // Pass 3: normalize + scale + shift
    #pragma unroll {unroll}
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        out_row[i] = gamma[i] * (in_row[i] - mean) * inv_std + beta[i];
}}
"""

# Single-pass layernorm (Phase B): accumulates sum(x) and sum(x^2) in
# parallel and derives mean / variance from those reductions.  Faster
# than the three-pass variant but more sensitive to fp32 precision.
LAYERNORM_SINGLEPASS_TEMPLATE = """\
/*
 * Generated layernorm variant (single-pass: sum + sum-of-squares)
 * block={block_size}  unroll={unroll}
 */
#include <cuda_runtime.h>

#define BLOCK {block_size}

__global__ void layernorm_opt(const float* __restrict__ input,
                              float* __restrict__ output,
                              const float* __restrict__ gamma,
                              const float* __restrict__ beta,
                              int rows, int cols, float eps)
{{
    extern __shared__ float smem[];         // size = 2 * blockDim.x floats
    float* s_sum = smem;
    float* s_sq  = smem + blockDim.x;

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {{
        float v = in_row[i];
        local_sum += v;
        local_sq  += v * v;
    }}
    s_sum[threadIdx.x] = local_sum;
    s_sq [threadIdx.x] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) {{
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sq [threadIdx.x] += s_sq [threadIdx.x + s];
        }}
        __syncthreads();
    }}

    float mean    = s_sum[0] / (float)cols;
    float var     = s_sq [0] / (float)cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    #pragma unroll {unroll}
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        out_row[i] = gamma[i] * (in_row[i] - mean) * inv_std + beta[i];
}}
"""

# ── Benchmark drivers ─────────────────────────────────────────────────────

# Statistical driver (default): emits SAMPLE + TIMING lines.  Used by
# benchmark.run_statistical_benchmark + reporter.py.
BENCHMARK_MAIN = """\

/* ── Benchmark driver ────────────────────────────────────────────────── */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define CUDA_CHECK(call) \\
    do {{ cudaError_t e=(call); if(e!=cudaSuccess){{ \\
        fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); \\
        exit(1); }} }} while(0)

static void fill_random(float* d, int n) {{
    float* h=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++) h[i]=(float)rand()/RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d,h,n*sizeof(float),cudaMemcpyHostToDevice));
    free(h);
}}

static void fill_ones(float* d, int n) {{
    float* h=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++) h[i]=1.0f;
    CUDA_CHECK(cudaMemcpy(d,h,n*sizeof(float),cudaMemcpyHostToDevice));
    free(h);
}}

int main(void) {{
    int warmup = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    int iters  = getenv("ITERS")  ? atoi(getenv("ITERS"))  : 30;

    {setup_code}

    /* Warmup */
    for(int i=0;i<warmup;i++) {{ {kernel_launch} }}
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total=0.0f, mn=FLT_MAX, mx=0.0f;

    for(int i=0;i<iters;i++) {{
        CUDA_CHECK(cudaEventRecord(start));
        {kernel_launch}
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms=0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
        printf("SAMPLE {variant_tag} %.4f\\n", ms);
        total+=ms;
        if(ms<mn) mn=ms;
        if(ms>mx) mx=ms;
    }}

    printf("TIMING {variant_tag} %.4f %.4f %.4f %d\\n",
           total/iters, mn, mx, iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    {cleanup_code}
    return 0;
}}
"""

# With-correctness driver (Phase A-D path): runs the naive reference kernel
# from naive_kernels.cuh on the same inputs, then compares output to the
# optimized result and emits a CHECK line.  Selected via with_correctness=True.
BENCHMARK_MAIN_WITH_CORRECTNESS = """\

/* ── Benchmark driver with in-process correctness check ─────────────── */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "{naive_header}"

#define CUDA_CHECK(call) \\
    do {{ cudaError_t e=(call); if(e!=cudaSuccess){{ \\
        fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); \\
        exit(1); }} }} while(0)

static void fill_random(float* d, int n) {{
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) h[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}}

static void fill_ones(float* d, int n) {{
    float* h=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++) h[i]=1.0f;
    CUDA_CHECK(cudaMemcpy(d,h,n*sizeof(float),cudaMemcpyHostToDevice));
    free(h);
}}

int main(void) {{
    int warmup = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    int iters  = getenv("ITERS")  ? atoi(getenv("ITERS"))  : 30;

    {setup_code}

    /* Run naive reference for correctness comparison */
    {launch_ref}
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Warmup on optimized */
    for (int i = 0; i < warmup; ++i) {{ {kernel_launch} }}
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timing (also emits SAMPLE lines for statistical driver compat) */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total = 0.0f, mn = FLT_MAX, mx = 0.0f;
    for (int i = 0; i < iters; ++i) {{
        CUDA_CHECK(cudaEventRecord(start));
        {kernel_launch}
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("SAMPLE {variant_tag} %.4f\\n", ms);
        total += ms;
        if (ms < mn) mn = ms;
        if (ms > mx) mx = ms;
    }}
    printf("TIMING {variant_tag} %.4f %.4f %.4f %d\\n",
           total / iters, mn, mx, iters);

    /* Correctness check: compare d_out against d_ref */
    {correctness_block}

    {cleanup_code}
    return 0;
}}
"""


# ── Code-generation helpers ───────────────────────────────────────────────

def _matmul_b_load(transpose_b: bool, tile_x: int = 0, tile_y: int = 0) -> str:
    if transpose_b:
        return (
            "        if (t * TILE_Y + ty < N && col < N)\n"
            "            Bs[ty][tx] = B[col * N + t * TILE_Y + ty];\n"
            "        else\n"
            "            Bs[ty][tx] = 0.0f;"
        )
    return (
        "        if (t * TILE_Y + ty < N && col < N)\n"
        "            Bs[ty][tx] = B[(t * TILE_Y + ty) * N + col];\n"
        "        else\n"
        "            Bs[ty][tx] = 0.0f;"
    )


def _reduction_body(block_size: int, warp_shuffle: bool) -> str:
    if warp_shuffle:
        return (
            "    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {\n"
            "        if (tid < s) sdata[tid] += sdata[tid + s];\n"
            "        __syncthreads();\n"
            "    }\n"
            "    if (tid < 32) {\n"
            "        float v = sdata[tid];\n"
            "        v += __shfl_down_sync(0xffffffff, v, 16);\n"
            "        v += __shfl_down_sync(0xffffffff, v,  8);\n"
            "        v += __shfl_down_sync(0xffffffff, v,  4);\n"
            "        v += __shfl_down_sync(0xffffffff, v,  2);\n"
            "        v += __shfl_down_sync(0xffffffff, v,  1);\n"
            "        if (tid == 0) sdata[0] = v;\n"
            "    }"
        )
    return (
        "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
        "        if (tid < s) sdata[tid] += sdata[tid + s];\n"
        "        __syncthreads();\n"
        "    }"
    )


def _variant_tag(kernel: str, params: dict) -> str:
    parts = [kernel]
    for k, v in sorted(params.items()):
        parts.append(f"{k}{v}")
    return "_".join(str(p) for p in parts)


# ── Correctness blocks (Phase B: in-process numerical compare) ────────────

def _elementwise_check_block(n_out_expr: str, tag: str) -> str:
    """Element-wise max relative error check (matmul / softmax / layernorm)."""
    return f"""{{
        int n = (int)({n_out_expr});
        float* h_opt = (float*)malloc(n * sizeof(float));
        float* h_ref = (float*)malloc(n * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_opt, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ref, d_ref, n * sizeof(float), cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < n; ++i) {{
            float a = h_opt[i], b = h_ref[i];
            float denom = fmaxf(fabsf(a), fabsf(b));
            if (denom > 1e-6f) {{
                float e = fabsf(a - b) / denom;
                if (e > max_err) max_err = e;
            }}
        }}
        int pass = (max_err < {CORRECTNESS_TOL}f);
        printf("CHECK {tag} %.4e %d\\n", max_err, pass);
        free(h_opt); free(h_ref);
    }}"""


def _scalar_sum_check_block(n_opt_expr: str, n_ref_expr: str, tag: str) -> str:
    """Scalar-sum compare (reduction; opt and ref grids may differ)."""
    return f"""{{
        int n_opt = (int)({n_opt_expr});
        int n_ref = (int)({n_ref_expr});
        float* h_opt = (float*)malloc(n_opt * sizeof(float));
        float* h_ref = (float*)malloc(n_ref * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_opt, d_out, n_opt * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ref, d_ref, n_ref * sizeof(float), cudaMemcpyDeviceToHost));
        double s_opt = 0.0, s_ref = 0.0;
        for (int i = 0; i < n_opt; ++i) s_opt += h_opt[i];
        for (int i = 0; i < n_ref; ++i) s_ref += h_ref[i];
        double denom = fmax(fabs(s_opt), fabs(s_ref));
        double rel = (denom > 1e-6) ? fabs(s_opt - s_ref) / denom : 0.0;
        int pass = (rel < {CORRECTNESS_TOL});
        printf("CHECK {tag} %.4e %d\\n", (float)rel, pass);
        free(h_opt); free(h_ref);
    }}"""


# ── Per-kernel generators ─────────────────────────────────────────────────

def generate_matmul(params: dict, matrix_size: int = 1024,
                    with_correctness: bool = False) -> str:
    """
    Generate a complete matmul variant .cu file.

    When params['reg_tile'] > 1, the register-blocked template is used
    (each thread emits RT contiguous columns of C from a single A-row load).
    Otherwise the shared-memory tiled template is used.
    """
    reg_tile = params.get("reg_tile", 1)
    N        = matrix_size
    tag      = _variant_tag("matmul", params)

    if reg_tile <= 1:
        b_load = _matmul_b_load(params.get("transpose_b", False),
                                params.get("tile_x", 16),
                                params.get("tile_y", 16))
        kernel = MATMUL_TEMPLATE.format(b_load=b_load, **params)
        bs     = params.get("tile_x", 16)
        launch = (f"matmul_opt<<<dim3((N+{bs}-1)/{bs},(N+{bs}-1)/{bs}),"
                  f"dim3({bs},{bs})>>>(A,B,C,N);")
    else:
        kernel = MATMUL_REGBLOCK_TEMPLATE.format(**params)
        bs = params.get("tile_x", 16)
        # Grid covers N×N output: y step = BLOCK rows, x step = BLOCK*RT cols
        launch = (f"matmul_opt<<<dim3((N + {bs}*{reg_tile} - 1)/({bs}*{reg_tile}),"
                  f"(N+{bs}-1)/{bs}),"
                  f"dim3({bs},{bs})>>>(A,B,C,N);")

    if with_correctness:
        setup = (
            f"int N = {N};\n"
            f"    float *A, *B, *C, *d_ref;\n"
            f"    CUDA_CHECK(cudaMalloc(&A,     N*N*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&B,     N*N*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&C,     N*N*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&d_ref, N*N*sizeof(float)));\n"
            f"    fill_random(A, N*N);\n"
            f"    fill_random(B, N*N);\n"
            f"    float* d_out = C;"
        )
        cleanup    = "cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(d_ref);"
        ref_launch = "launch_matmul_naive(A, B, d_ref, N, 16);"
        check      = _elementwise_check_block("N*N", tag)
        return kernel + BENCHMARK_MAIN_WITH_CORRECTNESS.format(
            naive_header=_NAIVE_HEADER,
            variant_tag=tag,
            setup_code=setup,
            launch_ref=ref_launch,
            kernel_launch=launch,
            correctness_block=check,
            cleanup_code=cleanup,
        )

    setup   = (f"int N={N}; float *A,*B,*C;\n"
               f"    CUDA_CHECK(cudaMalloc(&A,N*N*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&B,N*N*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&C,N*N*4));\n"
               f"    fill_random(A,N*N); fill_random(B,N*N);")
    cleanup = "cudaFree(A); cudaFree(B); cudaFree(C);"
    return kernel + BENCHMARK_MAIN.format(
        variant_tag=tag, setup_code=setup,
        kernel_launch=launch, cleanup_code=cleanup,
    )


def generate_reduction(params: dict, matrix_size: int = 1024,
                       with_correctness: bool = False) -> str:
    """Generate a complete reduction variant .cu file."""
    body   = _reduction_body(params["block_size"], params["warp_shuffle"])
    kernel = REDUCTION_TEMPLATE.format(reduce_body=body, **params)

    blk = params["block_size"]
    tag = _variant_tag("reduction", params)

    if with_correctness:
        N = 1 << 20
        setup = (
            f"int N = {N};\n"
            f"    int grid_opt = (N + {blk}*2 - 1) / ({blk}*2);\n"
            f"    int grid_ref = (N + 256 - 1) / 256;\n"
            f"    float *in, *d_out, *d_ref;\n"
            f"    CUDA_CHECK(cudaMalloc(&in,    N*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&d_out, grid_opt*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&d_ref, grid_ref*sizeof(float)));\n"
            f"    fill_random(in, N);"
        )
        cleanup    = "cudaFree(in); cudaFree(d_out); cudaFree(d_ref);"
        launch     = (f"reduction_opt<<<grid_opt, {blk}, "
                      f"{blk}*sizeof(float)>>>(in, d_out, N);")
        ref_launch = "launch_reduction_naive(in, d_ref, N, 256);"
        check      = _scalar_sum_check_block("grid_opt", "grid_ref", tag)
        return kernel + BENCHMARK_MAIN_WITH_CORRECTNESS.format(
            naive_header=_NAIVE_HEADER,
            variant_tag=tag,
            setup_code=setup,
            launch_ref=ref_launch,
            kernel_launch=launch,
            correctness_block=check,
            cleanup_code=cleanup,
        )

    N = matrix_size * 1024  # keep large regardless of matrix_size scale
    setup   = (f"int N={N}; int grid=(N+{blk*2}-1)/({blk*2});\n"
               f"    float *in,*out;\n"
               f"    CUDA_CHECK(cudaMalloc(&in,N*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&out,grid*4));\n"
               f"    fill_random(in,N);")
    launch  = f"reduction_opt<<<grid,{blk},{blk}*4>>>(in,out,N);"
    cleanup = "cudaFree(in); cudaFree(out);"
    return kernel + BENCHMARK_MAIN.format(
        variant_tag=tag, setup_code=setup,
        kernel_launch=launch, cleanup_code=cleanup,
    )


def generate_softmax(params: dict, matrix_size: int = 1024,
                     with_correctness: bool = False) -> str:
    """Generate a complete softmax variant .cu file."""
    kernel = SOFTMAX_TEMPLATE.format(**params)

    blk = params["block_size"]
    tag = _variant_tag("softmax", params)

    if with_correctness:
        rows, cols = 1024, 4096
        setup = (
            f"int rows = {rows}, cols = {cols};\n"
            f"    float *in, *d_out, *d_ref;\n"
            f"    CUDA_CHECK(cudaMalloc(&in,    rows*cols*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&d_out, rows*cols*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&d_ref, rows*cols*sizeof(float)));\n"
            f"    fill_random(in, rows*cols);"
        )
        cleanup    = "cudaFree(in); cudaFree(d_out); cudaFree(d_ref);"
        launch     = (f"softmax_opt<<<rows, {blk}, "
                      f"{blk}*sizeof(float)>>>(in, d_out, rows, cols);")
        ref_launch = "launch_softmax_naive(in, d_ref, rows, cols);"
        check      = _elementwise_check_block("rows*cols", tag)
        return kernel + BENCHMARK_MAIN_WITH_CORRECTNESS.format(
            naive_header=_NAIVE_HEADER,
            variant_tag=tag,
            setup_code=setup,
            launch_ref=ref_launch,
            kernel_launch=launch,
            correctness_block=check,
            cleanup_code=cleanup,
        )

    rows, cols = matrix_size, 4096
    setup   = (f"int rows={rows},cols={cols}; float *in,*out;\n"
               f"    CUDA_CHECK(cudaMalloc(&in,rows*cols*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&out,rows*cols*4));\n"
               f"    fill_random(in,rows*cols);")
    launch  = f"softmax_opt<<<rows,{blk},{blk}*4>>>(in,out,rows,cols);"
    cleanup = "cudaFree(in); cudaFree(out);"
    return kernel + BENCHMARK_MAIN.format(
        variant_tag=tag, setup_code=setup,
        kernel_launch=launch, cleanup_code=cleanup,
    )


def generate_layernorm(params: dict, matrix_size: int = 1024,
                       with_correctness: bool = False,
                       single_pass: bool = False) -> str:
    """
    Generate a complete layernorm variant .cu file.

    Args:
        single_pass: If True, use the Phase-B single-pass template
                     (sum + sum-of-squares accumulated in parallel).
                     Otherwise the three-pass (mean → var → norm) template.
    """
    template = LAYERNORM_SINGLEPASS_TEMPLATE if single_pass else LAYERNORM_TEMPLATE
    kernel   = template.format(**params)

    blk = params["block_size"]
    tag = _variant_tag("layernorm", params)

    if with_correctness:
        rows, cols = 512, 2048
        smem_words = 2 * blk if single_pass else blk
        setup = (
            f"int rows = {rows}, cols = {cols};\n"
            f"    float *in, *d_out, *d_ref, *gamma, *beta;\n"
            f"    CUDA_CHECK(cudaMalloc(&in,    rows*cols*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&d_out, rows*cols*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&d_ref, rows*cols*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&gamma, cols*sizeof(float)));\n"
            f"    CUDA_CHECK(cudaMalloc(&beta,  cols*sizeof(float)));\n"
            f"    fill_random(in, rows*cols);\n"
            f"    fill_random(gamma, cols);\n"
            f"    fill_random(beta, cols);"
        )
        cleanup = ("cudaFree(in); cudaFree(d_out); cudaFree(d_ref); "
                   "cudaFree(gamma); cudaFree(beta);")
        launch = (f"layernorm_opt<<<rows, {blk}, "
                  f"{smem_words}*sizeof(float)>>>"
                  f"(in, d_out, gamma, beta, rows, cols, 1e-5f);")
        ref_launch = "launch_layernorm_naive(in, d_ref, gamma, beta, rows, cols);"
        check      = _elementwise_check_block("rows*cols", tag)
        return kernel + BENCHMARK_MAIN_WITH_CORRECTNESS.format(
            naive_header=_NAIVE_HEADER,
            variant_tag=tag,
            setup_code=setup,
            launch_ref=ref_launch,
            kernel_launch=launch,
            correctness_block=check,
            cleanup_code=cleanup,
        )

    rows, cols = matrix_size // 2, 2048
    setup   = (f"int rows={rows},cols={cols}; float *in,*out,*gamma,*beta;\n"
               f"    CUDA_CHECK(cudaMalloc(&in,rows*cols*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&out,rows*cols*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&gamma,cols*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&beta,cols*4));\n"
               f"    fill_random(in,rows*cols);\n"
               f"    fill_ones(gamma,cols);\n"
               f"    CUDA_CHECK(cudaMemset(beta,0,cols*4));")
    launch  = f"layernorm_opt<<<rows,{blk},{blk}*4>>>(in,out,gamma,beta,rows,cols,1e-5f);"
    cleanup = "cudaFree(in); cudaFree(out); cudaFree(gamma); cudaFree(beta);"
    return kernel + BENCHMARK_MAIN.format(
        variant_tag=tag, setup_code=setup,
        kernel_launch=launch, cleanup_code=cleanup,
    )


GENERATORS = {
    "matmul":    generate_matmul,
    "softmax":   generate_softmax,
    "reduction": generate_reduction,
    "layernorm": generate_layernorm,
}


# ── Variant enumeration ───────────────────────────────────────────────────

def enumerate_variants(kernel: str, space: dict) -> list[tuple[dict, Path]]:
    """
    Expand the search space into a list of (params, output_path) tuples.

    Applies kernel-specific validity pruning (e.g. square tiles for matmul,
    transpose_b deduplication for register-blocked matmul).
    """
    param_keys = [k for k in space if not k.startswith("_")]
    param_vals = [space[k] for k in param_keys]

    variants:  list[tuple[dict, Path]] = []
    seen_tags: set[str] = set()

    for combo in itertools.product(*param_vals):
        params = dict(zip(param_keys, combo))

        if kernel == "matmul":
            # Matmul always uses square tiles.
            if params.get("tile_x") != params.get("tile_y"):
                continue
            if params.get("tile_x", 0) > params.get("block_size", 256):
                continue
            # transpose_b is meaningless in the regblock template
            # (B is loaded column-by-column anyway); drop those duplicates.
            if params.get("reg_tile", 1) > 1 and params.get("transpose_b", False):
                continue

        tag = _variant_tag(kernel, params)
        if tag in seen_tags:
            continue
        seen_tags.add(tag)

        path = GEN_DIR / f"{tag}.cu"
        variants.append((params, path))

    return variants


def write_variant(kernel: str, params: dict, out_path: Path,
                  matrix_size: int = 1024,
                  with_correctness: bool = False) -> None:
    """
    Write a generated .cu file for the given kernel and parameters.

    Args:
        kernel:           Kernel name.
        params:           Parameter dict.
        out_path:         Destination path for the generated file.
        matrix_size:      Override problem size (N for matmul, rows for
                          softmax/layernorm).
        with_correctness: If True, emit the Phase-B driver that runs the
                          naive reference kernel and prints a CHECK line.
    """
    gen = GENERATORS.get(kernel)
    if gen is None:
        raise ValueError(f"No generator for kernel {kernel!r}")
    src = gen(params, matrix_size=matrix_size, with_correctness=with_correctness)
    out_path.write_text(src, encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    kernel = sys.argv[1] if len(sys.argv) > 1 else "matmul"

    from parser import KernelProfile, MemoryAccessPattern, build_search_space
    dummy = KernelProfile(
        name=kernel, src_path=Path("."),
        block_dim=16, uses_shared=False,
        loop_depth=2, reduction_ops=["+="],
        memory=MemoryAccessPattern(
            has_strided_access=(kernel == "matmul"),
            has_reduction=(kernel in ("reduction", "softmax")),
        ),
    )
    space    = build_search_space(dummy)
    variants = enumerate_variants(kernel, space)
    print(f"Kernel: {kernel}  →  {len(variants)} variants")
    for params, path in variants[:3]:
        write_variant(kernel, params, path)
        print(f"  Written: {path.name}")
    if len(variants) > 3:
        print(f"  ... and {len(variants)-3} more")

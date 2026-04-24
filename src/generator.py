"""
generator.py — template-based CUDA code generator.

Takes a kernel profile + a concrete parameter configuration and emits a .cu
file containing the optimised kernel and a self-contained benchmark main().

The benchmark main() emits two line types:
  SAMPLE <tag> <ms>           — one per measured iteration (for statistics)
  TIMING <tag> <mean> <min> <max> <iters>  — aggregate summary (backward compat)
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
 * Generated layernorm variant
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

# ── Benchmark driver (emits SAMPLE + TIMING lines) ────────────────────────

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


# ── Code-generation helpers ───────────────────────────────────────────────

def _matmul_b_load(transpose_b: bool, tile_x: int, tile_y: int) -> str:
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


# ── Per-kernel generators ─────────────────────────────────────────────────

def generate_matmul(params: dict, matrix_size: int = 1024) -> str:
    """Generate a complete matmul variant .cu file."""
    b_load = _matmul_b_load(params["transpose_b"], params["tile_x"], params["tile_y"])
    kernel = MATMUL_TEMPLATE.format(b_load=b_load, **params)

    N   = matrix_size
    bs  = params["tile_x"]
    tag = _variant_tag("matmul", params)
    setup   = (f"int N={N}; float *A,*B,*C;\n"
               f"    CUDA_CHECK(cudaMalloc(&A,N*N*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&B,N*N*4));\n"
               f"    CUDA_CHECK(cudaMalloc(&C,N*N*4));\n"
               f"    fill_random(A,N*N); fill_random(B,N*N);")
    launch  = (f"matmul_opt<<<dim3((N+{bs}-1)/{bs},(N+{bs}-1)/{bs}),"
               f"dim3({bs},{bs})>>>(A,B,C,N);")
    cleanup = "cudaFree(A); cudaFree(B); cudaFree(C);"

    return kernel + BENCHMARK_MAIN.format(
        variant_tag=tag, setup_code=setup,
        kernel_launch=launch, cleanup_code=cleanup,
    )


def generate_reduction(params: dict, matrix_size: int = 1024) -> str:
    """Generate a complete reduction variant .cu file."""
    body   = _reduction_body(params["block_size"], params["warp_shuffle"])
    kernel = REDUCTION_TEMPLATE.format(reduce_body=body, **params)

    N    = matrix_size * 1024  # keep large regardless of matrix_size scale
    blk  = params["block_size"]
    tag  = _variant_tag("reduction", params)
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


def generate_softmax(params: dict, matrix_size: int = 1024) -> str:
    """Generate a complete softmax variant .cu file."""
    kernel = SOFTMAX_TEMPLATE.format(**params)

    rows = matrix_size; cols = 4096
    blk  = params["block_size"]
    tag  = _variant_tag("softmax", params)
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


def generate_layernorm(params: dict, matrix_size: int = 1024) -> str:
    """Generate a complete layernorm variant .cu file."""
    kernel = LAYERNORM_TEMPLATE.format(**params)

    rows = matrix_size // 2; cols = 2048
    blk  = params["block_size"]
    tag  = _variant_tag("layernorm", params)
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

    Applies kernel-specific validity pruning (e.g. square tiles for matmul).
    """
    param_keys = [k for k in space if not k.startswith("_")]
    param_vals = [space[k] for k in param_keys]

    variants: list[tuple[dict, Path]] = []
    for combo in itertools.product(*param_vals):
        params = dict(zip(param_keys, combo))

        if kernel == "matmul":
            if params["tile_x"] != params["tile_y"]:
                continue
            if params["tile_x"] > params["block_size"]:
                continue

        tag  = _variant_tag(kernel, params)
        path = GEN_DIR / f"{tag}.cu"
        variants.append((params, path))

    return variants


def write_variant(kernel: str, params: dict, out_path: Path,
                  matrix_size: int = 1024) -> None:
    """
    Write a generated .cu file for the given kernel and parameters.

    Args:
        kernel:      Kernel name.
        params:      Parameter dict.
        out_path:    Destination path for the generated file.
        matrix_size: Override problem size (N for matmul, rows for softmax/layernorm).
    """
    gen = GENERATORS.get(kernel)
    if gen is None:
        raise ValueError(f"No generator for kernel {kernel!r}")
    src = gen(params, matrix_size=matrix_size)
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

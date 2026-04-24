"""
attention.py — Multi-Head Attention kernel templates and search space.

Implements three attention variants with increasing optimization level:

  TILED  — shared-memory tiled QK^T + softmax + AV
           Reduces global reads; still materialises S×S score matrix in smem.

  FLASH  — Flash Attention-style single-pass kernel (Dao et al., 2022)
           Online numerically-stable softmax via (m, l, O) running accumulators.
           Never writes the S×S attention matrix to DRAM → lower memory traffic.

Problem size (fixed for benchmarking):
  S = 512 (sequence length)
  D =  64 (head dimension, typical for GPT-2 / BERT-base)
  scale = 1/√D ≈ 0.125

Search dimensions:
  seq_tile  [16, 32]     — K/V tile size (BC in flash attention literature)
  flash     [False, True] — which kernel variant to use
  unroll    [1, 2, 4]    — pragma unroll factor on inner loops

Reference:
  Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022).
  FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
  NeurIPS 2022.
"""

from __future__ import annotations

from pathlib import Path

# ── Fixed problem constants ────────────────────────────────────────────────
SEQ_LEN   = 512
D_HEAD    = 64
N_HEADS   = 8    # used by PyTorch integration
SCALE     = 1.0 / (D_HEAD ** 0.5)

GEN_DIR   = Path(__file__).parent.parent / "results" / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)


# ── CUDA kernel templates ──────────────────────────────────────────────────

# ── Tiled attention (explicit S×S score matrix in smem) ───────────────────
ATTENTION_TILED_TEMPLATE = r"""
/*
 * Tiled attention variant
 * seq_tile={seq_tile}  unroll={unroll}  flash=False
 * S={seq_len}  D={d_head}
 *
 * Each thread block processes one Q-row.
 * blockDim.x = D (one thread per d_head element).
 * K/V are loaded in tiles of seq_tile rows into shared memory.
 * Score row (S floats) kept in shared memory; softmax applied in-place.
 */
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define SEQ    {seq_len}
#define D_H    {d_head}
#define BC     {seq_tile}
#define SCALE  {scale_val}f

__device__ __forceinline__ float warp_reduce_sum(float v) {{
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}}

__global__ void attention_tiled(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                float* __restrict__ O,
                                int S, int D)
{{
    // smem layout: [K_tile: BC*D] [V_tile: BC*D] [scores: S] [warp_buf: D/32]
    extern __shared__ float smem[];
    float* K_tile  = smem;
    float* V_tile  = smem + BC * D;
    float* scores  = smem + 2 * BC * D;          // full S scores for this row
    float* wbuf    = smem + 2 * BC * D + S;       // cross-warp reduction

    int row = blockIdx.x;
    if (row >= S) return;

    int tid   = threadIdx.x;    // d_head index (0 .. D-1)
    int lane  = tid % 32;
    int warpid= tid / 32;

    // Load Q[row, tid] into register
    float my_q = Q[row * D + tid];

    // ── Phase 1: compute all S scores ─────────────────────────────────
    for (int j_start = 0; j_start < S; j_start += BC) {{
        // Load K tile [j_start .. j_start+BC, :]
        for (int e = tid; e < BC * D; e += D) {{
            int kr = e / D, kd = e % D;
            int gj = j_start + kr;
            K_tile[e] = (gj < S) ? K[gj * D + kd] : 0.0f;
        }}
        __syncthreads();

        // Compute BC dot products (parallel across D threads)
        #pragma unroll {unroll}
        for (int j = 0; j < BC; j++) {{
            float partial = my_q * K_tile[j * D + tid];
            float ws = warp_reduce_sum(partial);
            if (lane == 0) wbuf[warpid] = ws;
            __syncthreads();
            if (tid == 0) {{
                float score = 0.0f;
                for (int w = 0; w < (D + 31) / 32; w++) score += wbuf[w];
                int gj = j_start + j;
                scores[gj] = (gj < S) ? score * SCALE : -1e38f;
            }}
            __syncthreads();
        }}
    }}

    // ── Phase 2: softmax over scores[0..S-1] ──────────────────────────
    // Parallel max reduce
    float tmax = -1e38f;
    for (int j = tid; j < S; j += D) tmax = fmaxf(tmax, scores[j]);
    float wmax = warp_reduce_sum(fmaxf(tmax, -1e38f));
    // (use wbuf for cross-warp max)
    if (lane == 0) wbuf[warpid] = wmax;
    __syncthreads();
    float row_max = -1e38f;
    if (tid == 0) {{ for (int w = 0; w < (D+31)/32; w++) row_max = fmaxf(row_max, wbuf[w]); wbuf[0] = row_max; }}
    __syncthreads();
    row_max = wbuf[0];

    // Exp + partial sum
    float tsum = 0.0f;
    for (int j = tid; j < S; j += D) {{
        scores[j] = expf(scores[j] - row_max);
        tsum += scores[j];
    }}
    float wsum = warp_reduce_sum(tsum);
    if (lane == 0) wbuf[warpid] = wsum;
    __syncthreads();
    float row_sum = 0.0f;
    if (tid == 0) {{ for (int w = 0; w < (D+31)/32; w++) row_sum += wbuf[w]; wbuf[0] = row_sum; }}
    __syncthreads();
    row_sum = wbuf[0];

    // Normalise
    for (int j = tid; j < S; j += D) scores[j] /= row_sum;
    __syncthreads();

    // ── Phase 3: output O[row] = scores @ V ──────────────────────────
    float my_o = 0.0f;
    for (int j_start = 0; j_start < S; j_start += BC) {{
        for (int e = tid; e < BC * D; e += D) {{
            int vr = e / D, vd = e % D;
            int gj = j_start + vr;
            V_tile[e] = (gj < S) ? V[gj * D + vd] : 0.0f;
        }}
        __syncthreads();

        #pragma unroll {unroll}
        for (int j = 0; j < BC; j++) {{
            int gj = j_start + j;
            if (gj < S)
                my_o += scores[gj] * V_tile[j * D + tid];
        }}
        __syncthreads();
    }}

    O[row * D + tid] = my_o;
}}
"""

# ── Flash Attention (online softmax, no S×S materialisation) ──────────────
ATTENTION_FLASH_TEMPLATE = r"""
/*
 * Flash Attention kernel  (Dao et al., NeurIPS 2022)
 * seq_tile={seq_tile}  unroll={unroll}  flash=True
 * S={seq_len}  D={d_head}
 *
 * blockDim.x = D.  One block per Q row.
 * Iterates over BC-wide K/V tiles; maintains running (m, l, O) accumulators
 * per thread (each thread owns one d_head output slot).
 * Never writes the S×S attention matrix to DRAM — O(S·D) memory traffic
 * instead of O(S²+S·D).
 */
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define SEQ    {seq_len}
#define D_H    {d_head}
#define BC     {seq_tile}
#define SCALE  {scale_val}f

__device__ __forceinline__ float warp_reduce_sum_f(float v) {{
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}}

__global__ void attention_flash(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                float* __restrict__ O,
                                int S, int D)
{{
    // smem: [K_tile: BC*D] [V_tile: BC*D] [score_tile: BC] [wbuf: D/32]
    extern __shared__ float smem[];
    float* K_tile   = smem;
    float* V_tile   = smem + BC * D;
    float* stile    = smem + 2 * BC * D;   // BC scores for current tile
    float* wbuf     = smem + 2 * BC * D + BC;

    int row = blockIdx.x;
    if (row >= S) return;

    int tid    = threadIdx.x;
    int lane   = tid % 32;
    int warpid = tid / 32;
    int nwarps = (D + 31) / 32;

    float my_q = Q[row * D + tid];

    // Flash Attention running accumulators (per thread = per d_head element)
    float m = -1e38f;   // running max of scores seen so far
    float l = 0.0f;     // running normaliser ∑ exp(s - m)
    float o = 0.0f;     // running output for this d_head element

    for (int j_start = 0; j_start < S; j_start += BC) {{
        // Load K tile and V tile cooperatively
        for (int e = tid; e < BC * D; e += D) {{
            int kr = e / D, kd = e % D;
            int gj = j_start + kr;
            float kv = (gj < S) ? K[gj * D + kd] : 0.0f;
            float vv = (gj < S) ? V[gj * D + kd] : 0.0f;
            K_tile[e] = kv;
            V_tile[e] = vv;
        }}
        __syncthreads();

        // Compute BC scores for this tile
        int tile_len = min(BC, S - j_start);
        #pragma unroll {unroll}
        for (int j = 0; j < BC; j++) {{
            float partial = (j < tile_len) ? my_q * K_tile[j * D + tid] : 0.0f;
            float ws = warp_reduce_sum_f(partial);
            if (lane == 0) wbuf[warpid] = ws;
            __syncthreads();
            if (tid == 0) {{
                float score = 0.0f;
                for (int w = 0; w < nwarps; w++) score += wbuf[w];
                stile[j] = (j < tile_len) ? score * SCALE : -1e38f;
            }}
            __syncthreads();
        }}

        // Online softmax update
        // 1. tile max
        float m_tile = -1e38f;
        for (int j = 0; j < tile_len; j++) m_tile = fmaxf(m_tile, stile[j]);

        float m_new    = fmaxf(m, m_tile);
        float rescale  = expf(m - m_new);    // correction factor for old accumulators

        // 2. update (m, l, o) with this tile's contribution
        l = l * rescale;
        o = o * rescale;
        #pragma unroll {unroll}
        for (int j = 0; j < BC; j++) {{
            if (j < tile_len) {{
                float p = expf(stile[j] - m_new);
                l += p;
                o += p * V_tile[j * D + tid];
            }}
        }}
        m = m_new;
        __syncthreads();
    }}

    // Final normalisation
    O[row * D + tid] = o / l;
}}
"""

# ── Benchmark driver (shared by both attention variants) ───────────────────
ATTENTION_BENCHMARK_MAIN = r"""

/* ── Attention benchmark driver ───────────────────────────────────────── */
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define CUDA_CHECK(call) \
    do {{ cudaError_t e=(call); if(e!=cudaSuccess){{ \
        fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(1); }} }} while(0)

static void fill_random(float* d, int n) {{
    float* h=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++) h[i]=(float)rand()/RAND_MAX * 0.1f;
    CUDA_CHECK(cudaMemcpy(d,h,n*sizeof(float),cudaMemcpyHostToDevice));
    free(h);
}}

int main(void) {{
    int warmup = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    int iters  = getenv("ITERS")  ? atoi(getenv("ITERS"))  : 30;

    const int S = {seq_len}, D = {d_head};
    float *Q, *K, *V, *O;
    CUDA_CHECK(cudaMalloc(&Q, S*D*4));
    CUDA_CHECK(cudaMalloc(&K, S*D*4));
    CUDA_CHECK(cudaMalloc(&V, S*D*4));
    CUDA_CHECK(cudaMalloc(&O, S*D*4));
    fill_random(Q, S*D);
    fill_random(K, S*D);
    fill_random(V, S*D);

    // smem size: 2*BC*D*4 + extra (BC or S) + D/32*4
    size_t smem = (2*{seq_tile}*D + {smem_extra} + D/32 + 4) * sizeof(float);
    dim3 grid(S), block(D);

    // Warmup
    for(int i=0;i<warmup;i++) {{
        {kernel_call}
        CUDA_CHECK(cudaDeviceSynchronize());
    }}
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total=0.0f, mn=FLT_MAX, mx=0.0f;

    for(int i=0;i<iters;i++) {{
        CUDA_CHECK(cudaEventRecord(start));
        {kernel_call}
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms=0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
        printf("SAMPLE {tag} %.4f\n", ms);
        total+=ms; if(ms<mn) mn=ms; if(ms>mx) mx=ms;
    }}

    printf("TIMING {tag} %.4f %.4f %.4f %d\n", total/iters, mn, mx, iters);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
    return 0;
}}
"""


# ── Code generation ────────────────────────────────────────────────────────

def _make_tag(params: dict) -> str:
    """Build a unique tag string for this variant."""
    return (f"attention_flash{params['flash']}_"
            f"seqtile{params['seq_tile']}_unroll{params['unroll']}")


def generate_attention(params: dict) -> str:
    """
    Generate a complete self-contained .cu for the given attention params.

    Args:
        params: dict with keys flash (bool), seq_tile (int), unroll (int).

    Returns:
        Full CUDA source as a string.
    """
    flash     = params["flash"]
    seq_tile  = params["seq_tile"]
    unroll    = params["unroll"]
    seq_len   = SEQ_LEN
    d_head    = D_HEAD
    scale_val = f"{SCALE:.6f}"
    tag       = _make_tag(params)

    kernel_template = ATTENTION_FLASH_TEMPLATE if flash else ATTENTION_TILED_TEMPLATE
    smem_extra = seq_tile if flash else seq_len   # stile[BC] vs scores[S]

    kernel_src = kernel_template.format(
        seq_tile=seq_tile, unroll=unroll,
        seq_len=seq_len, d_head=d_head,
        scale_val=scale_val,
    )

    fn_name = "attention_flash" if flash else "attention_tiled"
    call    = (f"{fn_name}<<<grid, block, smem>>>(Q, K, V, O, S, D);")

    driver = ATTENTION_BENCHMARK_MAIN.format(
        seq_len=seq_len, d_head=d_head, seq_tile=seq_tile,
        smem_extra=smem_extra, kernel_call=call, tag=tag,
    )

    return kernel_src + driver


def write_attention_variant(params: dict, out_path: Path) -> None:
    """Write a generated attention .cu variant to disk."""
    src = generate_attention(params)
    out_path.write_text(src, encoding="utf-8")


def build_attention_search_space() -> dict:
    """
    Return the attention kernel search space.

    Dimensions:
      seq_tile [16, 32]     — K/V block size (BC in Flash Attention)
      flash    [False, True] — tiled vs flash variant
      unroll   [1, 2, 4]   — pragma unroll factor
    """
    space: dict = {
        "seq_tile": [16, 32],
        "flash":    [False, True],
        "unroll":   [1, 2, 4],
        "_kernel":  "attention",
        "_total_variants": 2 * 2 * 3,
    }
    return space


def enumerate_attention_variants() -> list[tuple[dict, Path]]:
    """
    Return all valid (params, path) tuples for attention variants.

    Returns:
        List of (params_dict, generated_cu_path) tuples.
    """
    import itertools
    space = build_attention_search_space()
    keys  = [k for k in space if not k.startswith("_")]
    vals  = [space[k] for k in keys]

    variants: list[tuple[dict, Path]] = []
    for combo in itertools.product(*vals):
        params = dict(zip(keys, combo))
        tag    = _make_tag(params)
        path   = GEN_DIR / f"{tag}.cu"
        variants.append((params, path))
    return variants


# ── FLOP / byte model for roofline ────────────────────────────────────────

def attention_flops(S: int, D: int) -> int:
    """
    Total FLOPs for single-head attention with sequence S and head dim D.

    QK^T:     2·S²·D   (S×D matrix × D×S matrix)
    Softmax:  5·S²      (max, sub, exp, sum, div per row)
    AV:       2·S²·D   (S×S weights × S×D values)
    Total ≈   4·S²·D + 5·S²
    """
    return 4 * S * S * D + 5 * S * S


def attention_bytes_naive(S: int, D: int) -> int:
    """Bytes transferred by naive/tiled attention (loads Q+K+V, score matrix, output)."""
    return int((3 * S * D + 2 * S * S + S * D) * 4)


def attention_bytes_flash(S: int, D: int) -> int:
    """Bytes transferred by Flash Attention (loads Q+K+V, writes output — no S×S)."""
    return int(4 * S * D * 4)

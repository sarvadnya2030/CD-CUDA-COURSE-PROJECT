/*
 * benchmark_runner.cu — standalone binary that profiles all baseline kernels.
 *
 * Reads WARMUP and ITERS from environment variables (defaults: 5, 30).
 * Prints per-sample timings and aggregate summary:
 *
 *   SAMPLE <kernel_name> <ms>           — one line per timed iteration
 *   TIMING <kernel_name> <mean> <min> <max> <iters>  — aggregate
 *
 * The SAMPLE lines are consumed by benchmark.py for statistical analysis
 * (mean, std, 95% CI, Welch t-test).  The TIMING line maintains backward
 * compatibility with the original parser in autotune.py.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* ── Kernel declarations ─────────────────────────────────────────────────── */

__global__ void matmul_naive(const float*, const float*, float*, int);
__global__ void softmax_naive(const float*, float*, int, int);
__global__ void reduction_naive(const float*, float*, int);
__global__ void layernorm_naive(const float*, float*, const float*,
                                const float*, int, int, float);
__global__ void attention_naive(const float*, const float*, const float*,
                                float*, int, int, float);

/* attention_naive kernel body — must match baseline_kernels.cu exactly */
__global__ void attention_naive(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                float* __restrict__ O,
                                int S, int D, float scale)
{
    extern __shared__ float scores[];
    int row = blockIdx.x;
    if (row >= S) return;
    if (threadIdx.x != 0) return;
    for (int j = 0; j < S; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += Q[row*D+d] * K[j*D+d];
        scores[j] = dot * scale;
    }
    float max_v = scores[0];
    for (int j = 1; j < S; ++j) max_v = fmaxf(max_v, scores[j]);
    float sum = 0.0f;
    for (int j = 0; j < S; ++j) { scores[j] = expf(scores[j]-max_v); sum += scores[j]; }
    for (int j = 0; j < S; ++j) scores[j] /= sum;
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < S; ++j) acc += scores[j] * V[j*D+d];
        O[row*D+d] = acc;
    }
}

/* ── Timing result ───────────────────────────────────────────────────────── */

typedef struct { float mean, min, max; int iters; } TimingResult;

static TimingResult time_kernel(const char* name,
                                void (*launch)(void*), void* ctx,
                                int warmup, int iters)
{
    /* Warmup */
    for (int i = 0; i < warmup; ++i) launch(ctx);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.0f, mn = FLT_MAX, mx = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch(ctx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("SAMPLE %s %.4f\n", name, ms);   /* per-sample for statistics */
        total += ms;
        if (ms < mn) mn = ms;
        if (ms > mx) mx = ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    TimingResult r; r.mean = total/iters; r.min = mn; r.max = mx; r.iters = iters;
    return r;
}

/* ── Kernel definitions ──────────────────────────────────────────────────── */

__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k) s += A[row*N+k] * B[k*N+col];
        C[row*N+col] = s;
    }
}

__global__ void softmax_naive(const float* __restrict__ in,
                              float* __restrict__ out, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* ir = in  + row * cols;
    float*       or_ = out + row * cols;
    float mx = ir[0];
    for (int i = 1; i < cols; ++i) mx = fmaxf(mx, ir[i]);
    float s = 0.0f;
    for (int i = 0; i < cols; ++i) { or_[i] = expf(ir[i]-mx); s += or_[i]; }
    for (int i = 0; i < cols; ++i) or_[i] /= s;
}

__global__ void reduction_naive(const float* __restrict__ in,
                                float* __restrict__ out, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < N) ? in[gid] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void layernorm_naive(const float* __restrict__ in,
                                float* __restrict__ out,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                int rows, int cols, float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* ir = in  + row * cols;
    float*       or_ = out + row * cols;
    float mean = 0.0f;
    for (int i = 0; i < cols; ++i) mean += ir[i];
    mean /= cols;
    float var = 0.0f;
    for (int i = 0; i < cols; ++i) { float d = ir[i]-mean; var += d*d; }
    var /= cols;
    float inv_std = rsqrtf(var + eps);
    for (int i = 0; i < cols; ++i)
        or_[i] = gamma[i] * (ir[i]-mean) * inv_std + beta[i];
}

/* ── Launch contexts ─────────────────────────────────────────────────────── */

struct MatmulCtx    { float *A,*B,*C; int N; int blk; };
struct SoftmaxCtx   { float *in,*out; int rows,cols; };
struct ReductionCtx { float *in,*out; int N,blk; };
struct LayernormCtx { float *in,*out,*g,*b; int rows,cols; };
struct AttentionCtx { float *Q,*K,*V,*O; int S,D; float scale; };

static void launch_matmul(void* p) {
    auto* c = (MatmulCtx*)p;
    dim3 block(c->blk, c->blk);
    dim3 grid((c->N+c->blk-1)/c->blk, (c->N+c->blk-1)/c->blk);
    matmul_naive<<<grid,block>>>(c->A, c->B, c->C, c->N);
}

static void launch_softmax(void* p) {
    auto* c = (SoftmaxCtx*)p;
    softmax_naive<<<c->rows, 1>>>(c->in, c->out, c->rows, c->cols);
}

static void launch_reduction(void* p) {
    auto* c = (ReductionCtx*)p;
    int grid = (c->N + c->blk - 1) / c->blk;
    reduction_naive<<<grid, c->blk, c->blk*sizeof(float)>>>(c->in, c->out, c->N);
}

static void launch_layernorm(void* p) {
    auto* c = (LayernormCtx*)p;
    layernorm_naive<<<c->rows, 1>>>(c->in, c->out, c->g, c->b,
                                    c->rows, c->cols, 1e-5f);
}

static void launch_attention_ctx(void* p) {
    auto* c = (AttentionCtx*)p;
    size_t smem = c->S * sizeof(float);
    attention_naive<<<c->S, 1, smem>>>(c->Q, c->K, c->V, c->O,
                                       c->S, c->D, c->scale);
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static void fill_random(float* d, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) h[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

static void fill_ones(float* d, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

static void print_timing(const char* name, TimingResult t) {
    printf("TIMING %s %.4f %.4f %.4f %d\n",
           name, t.mean, t.min, t.max, t.iters);
    printf("  mean=%.3fms  min=%.3fms  max=%.3fms\n",
           t.mean, t.min, t.max);
    fflush(stdout);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void)
{
    int warmup = 5, iters = 30;
    if (const char* e = getenv("WARMUP")) warmup = atoi(e);
    if (const char* e = getenv("ITERS"))  iters  = atoi(e);

    /* Device info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Device: %s ===\n", prop.name);
    printf("  SM count:          %d\n",  prop.multiProcessorCount);
    printf("  Max threads/block: %d\n",  prop.maxThreadsPerBlock);
    printf("  Shared mem/block:  %zu KB\n", prop.sharedMemPerBlock/1024);
    printf("  Global memory:     %.1f GB\n", (double)prop.totalGlobalMem/1e9);
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3
                   * (prop.memoryBusWidth / 8.0) / 1e9;
    printf("  Theoretical BW:    %.1f GB/s\n\n", peak_bw);
    fflush(stdout);

    /* ── 1. Matrix multiplication 1024×1024 ─── */
    {
        const int N = 1024, blk = 16;
        float *A, *B, *C;
        CUDA_CHECK(cudaMalloc(&A, N*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B, N*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&C, N*N*sizeof(float)));
        fill_random(A, N*N); fill_random(B, N*N);

        MatmulCtx ctx = {A, B, C, N, blk};
        printf("[Matmul %dx%d, block %dx%d]\n", N, N, blk, blk);
        TimingResult t = time_kernel("matmul_1024", launch_matmul, &ctx, warmup, iters);
        print_timing("matmul_1024", t);

        double flops = 2.0 * N * N * N;
        printf("  GFLOPS: %.2f\n\n", flops / (t.mean * 1e6));
        CUDA_CHECK(cudaFree(A)); CUDA_CHECK(cudaFree(B)); CUDA_CHECK(cudaFree(C));
    }

    /* ── 2. Softmax 1024×4096 ─── */
    {
        const int rows = 1024, cols = 4096;
        float *in, *out;
        CUDA_CHECK(cudaMalloc(&in,  rows*cols*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out, rows*cols*sizeof(float)));
        fill_random(in, rows*cols);

        SoftmaxCtx ctx = {in, out, rows, cols};
        printf("[Softmax %dx%d]\n", rows, cols);
        TimingResult t = time_kernel("softmax_1024x4096", launch_softmax, &ctx, warmup, iters);
        print_timing("softmax_1024x4096", t);

        double bytes = 3.0 * rows * cols * sizeof(float);
        printf("  Eff. BW: %.1f GB/s\n\n", bytes / (t.mean * 1e6));
        CUDA_CHECK(cudaFree(in)); CUDA_CHECK(cudaFree(out));
    }

    /* ── 3. Reduction 1M elements ─── */
    {
        const int N = 1 << 20, blk = 256;
        float *in, *out;
        int grid = (N + blk - 1) / blk;
        CUDA_CHECK(cudaMalloc(&in,  N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out, grid*sizeof(float)));
        fill_random(in, N);

        ReductionCtx ctx = {in, out, N, blk};
        printf("[Reduction N=%d, block=%d]\n", N, blk);
        TimingResult t = time_kernel("reduction_1M", launch_reduction, &ctx, warmup, iters);
        print_timing("reduction_1M", t);

        double bytes = (double)N * sizeof(float);
        printf("  Eff. BW: %.1f GB/s\n\n", bytes / (t.mean * 1e6));
        CUDA_CHECK(cudaFree(in)); CUDA_CHECK(cudaFree(out));
    }

    /* ── 4. LayerNorm 512×2048 ─── */
    {
        const int rows = 512, cols = 2048;
        float *in, *out, *gamma, *beta;
        CUDA_CHECK(cudaMalloc(&in,    rows*cols*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out,   rows*cols*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gamma, cols*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&beta,  cols*sizeof(float)));
        fill_random(in, rows*cols);
        fill_ones(gamma, cols);
        CUDA_CHECK(cudaMemset(beta, 0, cols*sizeof(float)));

        LayernormCtx ctx = {in, out, gamma, beta, rows, cols};
        printf("[LayerNorm %dx%d]\n", rows, cols);
        TimingResult t = time_kernel("layernorm_512x2048", launch_layernorm, &ctx, warmup, iters);
        print_timing("layernorm_512x2048", t);

        double bytes = 3.0 * rows * cols * sizeof(float);
        printf("  Eff. BW: %.1f GB/s\n\n", bytes / (t.mean * 1e6));
        CUDA_CHECK(cudaFree(in)); CUDA_CHECK(cudaFree(out));
        CUDA_CHECK(cudaFree(gamma)); CUDA_CHECK(cudaFree(beta));
    }

    /* ── 5. Attention S=512, D=64 (single head) ─── */
    {
        const int S = 512, D = 64;
        float scale = 1.0f / sqrtf((float)D);
        float *Q, *K, *V, *O;
        CUDA_CHECK(cudaMalloc(&Q, S*D*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&K, S*D*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&V, S*D*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&O, S*D*sizeof(float)));
        fill_random(Q, S*D); fill_random(K, S*D); fill_random(V, S*D);

        AttentionCtx ctx = {Q, K, V, O, S, D, scale};
        printf("[Attention S=%d D=%d (naive)]\n", S, D);
        TimingResult t = time_kernel("attention_512x64", launch_attention_ctx, &ctx, warmup, iters);
        print_timing("attention_512x64", t);

        double flops = 4.0 * S * S * D + 5.0 * S * S;
        printf("  GFLOPS: %.2f\n\n", flops / (t.mean * 1e6));
        CUDA_CHECK(cudaFree(Q)); CUDA_CHECK(cudaFree(K));
        CUDA_CHECK(cudaFree(V)); CUDA_CHECK(cudaFree(O));
    }

    printf("=== Baseline profiling complete ===\n");
    fflush(stdout);
    return 0;
}

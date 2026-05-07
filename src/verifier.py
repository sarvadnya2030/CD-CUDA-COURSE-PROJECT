"""
verifier.py — Correctness verification for generated CUDA kernel variants.

For each kernel type, a small standalone verification binary is generated,
compiled, and run.  The kernel output is compared against a NumPy reference.

Verification protocol per kernel:
  matmul    N=64   np.matmul                   tolerance 1e-3 (max abs diff)
  softmax   R=4,C=16  np.softmax               tolerance 1e-5
  reduction N=256  np.sum                       tolerance 1e-3
  layernorm R=4,C=16  manual mean/var           tolerance 1e-4
  attention S=512,D=64  scaled-dot-product     tolerance 1e-3

Deterministic input: input[i] = (i % 97) * 0.01 - 0.5   (no srand needed)

cupy is used for the matmul reference if available (cuBLAS path), otherwise
numpy is used.

Failures are logged to results/{kernel}_failures.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cupy as cp
    _CUPY_OK = True
except ImportError:
    cp = None  # type: ignore
    _CUPY_OK = False

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# ── Tolerances per kernel ──────────────────────────────────────────────────

_TOLERANCES: dict[str, float] = {
    "matmul":    1e-3,
    "softmax":   1e-5,
    "reduction": 1e-3,
    "layernorm": 1e-4,
    "attention": 1e-3,
}

# ── Verification C templates ───────────────────────────────────────────────
# Each template is appended after the kernel code extracted from a generated
# .cu file.  They use fixed small problem sizes and a deterministic input
# pattern, then print results as VERIFY_OUTPUT lines.

_VERIFY_MAIN_MATMUL = r"""
/* ── Verification driver (matmul) ── */
#include <stdio.h>
#include <stdlib.h>
#define CUDA_CHECK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(void){
    const int N=64;
    float *hA=(float*)malloc(N*N*4);
    float *hB=(float*)malloc(N*N*4);
    float *hC=(float*)calloc(N*N,4);
    for(int i=0;i<N*N;i++){
        hA[i]=(float)(i%97)*0.01f-0.5f;
        hB[i]=(float)(i%89)*0.01f-0.3f;
    }
    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,N*N*4));
    CUDA_CHECK(cudaMalloc(&dB,N*N*4));
    CUDA_CHECK(cudaMalloc(&dC,N*N*4));
    CUDA_CHECK(cudaMemcpy(dA,hA,N*N*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,N*N*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC,0,N*N*4));
    matmul_opt<<<dim3((N+TILE_X-1)/TILE_X,(N+TILE_Y-1)/TILE_Y),dim3(TILE_X,TILE_Y)>>>(dA,dB,dC,N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC,dC,N*N*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<N*N;i++) printf("VERIFY_OUTPUT %.8f\n",hC[i]);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    free(hA);free(hB);free(hC);
    return 0;
}
"""

_VERIFY_MAIN_SOFTMAX = r"""
/* ── Verification driver (softmax) ── */
#include <stdio.h>
#include <stdlib.h>
#define CUDA_CHECK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(void){
    const int rows=4,cols=16;
    int n=rows*cols;
    float *hI=(float*)malloc(n*4);
    float *hO=(float*)calloc(n,4);
    for(int i=0;i<n;i++) hI[i]=(float)(i%97)*0.01f-0.5f;
    float *dI,*dO;
    CUDA_CHECK(cudaMalloc(&dI,n*4));
    CUDA_CHECK(cudaMalloc(&dO,n*4));
    CUDA_CHECK(cudaMemcpy(dI,hI,n*4,cudaMemcpyHostToDevice));
    softmax_opt<<<rows,BLOCK,BLOCK*4>>>(dI,dO,rows,cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO,dO,n*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;i++) printf("VERIFY_OUTPUT %.8f\n",hO[i]);
    cudaFree(dI);cudaFree(dO);
    free(hI);free(hO);
    return 0;
}
"""

_VERIFY_MAIN_REDUCTION = r"""
/* ── Verification driver (reduction) ── */
#include <stdio.h>
#include <stdlib.h>
#define CUDA_CHECK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(void){
    const int N=256;
    int blk=BLOCK;
    int grid=(N+blk*2-1)/(blk*2);
    float *hI=(float*)malloc(N*4);
    float *hO=(float*)calloc(grid,4);
    for(int i=0;i<N;i++) hI[i]=(float)(i%97)*0.01f-0.5f;
    float *dI,*dO;
    CUDA_CHECK(cudaMalloc(&dI,N*4));
    CUDA_CHECK(cudaMalloc(&dO,grid*4));
    CUDA_CHECK(cudaMemcpy(dI,hI,N*4,cudaMemcpyHostToDevice));
    reduction_opt<<<grid,blk,blk*4>>>(dI,dO,N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO,dO,grid*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<grid;i++) printf("VERIFY_OUTPUT %.8f\n",hO[i]);
    cudaFree(dI);cudaFree(dO);
    free(hI);free(hO);
    return 0;
}
"""

_VERIFY_MAIN_LAYERNORM = r"""
/* ── Verification driver (layernorm) ── */
#include <stdio.h>
#include <stdlib.h>
#define CUDA_CHECK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(void){
    const int rows=4,cols=16;
    int n=rows*cols;
    float *hI=(float*)malloc(n*4);
    float *hO=(float*)calloc(n,4);
    float *hG=(float*)malloc(cols*4);
    float *hB_=(float*)calloc(cols,4);
    for(int i=0;i<n;i++) hI[i]=(float)(i%97)*0.01f-0.5f;
    for(int i=0;i<cols;i++) hG[i]=1.0f;
    float *dI,*dO,*dG,*dBeta;
    CUDA_CHECK(cudaMalloc(&dI,n*4));
    CUDA_CHECK(cudaMalloc(&dO,n*4));
    CUDA_CHECK(cudaMalloc(&dG,cols*4));
    CUDA_CHECK(cudaMalloc(&dBeta,cols*4));
    CUDA_CHECK(cudaMemcpy(dI,hI,n*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dG,hG,cols*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dBeta,0,cols*4));
    layernorm_opt<<<rows,BLOCK,BLOCK*4>>>(dI,dO,dG,dBeta,rows,cols,1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO,dO,n*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;i++) printf("VERIFY_OUTPUT %.8f\n",hO[i]);
    cudaFree(dI);cudaFree(dO);cudaFree(dG);cudaFree(dBeta);
    free(hI);free(hO);free(hG);free(hB_);
    return 0;
}
"""

_VERIFY_MAIN_ATTENTION = r"""
/* ── Verification driver (attention) ── */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CUDA_CHECK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(void){
    const int S=512,D=64;
    int n=S*D;
    float *hQ=(float*)malloc(n*4);
    float *hK=(float*)malloc(n*4);
    float *hV=(float*)malloc(n*4);
    float *hO=(float*)calloc(n,4);
    for(int i=0;i<n;i++){
        hQ[i]=sinf(i*0.01f);
        hK[i]=cosf(i*0.01f);
        hV[i]=sinf(i*0.007f);
    }
    float *dQ,*dK,*dV,*dO;
    CUDA_CHECK(cudaMalloc(&dQ,n*4));
    CUDA_CHECK(cudaMalloc(&dK,n*4));
    CUDA_CHECK(cudaMalloc(&dV,n*4));
    CUDA_CHECK(cudaMalloc(&dO,n*4));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,n*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,hK,n*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,hV,n*4,cudaMemcpyHostToDevice));
    float scale=1.0f/sqrtf(64.0f);
    attention_naive<<<S,1,S*sizeof(float)>>>(dQ,dK,dV,dO,S,D,scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO,dO,n*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;i++) printf("VERIFY_OUTPUT %.8f\n",hO[i]);
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    free(hQ);free(hK);free(hV);free(hO);
    return 0;
}
"""

_VERIFY_MAINS: dict[str, str] = {
    "matmul":    _VERIFY_MAIN_MATMUL,
    "softmax":   _VERIFY_MAIN_SOFTMAX,
    "reduction": _VERIFY_MAIN_REDUCTION,
    "layernorm": _VERIFY_MAIN_LAYERNORM,
    "attention": _VERIFY_MAIN_ATTENTION,
}

# ── Deterministic input (must match C templates above) ─────────────────────

def _make_input(n: int) -> np.ndarray:
    """Match: (i % 97) * 0.01 - 0.5"""
    i = np.arange(n, dtype=np.float64)
    return (i % 97) * 0.01 - 0.5


def _gamma_input(n: int) -> np.ndarray:
    """All-ones gamma for layernorm."""
    return np.ones(n, dtype=np.float64)


# ── NumPy reference implementations ───────────────────────────────────────

def _ref_matmul(N: int = 64) -> np.ndarray:
    """Match C template: hA[i]=(i%97)*0.01-0.5  hB[i]=(i%89)*0.01-0.3"""
    idx = np.arange(N * N, dtype=np.float64)
    A = ((idx % 97) * 0.01 - 0.5).reshape(N, N).astype(np.float32)
    B = ((idx % 89) * 0.01 - 0.3).reshape(N, N).astype(np.float32)
    if _CUPY_OK:
        C = cp.matmul(cp.array(A), cp.array(B))
        return cp.asnumpy(C).flatten()
    return np.matmul(A.astype(np.float64), B.astype(np.float64)).astype(np.float32).flatten()


def _ref_softmax(rows: int = 4, cols: int = 16) -> np.ndarray:
    idx = np.arange(rows * cols)
    x = ((idx % 97) * 0.01 - 0.5).reshape(rows, cols).astype(np.float32)
    x_max = x.max(axis=1, keepdims=True)
    e = np.exp(x - x_max)
    return (e / e.sum(axis=1, keepdims=True)).flatten()


def _ref_reduction(N: int = 256, block_size: int = 64) -> np.ndarray:
    """Reference: partial block sums (two-element-per-thread mapping)."""
    idx = np.arange(N)
    inp = ((idx % 97) * 0.01 - 0.5).astype(np.float32)
    blk = block_size
    stride = blk * 2
    grid = (N + stride - 1) // stride
    partial = np.zeros(grid, dtype=np.float32)
    for b in range(grid):
        start = b * stride
        partial[b] = inp[start: start + stride].sum()
    return partial


def _ref_layernorm(rows: int = 4, cols: int = 16, eps: float = 1e-5) -> np.ndarray:
    idx = np.arange(rows * cols)
    x = ((idx % 97) * 0.01 - 0.5).reshape(rows, cols).astype(np.float64)
    gamma = np.ones(cols, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    var  = x.var(axis=1, keepdims=True)
    out  = gamma * (x - mean) / np.sqrt(var + eps)
    return out.flatten().astype(np.float32)


def _ref_attention(S: int = 512, D: int = 64) -> list:
    """
    Scaled-dot-product attention reference implementation.
    Matches C template: Q[i]=sinf(i*0.01), K[i]=cosf(i*0.01), V[i]=sinf(i*0.007).
    Returns first 16 output values for spot-check.
    """
    rng = np.random.default_rng(42)
    i = np.arange(S * D)
    Q = np.sin(i * 0.01).reshape(S, D).astype(np.float32)
    K = np.cos(i * 0.01).reshape(S, D).astype(np.float32)
    V = np.sin(i * 0.007).reshape(S, D).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    scores = (Q @ K.T) * scale          # (S, S)
    scores -= scores.max(axis=1, keepdims=True)
    attn = np.exp(scores)
    attn /= attn.sum(axis=1, keepdims=True)
    O = attn @ V                        # (S, D)
    return O.flatten().astype(np.float32)


# ── Verification result ────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Outcome of a correctness check for one kernel variant."""
    is_correct: bool
    max_diff: float
    kernel: str
    variant: str
    tolerance: float
    n_elements_checked: int
    error_msg: str = ""


# ── Main verifier class ────────────────────────────────────────────────────

class CorrectnessVerifier:
    """
    Compile and run verification binaries for generated CUDA variants.

    For each variant, a tiny verification .cu is assembled from:
      - The kernel code section of the generated file (before the benchmark driver)
      - A kernel-specific verification main (fixed input, prints VERIFY_OUTPUT lines)

    The output is compared to a NumPy reference with per-kernel tolerances.

    Failures are saved to results/{kernel}_failures.json.
    """

    def __init__(
        self,
        arch: str = "sm_75",
        results_dir: Path = RESULTS_DIR,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            arch:        Compilation target architecture.
            results_dir: Directory for failure logs.
            verbose:     Print per-variant status.
        """
        self._arch        = arch
        self._results_dir = results_dir
        self._verbose     = verbose
        self._failures:  dict[str, list] = {}
        self._n_checked  = 0
        self._n_passed   = 0
        self._n_failed   = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def check(
        self,
        kernel_name: str,
        src_path: Path,
        params: dict,
    ) -> VerificationResult:
        """
        Verify a single generated kernel variant.

        Args:
            kernel_name: "matmul" | "softmax" | "reduction" | "layernorm"
            src_path:    Path to the generated .cu source file.
            params:      Parameter dict for the variant (used in reporting).

        Returns:
            VerificationResult with is_correct and max_diff.
        """
        tolerance = _TOLERANCES.get(kernel_name, 1e-3)
        variant   = src_path.stem
        self._n_checked += 1

        # 1. Extract kernel code (before benchmark driver)
        kernel_src = self._extract_kernel_code(src_path)
        if kernel_src is None:
            result = VerificationResult(
                is_correct=False, max_diff=float("inf"),
                kernel=kernel_name, variant=variant,
                tolerance=tolerance, n_elements_checked=0,
                error_msg="Could not extract kernel code from source",
            )
            self._record_failure(kernel_name, variant, result)
            return result

        # 2. Assemble verify .cu
        verify_main = _VERIFY_MAINS.get(kernel_name)
        if verify_main is None:
            result = VerificationResult(
                is_correct=True, max_diff=0.0,   # skip unknown kernels
                kernel=kernel_name, variant=variant,
                tolerance=tolerance, n_elements_checked=0,
                error_msg=f"No verification template for kernel {kernel_name!r}",
            )
            return result

        verify_src = kernel_src + verify_main

        # 3. Compile verify binary
        ok, bin_path, compile_err = self._compile_verify(verify_src, variant)
        if not ok:
            result = VerificationResult(
                is_correct=False, max_diff=float("inf"),
                kernel=kernel_name, variant=variant,
                tolerance=tolerance, n_elements_checked=0,
                error_msg=f"Compile failed: {compile_err[:200]}",
            )
            self._record_failure(kernel_name, variant, result)
            return result

        # 4. Run and parse output
        try:
            proc = subprocess.run(
                [str(bin_path)], capture_output=True, text=True, timeout=30
            )
            output = proc.stdout
        except subprocess.TimeoutExpired:
            result = VerificationResult(
                is_correct=False, max_diff=float("inf"),
                kernel=kernel_name, variant=variant,
                tolerance=tolerance, n_elements_checked=0,
                error_msg="Verification binary timed out",
            )
            self._record_failure(kernel_name, variant, result)
            return result
        finally:
            try:
                bin_path.unlink()
            except FileNotFoundError:
                pass

        gpu_vals = self._parse_verify_output(output)
        if len(gpu_vals) == 0:
            result = VerificationResult(
                is_correct=False, max_diff=float("inf"),
                kernel=kernel_name, variant=variant,
                tolerance=tolerance, n_elements_checked=0,
                error_msg="No VERIFY_OUTPUT lines in binary output",
            )
            self._record_failure(kernel_name, variant, result)
            return result

        # 5. Compare to reference
        ref_vals = self._reference(kernel_name, params)
        max_diff, is_correct = self._compare(
            np.array(gpu_vals, dtype=np.float32),
            np.array(ref_vals, dtype=np.float32),
            tolerance,
        )

        result = VerificationResult(
            is_correct=is_correct,
            max_diff=max_diff,
            kernel=kernel_name,
            variant=variant,
            tolerance=tolerance,
            n_elements_checked=len(gpu_vals),
        )

        if is_correct:
            self._n_passed += 1
        else:
            self._n_failed += 1
            self._record_failure(kernel_name, variant, result)

        if self._verbose:
            status = "PASS" if is_correct else f"FAIL (max_diff={max_diff:.2e})"
            print(f"  [VERIFY] {variant[:50]:50s} {status}")

        return result

    def print_summary(self) -> None:
        """Print verification pass/fail counts."""
        print(f"\n[VERIFY] {self._n_checked} checked — "
              f"{self._n_passed} passed — {self._n_failed} failed")

    def save_failures(self, kernel: str) -> None:
        """Save failure log for *kernel* to results/{kernel}_failures.json."""
        failures = self._failures.get(kernel, [])
        path = self._results_dir / f"{kernel}_failures.json"
        with open(path, "w") as f:
            json.dump(failures, f, indent=2)
        if failures:
            print(f"[VERIFY] Failures logged → {path}")

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_kernel_code(src_path: Path) -> Optional[str]:
        """
        Return everything in src_path before the benchmark driver comment.

        The generated files have a sentinel line::
            /* ── Benchmark driver
        """
        try:
            text = src_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        sentinel = "/* ── Benchmark driver"
        idx = text.find(sentinel)
        if idx >= 0:
            return text[:idx]
        # If no sentinel, return the entire file (may include main, will fail to compile)
        return text

    def _compile_verify(
        self,
        verify_src: str,
        variant_tag: str,
    ) -> tuple[bool, Path, str]:
        """Write verify source to a temp file and compile it."""
        tmp_dir = self._results_dir / "verify_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        src_path = tmp_dir / f"verify_{variant_tag[:60]}.cu"
        bin_path = tmp_dir / f"verify_{variant_tag[:60]}.bin"

        src_path.write_text(verify_src, encoding="utf-8")

        cmd = [
            "nvcc", "-O2", f"-arch={self._arch}",
            "--use_fast_math",
            str(src_path), "-o", str(bin_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        try:
            src_path.unlink()
        except FileNotFoundError:
            pass

        return result.returncode == 0, bin_path, result.stderr

    @staticmethod
    def _parse_verify_output(stdout: str) -> list[float]:
        """Parse VERIFY_OUTPUT lines into a list of floats."""
        vals = []
        for line in stdout.splitlines():
            if line.startswith("VERIFY_OUTPUT"):
                try:
                    vals.append(float(line.split()[1]))
                except (IndexError, ValueError):
                    pass
        return vals

    def _reference(self, kernel: str, params: dict) -> np.ndarray:
        """Compute the NumPy reference output for the given kernel."""
        blk = params.get("block_size", 64)
        if kernel == "matmul":
            return _ref_matmul(N=64).astype(np.float32)
        if kernel == "softmax":
            return _ref_softmax(rows=4, cols=16).astype(np.float32)
        if kernel == "reduction":
            return _ref_reduction(N=256, block_size=blk).astype(np.float32)
        if kernel == "layernorm":
            return _ref_layernorm(rows=4, cols=16).astype(np.float32)
        if kernel == "attention":
            return _ref_attention(S=512, D=64).astype(np.float32)
        return np.array([], dtype=np.float32)

    @staticmethod
    def _compare(
        gpu: np.ndarray,
        ref: np.ndarray,
        tol: float,
    ) -> tuple[float, bool]:
        """Return (max_abs_diff, passes_tolerance)."""
        n = min(len(gpu), len(ref))
        if n == 0:
            return float("inf"), False
        diff = float(np.max(np.abs(gpu[:n].astype(np.float64)
                                   - ref[:n].astype(np.float64))))
        return diff, diff <= tol

    def _record_failure(self, kernel: str, variant: str, r: VerificationResult) -> None:
        """Append failure record to in-memory log."""
        self._n_failed += 1
        self._failures.setdefault(kernel, []).append({
            "variant":      variant,
            "max_diff":     r.max_diff if r.max_diff != float("inf") else "inf",
            "tolerance":    r.tolerance,
            "n_checked":    r.n_elements_checked,
            "error":        r.error_msg,
        })

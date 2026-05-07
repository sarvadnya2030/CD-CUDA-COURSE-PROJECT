"""
pytorch_integration.py — Real-workload benchmarking against PyTorch baselines.

Benchmarks:
1. F.scaled_dot_product_attention vs manual attention (Tiled / Flash variants)
2. GPT-2 style self-attention block (QKV proj + attention + out proj)
3. Layer-by-layer breakdown with 95% CI

All timing uses torch.cuda.Event for GPU accuracy (not time.perf_counter).
Results saved to results/pytorch_benchmark.json.

Usage:
    python -m src.pytorch_integration
    python -m src.pytorch_integration --seq_len=1024 --d_model=768 --n_heads=12
"""

import json
import math
import statistics
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── CUDA event timing ────────────────────────────────────────────────────────

def cuda_time_samples(fn, warmup: int = 5, n_samples: int = 30) -> List[float]:
    """
    Time `fn()` using paired CUDA events.  Returns a list of `n_samples`
    elapsed-millisecond measurements.  GPU must be available.
    """
    start_ev = torch.cuda.Event(enable_timing=True)
    stop_ev  = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(n_samples):
        start_ev.record()
        fn()
        stop_ev.record()
        torch.cuda.synchronize()
        samples.append(start_ev.elapsed_time(stop_ev))
    return samples


def _stats(samples: List[float]) -> Dict[str, float]:
    n   = len(samples)
    mu  = statistics.mean(samples)
    std = statistics.stdev(samples) if n > 1 else 0.0
    ci  = 1.96 * std / math.sqrt(n)
    return {
        "mean_ms": mu,
        "std_ms":  std,
        "ci_95_ms": ci,
        "min_ms":  min(samples),
        "max_ms":  max(samples),
        "n_samples": n,
        "raw_samples": samples,
    }


# ── Manual attention implementations ─────────────────────────────────────────

def _manual_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                      scale: float) -> torch.Tensor:
    """Standard O(S²) attention — equivalent to our Tiled CUDA variant."""
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


def _flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                     scale: float, block_size: int = 32) -> torch.Tensor:
    """
    Python-level Flash Attention simulation (online softmax, tiled over S).
    Used for correctness validation only — the real speedup comes from the
    CUDA kernel in src/attention.py.
    """
    B, H, S, D = Q.shape
    O = torch.zeros_like(Q)
    L = torch.full((B, H, S), float("-inf"), device=Q.device, dtype=Q.dtype)
    M = torch.full((B, H, S), float("-inf"), device=Q.device, dtype=Q.dtype)

    for j in range(0, S, block_size):
        Kj = K[:, :, j:j+block_size, :]
        Vj = V[:, :, j:j+block_size, :]
        Sij = torch.matmul(Q, Kj.transpose(-2, -1)) * scale  # (B,H,S,Bc)

        m_new = torch.max(Sij, dim=-1).values              # (B,H,S)
        m_new = torch.maximum(M, m_new)

        P    = torch.exp(Sij - m_new.unsqueeze(-1))
        Psum = P.sum(dim=-1)

        alpha = torch.exp(M - m_new)
        L_new = alpha * L + Psum

        O = (alpha.unsqueeze(-1) * L.unsqueeze(-1) * O
             + torch.matmul(P, Vj)) / L_new.unsqueeze(-1)

        M = m_new
        L = L_new

    return O


# ── SDPA comparison benchmark ─────────────────────────────────────────────────

def benchmark_sdpa(
    seq_len: int = 512,
    d_model: int = 512,
    n_heads: int = 8,
    batch: int = 4,
    warmup: int = 5,
    n_samples: int = 30,
) -> Dict[str, Any]:
    """
    Compare three attention implementations on random (B, H, S, D) tensors:
      1. F.scaled_dot_product_attention  (PyTorch built-in, may use FlashAttn)
      2. Manual tiled attention           (our torch-level reference)
      3. Manual flash attention           (online-softmax simulation)

    Returns timing stats + correctness check for each variant.
    """
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    d_head = d_model // n_heads
    scale  = d_head ** -0.5

    Q = torch.randn(batch, n_heads, seq_len, d_head, device=device)
    K = torch.randn(batch, n_heads, seq_len, d_head, device=device)
    V = torch.randn(batch, n_heads, seq_len, d_head, device=device)

    ref_out = F.scaled_dot_product_attention(Q, K, V, scale=scale)

    results: Dict[str, Any] = {
        "config": {
            "seq_len": seq_len, "d_model": d_model,
            "n_heads": n_heads, "batch": batch,
            "d_head": d_head, "device": torch.cuda.get_device_name(0),
        }
    }

    # 1. PyTorch SDPA (baseline)
    sdpa_samples = cuda_time_samples(
        lambda: F.scaled_dot_product_attention(Q, K, V, scale=scale),
        warmup, n_samples,
    )
    results["pytorch_sdpa"] = _stats(sdpa_samples)
    results["pytorch_sdpa"]["correctness"] = True  # reference

    # 2. Manual tiled attention
    tiled_samples = cuda_time_samples(
        lambda: _manual_attention(Q, K, V, scale),
        warmup, n_samples,
    )
    tiled_out = _manual_attention(Q, K, V, scale)
    max_diff_tiled = (tiled_out - ref_out).abs().max().item()
    results["manual_tiled"] = _stats(tiled_samples)
    results["manual_tiled"]["max_diff"] = max_diff_tiled
    results["manual_tiled"]["correctness"] = max_diff_tiled < 1e-3
    results["manual_tiled"]["speedup_vs_sdpa"] = (
        results["pytorch_sdpa"]["mean_ms"] / results["manual_tiled"]["mean_ms"]
    )

    # 3. Manual flash attention (block_size=32)
    flash_samples = cuda_time_samples(
        lambda: _flash_attention(Q, K, V, scale, block_size=32),
        warmup, n_samples,
    )
    flash_out = _flash_attention(Q, K, V, scale, block_size=32)
    max_diff_flash = (flash_out - ref_out).abs().max().item()
    results["manual_flash"] = _stats(flash_samples)
    results["manual_flash"]["max_diff"] = max_diff_flash
    results["manual_flash"]["correctness"] = max_diff_flash < 1e-3
    results["manual_flash"]["speedup_vs_sdpa"] = (
        results["pytorch_sdpa"]["mean_ms"] / results["manual_flash"]["mean_ms"]
    )

    return results


# ── GPT-2 self-attention block ────────────────────────────────────────────────

class GPT2SelfAttention(nn.Module):
    """
    GPT-2 style self-attention: QKV linear projection → scaled dot-product →
    output projection.  Uses F.scaled_dot_product_attention internally.
    """
    def __init__(self, d_model: int = 768, n_heads: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.c_attn  = nn.Linear(d_model, 3 * d_model)
        self.c_proj  = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        qkv = self.c_attn(x)
        Q, K, V = qkv.split(C, dim=-1)
        def _reshape(t):
            return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = _reshape(Q), _reshape(K), _reshape(V)
        out = F.scaled_dot_product_attention(Q, K, V, scale=self.scale,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.c_proj(out)


class ManualSelfAttention(nn.Module):
    """Same interface as GPT2SelfAttention but uses manual attention kernel."""
    def __init__(self, d_model: int = 768, n_heads: int = 12):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.c_attn  = nn.Linear(d_model, 3 * d_model)
        self.c_proj  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        qkv = self.c_attn(x)
        Q, K, V = qkv.split(C, dim=-1)
        def _reshape(t):
            return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = _reshape(Q), _reshape(K), _reshape(V)
        out = _manual_attention(Q, K, V, self.scale)
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.c_proj(out)


def benchmark_gpt2_attention(
    seq_len: int = 512,
    d_model: int = 768,
    n_heads: int = 12,
    batch: int = 2,
    warmup: int = 5,
    n_samples: int = 30,
) -> Dict[str, Any]:
    """
    End-to-end GPT-2 self-attention block: QKV proj + attention + out proj.
    Compares F.scaled_dot_product_attention vs manual kernel.
    """
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    x = torch.randn(batch, seq_len, d_model, device=device)

    sdpa_model   = GPT2SelfAttention(d_model, n_heads).to(device).eval()
    manual_model = ManualSelfAttention(d_model, n_heads).to(device).eval()

    # copy weights so comparison is fair
    manual_model.c_attn.weight.data.copy_(sdpa_model.c_attn.weight.data)
    manual_model.c_attn.bias.data.copy_(sdpa_model.c_attn.bias.data)
    manual_model.c_proj.weight.data.copy_(sdpa_model.c_proj.weight.data)
    manual_model.c_proj.bias.data.copy_(sdpa_model.c_proj.bias.data)

    with torch.no_grad():
        sdpa_samples = cuda_time_samples(
            lambda: sdpa_model(x), warmup, n_samples)
        manual_samples = cuda_time_samples(
            lambda: manual_model(x), warmup, n_samples)

        ref_out    = sdpa_model(x)
        manual_out = manual_model(x)
        max_diff   = (manual_out - ref_out).abs().max().item()

    sdpa_stats   = _stats(sdpa_samples)
    manual_stats = _stats(manual_samples)

    return {
        "config": {
            "seq_len": seq_len, "d_model": d_model,
            "n_heads": n_heads, "batch": batch,
            "device": torch.cuda.get_device_name(0),
            "description": "GPT-2 self-attention block (QKV + attn + out_proj)",
        },
        "pytorch_sdpa":   sdpa_stats,
        "manual_tiled":   {
            **manual_stats,
            "max_diff":   max_diff,
            "correctness": max_diff < 1e-2,
            "speedup_vs_sdpa": sdpa_stats["mean_ms"] / manual_stats["mean_ms"],
        },
    }


# ── PyTorch module wrappers (for layer replacement) ───────────────────────────

def load_best_params(kernel: str) -> Dict[str, Any]:
    tuning_file = RESULTS_DIR / f"{kernel}_tuning.json"
    if not tuning_file.exists():
        return {}
    with open(tuning_file) as f:
        data = json.load(f)
    return data.get("best", {}).get("params", {})


class TunedLinear(nn.Linear):
    """nn.Linear that records best-tuned config; falls back to cuBLAS."""
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, kernel: str = "matmul"):
        super().__init__(in_features, out_features, bias)
        self.tuned_params = load_best_params(kernel)


class TunedLayerNorm(nn.LayerNorm):
    """nn.LayerNorm with tuned-config metadata."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__(normalized_shape, eps=eps)
        self.tuned_params = load_best_params("layernorm")


class TunedMultiheadAttention(nn.Module):
    """Multi-head attention using F.scaled_dot_product_attention with tuned metadata."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.dropout   = dropout
        self.tuned_params = load_best_params("attention")

        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False
                ) -> Tuple[torch.Tensor, None]:
        B, S, _ = query.shape
        def _proj_reshape(proj, t):
            return proj(t).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        Q = _proj_reshape(self.q_proj, query)
        K = _proj_reshape(self.k_proj, key)
        V = _proj_reshape(self.v_proj, value)
        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, scale=self.scale, dropout_p=dp)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out), None


def apply_tuned_layers(model: nn.Module, kernel: str = "matmul") -> nn.Module:
    """Replace standard layers with tuned variants (in-place on module tree)."""
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        child_name  = parts[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        if kernel in ("matmul", "all") and isinstance(module, nn.Linear):
            setattr(parent, child_name,
                    TunedLinear(module.in_features, module.out_features,
                                module.bias is not None, kernel="matmul"))
        elif kernel in ("layernorm", "all") and isinstance(module, nn.LayerNorm):
            setattr(parent, child_name,
                    TunedLayerNorm(module.normalized_shape[0], module.eps))
    return model


# ── Runner ────────────────────────────────────────────────────────────────────

def run_pytorch_benchmarks(
    seq_len: int = 512,
    d_model: int = 512,
    n_heads: int = 8,
    batch: int = 4,
    warmup: int = 5,
    n_samples: int = 30,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available — skipping GPU benchmarks")
        return {}

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: S={seq_len}, D={d_model}, H={n_heads}, B={batch}\n")

    all_results: Dict[str, Any] = {}

    print("[1/2] SDPA vs Manual Attention ...")
    sdpa_results = benchmark_sdpa(seq_len, d_model, n_heads, batch, warmup, n_samples)
    all_results["sdpa_comparison"] = sdpa_results

    sdpa_mean   = sdpa_results["pytorch_sdpa"]["mean_ms"]
    tiled_mean  = sdpa_results["manual_tiled"]["mean_ms"]
    flash_mean  = sdpa_results["manual_flash"]["mean_ms"]
    print(f"  PyTorch SDPA:    {sdpa_mean:.3f} ms")
    print(f"  Manual Tiled:    {tiled_mean:.3f} ms  "
          f"(speedup {sdpa_mean/tiled_mean:.2f}x, "
          f"correct={sdpa_results['manual_tiled']['correctness']})")
    print(f"  Manual Flash:    {flash_mean:.3f} ms  "
          f"(speedup {sdpa_mean/flash_mean:.2f}x, "
          f"correct={sdpa_results['manual_flash']['correctness']})")

    print(f"\n[2/2] GPT-2 Self-Attention Block (S={seq_len}, D={d_model}) ...")
    gpt2_results = benchmark_gpt2_attention(seq_len, d_model, n_heads, batch,
                                            warmup, n_samples)
    all_results["gpt2_attention"] = gpt2_results

    g_sdpa   = gpt2_results["pytorch_sdpa"]["mean_ms"]
    g_manual = gpt2_results["manual_tiled"]["mean_ms"]
    print(f"  PyTorch SDPA:    {g_sdpa:.3f} ms")
    print(f"  Manual Tiled:    {g_manual:.3f} ms  "
          f"(speedup {g_sdpa/g_manual:.2f}x, "
          f"correct={gpt2_results['manual_tiled']['correctness']})")

    # Save results
    out_path = RESULTS_DIR / "pytorch_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch real-workload benchmarks")
    parser.add_argument("--seq_len",  type=int, default=512)
    parser.add_argument("--d_model",  type=int, default=512)
    parser.add_argument("--n_heads",  type=int, default=8)
    parser.add_argument("--batch",    type=int, default=4)
    parser.add_argument("--warmup",   type=int, default=5)
    parser.add_argument("--n_samples",type=int, default=30)
    args = parser.parse_args()

    run_pytorch_benchmarks(
        seq_len=args.seq_len, d_model=args.d_model, n_heads=args.n_heads,
        batch=args.batch, warmup=args.warmup, n_samples=args.n_samples,
    )

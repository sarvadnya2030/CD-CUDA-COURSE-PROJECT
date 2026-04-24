"""
parser.py — CUDA kernel analyzer.

Extracts optimization-relevant parameters from a .cu source file.

Two parser implementations are provided and auto-selected at import time:
  LibclangKernelParser — AST-based via libclang (more robust)
  RegexKernelParser    — regex heuristics (fallback when libclang unavailable)

Both return the same KernelProfile / KernelInfo dataclasses so the rest
of the pipeline is agnostic to which parser is in use.

Auto-selection:
  If clang.cindex is importable → LibclangKernelParser
  Otherwise                     → RegexKernelParser
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try libclang
try:
    import clang.cindex as _ci
    _LIBCLANG_OK = True
except ImportError:
    _ci = None  # type: ignore
    _LIBCLANG_OK = False


# ── Shared dataclasses (both parsers return these) ─────────────────────────

@dataclass
class MemoryAccessPattern:
    """Memory access characteristics extracted from kernel source."""
    has_global_load:    bool = False
    has_global_store:   bool = False
    has_shared_mem:     bool = False
    has_coalesced_hint: bool = False
    has_strided_access: bool = False
    has_reduction:      bool = False
    has_warp_shuffle:   bool = False


@dataclass
class KernelInfo:
    """
    Detailed kernel metadata extracted by LibclangKernelParser.

    Superset of KernelProfile; the two can be converted via to_profile().
    """
    name:               str
    src_path:           Path
    param_names:        list[str] = field(default_factory=list)
    param_types:        list[str] = field(default_factory=list)
    param_count:        int = 0
    has_shared_mem:     bool = False
    has_unroll:         bool = False
    uses_threadidx:     bool = False
    uses_blockidx:      bool = False
    memory:             MemoryAccessPattern = field(default_factory=MemoryAccessPattern)


@dataclass
class KernelProfile:
    """
    Optimization-focused kernel profile used by generator.py and search.py.

    Both parsers return a list of KernelProfile objects.
    """
    name:           str
    src_path:       Path
    block_dim:      int
    uses_shared:    bool
    loop_depth:     int
    reduction_ops:  list[str] = field(default_factory=list)
    memory:         MemoryAccessPattern = field(default_factory=MemoryAccessPattern)
    raw_params:     dict = field(default_factory=dict)


# ── LibclangKernelParser ───────────────────────────────────────────────────

class LibclangKernelParser:
    """
    AST-based kernel parser using libclang (python-clang binding).

    Walks the CUDA translation unit AST to find __global__ kernel
    declarations and extract structural properties without regex fragility.

    The parser treats the file as C++ (CUDA is a C++ superset) and uses
    clang's built-in CUDA support when available.
    """

    def __init__(self, clang_args: Optional[list[str]] = None) -> None:
        """
        Args:
            clang_args: Extra arguments forwarded to libclang parser.
                        Defaults to ["-x", "cuda", "--cuda-gpu-arch=sm_75"].
        """
        if not _LIBCLANG_OK:
            raise RuntimeError("libclang (python package 'clang') is not installed.")
        self._index = _ci.Index.create()
        self._clang_args = clang_args or [
            "-x", "cuda",
            "--cuda-gpu-arch=sm_75",
            "-std=c++17",
        ]

    def parse_file(self, src_path: Path) -> list[KernelInfo]:
        """
        Parse a .cu file and return one KernelInfo per __global__ function.

        Args:
            src_path: Path to the CUDA source file.

        Returns:
            List of KernelInfo objects (one per kernel).
        """
        tu = self._index.parse(
            str(src_path),
            args=self._clang_args,
            options=_ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
        kernels: list[KernelInfo] = []
        src_text = src_path.read_text(encoding="utf-8")

        for cursor in tu.cursor.walk_preorder():
            if cursor.kind != _ci.CursorKind.FUNCTION_DECL:
                continue
            if not cursor.location.file:
                continue
            if str(cursor.location.file.name) != str(src_path):
                continue
            # Check for __global__ attribute
            tokens = list(cursor.get_tokens())
            tok_text = " ".join(t.spelling for t in tokens[:12])
            if "__global__" not in tok_text:
                continue

            name  = cursor.spelling
            body  = src_text[cursor.extent.start.offset: cursor.extent.end.offset]
            params_info = self._extract_params(cursor)

            info = KernelInfo(
                name=name,
                src_path=src_path,
                param_names=params_info[0],
                param_types=params_info[1],
                param_count=len(params_info[0]),
                has_shared_mem=bool(re.search(r"__shared__", body)),
                has_unroll=bool(re.search(r"#pragma\s+unroll", body)),
                uses_threadidx=bool(re.search(r"\bthreadIdx\b", body)),
                uses_blockidx=bool(re.search(r"\bblockIdx\b", body)),
                memory=self._extract_memory(body),
            )
            kernels.append(info)

        return kernels

    def parse_to_profiles(self, src_path: Path) -> list[KernelProfile]:
        """
        Convenience wrapper: parse file and convert to KernelProfile list.

        Args:
            src_path: Path to CUDA source.
        """
        infos = self.parse_file(src_path)
        return [self._to_profile(info, src_path) for info in infos]

    @staticmethod
    def _extract_params(cursor) -> tuple[list[str], list[str]]:
        """Extract parameter names and type strings from a function cursor."""
        names, types = [], []
        for child in cursor.get_children():
            if child.kind == _ci.CursorKind.PARM_DECL:
                names.append(child.spelling)
                types.append(child.type.spelling)
        return names, types

    @staticmethod
    def _extract_memory(body: str) -> MemoryAccessPattern:
        """Derive MemoryAccessPattern from kernel body text."""
        return MemoryAccessPattern(
            has_global_load    = bool(re.search(r"\b[A-Z]\[|\bin\[|\binput\[", body)),
            has_global_store   = bool(re.search(r"\b[A-Z]\[|\bout\[|\boutput\[", body)),
            has_shared_mem     = bool(re.search(r"__shared__", body)),
            has_strided_access = bool(re.search(r"\[\s*\w+\s*\*\s*\w+\s*\+\s*\w+\s*\]", body)),
            has_reduction      = bool(re.search(r"\b\w+\s*\+=", body)),
            has_warp_shuffle   = bool(re.search(r"__shfl", body)),
        )

    @staticmethod
    def _to_profile(info: KernelInfo, src_path: Path) -> KernelProfile:
        """Convert KernelInfo to the KernelProfile format."""
        body = src_path.read_text(encoding="utf-8")
        block_dim = _extract_block_dim(body)
        loop_depth = _count_loop_depth(body)
        reductions = re.findall(r"\b\w+\s*\+=", body)
        return KernelProfile(
            name=info.name,
            src_path=src_path,
            block_dim=block_dim,
            uses_shared=info.has_shared_mem,
            loop_depth=loop_depth,
            reduction_ops=reductions,
            memory=info.memory,
        )


# ── RegexKernelParser ──────────────────────────────────────────────────────

_RE_KERNEL   = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
_RE_SHARED   = re.compile(r"__shared__")
_RE_REDUCTION = re.compile(r"\b\w+\s*\+=")
_RE_SHFL     = re.compile(r"__shfl")
_RE_STRIDE   = re.compile(r"\[\s*\w+\s*\*\s*\w+\s*\+\s*\w+\s*\]")


class RegexKernelParser:
    """
    Regex-based kernel parser (Phase 1 / fallback).

    Uses pattern matching to extract kernel names and structural properties
    without requiring libclang.  Less robust than AST parsing but has no
    external dependencies.
    """

    def parse_file(self, src_path: Path) -> list[KernelProfile]:
        """
        Parse a .cu file and return one KernelProfile per __global__ kernel.

        Args:
            src_path: Path to the CUDA source file.
        """
        return analyze_file(src_path)

    def parse_to_profiles(self, src_path: Path) -> list[KernelProfile]:
        """Alias for parse_file (satisfies unified interface)."""
        return self.parse_file(src_path)


# ── Shared helpers ─────────────────────────────────────────────────────────

def _count_loop_depth(src: str) -> int:
    """Estimate max loop nesting depth by counting nested for-loops."""
    max_depth = depth = 0
    i = 0
    while i < len(src):
        if src[i:i+3] == "for":
            depth += 1
            max_depth = max(max_depth, depth)
        elif src[i] == "}":
            depth = max(0, depth - 1)
        i += 1
    return max_depth


def _extract_block_dim(src: str) -> int:
    """Read a literal block dimension from dim3 or #define, default 16."""
    m = re.search(r"dim3\s+block\(\s*(\d+)", src)
    if m:
        return int(m.group(1))
    m = re.search(r"#define\s+BLOCK_SIZE\s+(\d+)", src)
    if m:
        return int(m.group(1))
    return 16


# ── Original regex functions (kept for backward compat) ────────────────────

def analyze_file(src_path: Path) -> list[KernelProfile]:
    """
    Parse a .cu file using regex heuristics.

    Returns one KernelProfile per __global__ kernel found.
    """
    src = src_path.read_text(encoding="utf-8")
    profiles: list[KernelProfile] = []

    kernel_starts = [(m.group(1), m.start()) for m in _RE_KERNEL.finditer(src)]
    for idx, (name, start) in enumerate(kernel_starts):
        end  = kernel_starts[idx + 1][1] if idx + 1 < len(kernel_starts) else len(src)
        body = src[start:end]

        mem = MemoryAccessPattern(
            has_global_load    = bool(re.search(r"\bA\[|\bin\[|\binput\[", body)),
            has_global_store   = bool(re.search(r"\bC\[|\bout\[|\boutput\[", body)),
            has_shared_mem     = bool(_RE_SHARED.search(body)),
            has_strided_access = bool(_RE_STRIDE.search(body)),
            has_reduction      = bool(_RE_REDUCTION.search(body)),
            has_warp_shuffle   = bool(_RE_SHFL.search(body)),
        )
        reductions = _RE_REDUCTION.findall(body)
        profiles.append(KernelProfile(
            name=name,
            src_path=src_path,
            block_dim=_extract_block_dim(body),
            uses_shared=mem.has_shared_mem,
            loop_depth=_count_loop_depth(body),
            reduction_ops=reductions,
            memory=mem,
        ))

    return profiles


def build_search_space(profile: KernelProfile) -> dict:
    """
    Given a kernel profile, return the candidate parameter grid.

    Heuristic pruning:
      - block_size must be a multiple of 32 (warp size)
      - tile sizes only matter if kernel is not already using shared memory
      - warp shuffle only makes sense for reduction kernels
    """
    space: dict[str, list] = {}

    space["block_size"] = [64, 128, 192, 256]

    if profile.loop_depth >= 1:
        space["unroll"] = [1, 2, 4, 8]
    else:
        space["unroll"] = [1]

    if not profile.uses_shared:
        space["tile_x"] = [16, 32]
        space["tile_y"] = [16, 32]
    else:
        space["tile_x"] = [16]
        space["tile_y"] = [16]

    if profile.memory.has_strided_access:
        space["transpose_b"] = [False, True]
    else:
        space["transpose_b"] = [False]

    if profile.memory.has_reduction:
        space["warp_shuffle"] = [False, True]
    else:
        space["warp_shuffle"] = [False]

    total = 1
    for v in space.values():
        total *= len(v)

    space["_total_variants"] = total
    space["_kernel"]         = profile.name
    return space


# ── Auto-select parser ─────────────────────────────────────────────────────

def make_parser():
    """
    Return the best available parser.

    Prefers LibclangKernelParser if libclang is installed; otherwise falls
    back to RegexKernelParser and emits a RuntimeWarning.
    """
    if _LIBCLANG_OK:
        try:
            p = LibclangKernelParser()
            print("[PARSER] Using LibclangKernelParser (libclang AST)")
            return p
        except Exception as e:
            warnings.warn(f"libclang init failed ({e}); using regex parser.")
    print("[PARSER] Using RegexKernelParser (regex heuristics)")
    return RegexKernelParser()


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
           Path(__file__).parent / "kernels" / "baseline_kernels.cu"

    parser = make_parser()
    profiles = parser.parse_to_profiles(path)
    for p in profiles:
        space = build_search_space(p)
        print(f"\nKernel: {p.name}")
        print(f"  block_dim={p.block_dim}  loop_depth={p.loop_depth}")
        print(f"  shared={p.uses_shared}  strided={p.memory.has_strided_access}")
        print(f"  search space: {space['_total_variants']} variants")
        print(f"  params: {{" + ", ".join(
            f"{k!r}: {v!r}" for k, v in space.items() if not k.startswith("_")
        ) + "}")

"""
parser.py — CUDA kernel analyzer.

Extracts optimization-relevant parameters from a .cu source file and emits a
KernelProfile used by generator.py to build the search space.

Two parser implementations are provided and auto-selected at runtime:

  LibclangKernelParser — AST-based via libclang (more robust, primary)
  RegexKernelParser    — regex heuristics (fallback, no dependencies)

Auto-selection precedence:
  1. If env var CUDA_AUTOTUNER_NO_LIBCLANG is set → regex
  2. Else if clang.cindex importable → libclang
  3. Else → regex

Both parsers return identically-shaped KernelProfile objects (with a
`backend` tag recording which analyzer produced the profile), so downstream
consumers (generator.py, search.py, autotune.py) are agnostic.

Backward-compat shims:
  analyze_file(path, verbose=False) — module-level function used by autotune.
  find_kernel(path, name, verbose=False) — convenience helper used by autotune.
  build_search_space(profile) — used by generator.py and autotune.py.
  make_parser() — returns the best available parser instance.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try libclang
try:
    import clang.cindex as _ci  # provided by the `libclang` PyPI package
    _LIBCLANG_OK = True
except ImportError:
    _ci = None  # type: ignore
    _LIBCLANG_OK = False

# An explicit env var lets the user force the regex fallback.
if os.environ.get("CUDA_AUTOTUNER_NO_LIBCLANG"):
    _LIBCLANG_OK = False

# Optional override for the libclang shared library path.
if _LIBCLANG_OK:
    _lib = os.environ.get("LIBCLANG_LIBRARY_FILE")
    if _lib:
        try:
            _ci.Config.set_library_file(_lib)
        except Exception:
            pass


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

    Superset of KernelProfile; the two can be converted via _to_profile().
    """
    name:           str
    src_path:       Path
    param_names:    list[str] = field(default_factory=list)
    param_types:    list[str] = field(default_factory=list)
    param_count:    int = 0
    has_shared_mem: bool = False
    has_unroll:     bool = False
    uses_threadidx: bool = False
    uses_blockidx:  bool = False
    memory:         MemoryAccessPattern = field(default_factory=MemoryAccessPattern)


@dataclass
class KernelProfile:
    """
    Optimization-focused kernel profile used by generator.py and search.py.

    Both parsers return a list of KernelProfile objects.  The `backend` field
    records which analyzer produced the profile ("libclang" | "regex" |
    "hardcoded").
    """
    name:           str
    src_path:       Path
    block_dim:      int
    uses_shared:    bool
    loop_depth:     int
    reduction_ops:  list[str] = field(default_factory=list)
    memory:         MemoryAccessPattern = field(default_factory=MemoryAccessPattern)
    raw_params:     dict = field(default_factory=dict)
    backend:        str = "regex"


# ── Heuristic name sets used by the libclang AST visitor ───────────────────

_LOAD_BASES  = {"A", "B", "in", "input", "ir", "gamma", "beta", "sdata"}
_STORE_BASES = {"C", "out", "output", "or_", "sdata"}


# ── LibclangKernelParser ───────────────────────────────────────────────────

# Flags that let libclang's frontend digest a .cu file without needing the
# full CUDA header tree installed on the host running the tuner.
_DEFAULT_CLANG_ARGS = [
    "-x", "cuda",
    "-std=c++17",
    "--cuda-host-only",
    "-nocudainc",
    "-nocudalib",
    "-ferror-limit=0",
    "-Wno-everything",
]


class LibclangKernelParser:
    """
    AST-based kernel parser using libclang (python-clang binding).

    Walks the CUDA translation unit AST to find __global__ kernel
    declarations and extract structural properties without regex fragility.
    """

    def __init__(self, clang_args: Optional[list[str]] = None) -> None:
        if not _LIBCLANG_OK:
            raise RuntimeError("libclang (python package 'clang') is not installed.")
        self._index = _ci.Index.create()
        self._clang_args = clang_args or list(_DEFAULT_CLANG_ARGS)

    # ── Detailed kernel info (libclang-only) ───────────────────────────

    def parse_file(self, src_path: Path) -> list[KernelInfo]:
        """Parse a .cu file and return one KernelInfo per __global__ function."""
        tu = self._index.parse(str(src_path), args=self._clang_args)
        kernels: list[KernelInfo] = []
        src_text = src_path.read_text(encoding="utf-8")

        for cursor in tu.cursor.walk_preorder():
            if cursor.kind != _ci.CursorKind.FUNCTION_DECL:
                continue
            if not cursor.location.file:
                continue
            if str(cursor.location.file.name) != str(src_path):
                continue
            if not _is_cuda_kernel(cursor):
                continue

            name  = cursor.spelling
            body  = src_text[cursor.extent.start.offset: cursor.extent.end.offset]
            param_names, param_types = self._extract_params(cursor)

            info = KernelInfo(
                name=name,
                src_path=src_path,
                param_names=param_names,
                param_types=param_types,
                param_count=len(param_names),
                has_shared_mem=bool(re.search(r"__shared__", body)),
                has_unroll=bool(re.search(r"#pragma\s+unroll", body)),
                uses_threadidx=bool(re.search(r"\bthreadIdx\b", body)),
                uses_blockidx=bool(re.search(r"\bblockIdx\b", body)),
                memory=self._extract_memory(body),
            )
            kernels.append(info)

        return kernels

    # ── KernelProfile pathway (matches RegexKernelParser) ──────────────

    def parse_to_profiles(self, src_path: Path) -> list[KernelProfile]:
        """Parse `src_path` and return one KernelProfile per __global__ kernel."""
        tu = self._index.parse(str(src_path), args=self._clang_args)
        profiles: list[KernelProfile] = []

        for cursor in tu.cursor.walk_preorder():
            if cursor.kind != _ci.CursorKind.FUNCTION_DECL:
                continue
            if not cursor.is_definition():
                continue
            if not _is_cuda_kernel(cursor):
                continue

            v = _LibclangVisitor()
            v.visit(cursor)
            reductions = _count_reductions_from_tokens(cursor)

            mem = MemoryAccessPattern(
                has_global_load    = v.has_global_load,
                has_global_store   = v.has_global_store,
                has_shared_mem     = v.uses_shared,
                has_strided_access = v.has_strided_access,
                has_reduction      = reductions > 0,
                has_warp_shuffle   = v.has_warp_shuffle,
            )
            profiles.append(KernelProfile(
                name          = cursor.spelling,
                src_path      = src_path,
                block_dim     = _extract_block_dim_from_tu(tu.cursor),
                uses_shared   = v.uses_shared,
                loop_depth    = v.max_loop_depth,
                reduction_ops = ["+="] * reductions,
                memory        = mem,
                backend       = "libclang",
            ))
        return profiles

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
            has_reduction      = bool(re.search(r"\+=", body)),
            has_warp_shuffle   = bool(re.search(r"__shfl", body)),
        )


# ── Libclang AST visitor helpers ───────────────────────────────────────────

def _is_cuda_kernel(cursor) -> bool:
    """A FUNCTION_DECL whose signature tokens contain `__global__`."""
    if not _LIBCLANG_OK:
        return False
    for tok in cursor.get_tokens():
        spell = tok.spelling
        if spell == "__global__":
            return True
        if spell == "(":          # reached the parameter list
            return False
    return False


class _LibclangVisitor:
    """
    Walk a libclang FUNCTION_DECL subtree and accumulate the fields that
    feed KernelProfile / MemoryAccessPattern.
    """

    def __init__(self) -> None:
        self.uses_shared        = False
        self.max_loop_depth     = 0
        self.has_warp_shuffle   = False
        self.has_strided_access = False
        self.has_global_load    = False
        self.has_global_store   = False

    def visit(self, cursor, loop_depth: int = 0):
        if not _LIBCLANG_OK:
            return
        CursorKind = _ci.CursorKind
        for child in cursor.get_children():
            new_depth = loop_depth
            if child.kind == CursorKind.FOR_STMT:
                new_depth = loop_depth + 1
                if new_depth > self.max_loop_depth:
                    self.max_loop_depth = new_depth
            self._inspect(child, CursorKind)
            self.visit(child, new_depth)

    def _inspect(self, cursor, CursorKind):
        kind = cursor.kind

        if kind == CursorKind.VAR_DECL:
            for tok in cursor.get_tokens():
                if tok.spelling == "__shared__":
                    self.uses_shared = True
                    break
            return

        if kind == CursorKind.CALL_EXPR:
            name = cursor.spelling or ""
            if name.startswith("__shfl"):
                self.has_warp_shuffle = True
            return

        if kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            children = list(cursor.get_children())
            if len(children) >= 2:
                idx_tokens = [t.spelling for t in children[1].get_tokens()]
                if "*" in idx_tokens and "+" in idx_tokens:
                    self.has_strided_access = True
            if children:
                base_tokens = [t.spelling for t in children[0].get_tokens()]
                if base_tokens:
                    base = base_tokens[0]
                    if base in _LOAD_BASES:
                        self.has_global_load = True
                    if base in _STORE_BASES:
                        self.has_global_store = True
            return


def _count_reductions_from_tokens(cursor) -> int:
    """Scan the token stream of a function body for `+=` occurrences."""
    count = 0
    for tok in cursor.get_tokens():
        if tok.spelling == "+=":
            count += 1
    return count


def _extract_block_dim_from_tu(tu_cursor) -> int:
    """Walk the TU for a `dim3 block(N, ...)` literal; default 16."""
    default = 16
    tokens = [t.spelling for t in tu_cursor.get_tokens()]
    for i, tok in enumerate(tokens):
        if tok == "dim3" and i + 3 < len(tokens):
            if tokens[i + 1] == "block" and tokens[i + 2] == "(":
                try:
                    return int(tokens[i + 3])
                except ValueError:
                    continue
    return default


# ── RegexKernelParser ──────────────────────────────────────────────────────

_RE_KERNEL    = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
_RE_SHARED    = re.compile(r"__shared__")
_RE_REDUCTION = re.compile(r"\+=")
_RE_SHFL      = re.compile(r"__shfl")
_RE_STRIDE    = re.compile(r"\[\s*\w+\s*\*\s*\w+\s*\+\s*\w+\s*\]")
_RE_DIM3      = re.compile(r"dim3\s+block\(\s*(\d+)")
_RE_DEFINE    = re.compile(r"#define\s+BLOCK_SIZE\s+(\d+)")


class RegexKernelParser:
    """
    Regex-based kernel parser (Phase 1 / fallback).

    Pattern matching to extract kernel names and structural properties
    without requiring libclang.  Less robust than AST parsing but has no
    external dependencies.
    """

    def parse_file(self, src_path: Path) -> list[KernelProfile]:
        return analyze_file(src_path)

    def parse_to_profiles(self, src_path: Path) -> list[KernelProfile]:
        """Alias for parse_file (satisfies unified interface)."""
        return self.parse_file(src_path)


# ── Shared helpers (regex-only loop / block_dim extraction) ────────────────

def _count_loop_depth(src: str) -> int:
    """
    Estimate max nesting depth of `for` loops via brace tracking.
    Word-boundary aware so substrings like "before" don't match.
    """
    max_depth = depth = 0
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "}":
            depth = max(0, depth - 1)
        prev_ok = (i == 0) or not (src[i - 1].isalnum() or src[i - 1] == "_")
        if (prev_ok and src[i:i + 3] == "for"
                and i + 3 < n and src[i + 3] in " ("):
            depth += 1
            if depth > max_depth:
                max_depth = depth
            i += 3
            continue
        i += 1
    return max_depth


def _extract_block_dim(src: str) -> int:
    m = _RE_DIM3.search(src)
    if m:
        return int(m.group(1))
    m = _RE_DEFINE.search(src)
    if m:
        return int(m.group(1))
    return 16


# ── Public regex-fallback API (also used as the universal entry point) ────

def analyze_file(src_path: Path, verbose: bool = False) -> list[KernelProfile]:
    """
    Parse `src_path` and return one KernelProfile per __global__ kernel.

    Tries libclang first, then falls back to regex if unavailable or on
    parse failure (zero kernels found).
    """
    if _LIBCLANG_OK:
        try:
            parser = LibclangKernelParser()
            profiles = parser.parse_to_profiles(src_path)
            if profiles:
                if verbose:
                    print(f"[parser] libclang: {len(profiles)} kernel(s) "
                          f"from {src_path.name}")
                return profiles
            if verbose:
                print(f"[parser] libclang returned 0 kernels for "
                      f"{src_path.name}; falling back to regex.")
        except Exception as e:
            if verbose:
                print(f"[parser] libclang error on {src_path.name}: {e}; "
                      f"falling back to regex.")

    return _regex_analyze(src_path)


def _regex_analyze(src_path: Path) -> list[KernelProfile]:
    """Pure-regex parse (no libclang). Used as the fallback path."""
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
            block_dim=_extract_block_dim(src),
            uses_shared=mem.has_shared_mem,
            loop_depth=_count_loop_depth(body),
            reduction_ops=["+="] * len(reductions),
            memory=mem,
            backend="regex",
        ))
    return profiles


def find_kernel(src_path: Path, kernel: str,
                verbose: bool = False) -> Optional[KernelProfile]:
    """
    Return the first profile whose function name starts with `kernel`.
    Example: find_kernel(path, "matmul") → profile for `matmul_naive`.
    """
    for p in analyze_file(src_path, verbose=verbose):
        if p.name.startswith(kernel):
            return p
    return None


def build_search_space(profile: KernelProfile) -> dict:
    """
    Given a kernel profile, return the candidate parameter grid.

    Pruning rules:
      - block_size drives 1D thread blocks (softmax/reduction/layernorm)
      - tile_x/tile_y drive matmul 2D tiling (forced equal in enumeration)
      - transpose_b only helps when strided access was detected (matmul)
      - warp_shuffle only helps when reductions are present (reduction)
      - reg_tile (1xRT per-thread register blocking) is matmul-only
    """
    space: dict[str, list] = {}
    is_matmul = profile.name.startswith("matmul")

    space["block_size"] = [64, 128, 192, 256]

    space["unroll"] = [1, 2, 4, 8] if profile.loop_depth >= 1 else [1]

    if is_matmul:
        space["tile_x"] = [16, 32]
        space["tile_y"] = [16, 32]
    elif not profile.uses_shared:
        space["tile_x"] = [16, 32]
        space["tile_y"] = [16, 32]
    else:
        space["tile_x"] = [16]
        space["tile_y"] = [16]

    # transpose_b: only matmul's template reads it.
    if is_matmul and profile.memory.has_strided_access:
        space["transpose_b"] = [False, True]
    elif profile.memory.has_strided_access:
        space["transpose_b"] = [False, True]
    else:
        space["transpose_b"] = [False]

    # warp_shuffle: only the reduction template branches on it.
    if profile.memory.has_reduction:
        space["warp_shuffle"] = [False, True]
    else:
        space["warp_shuffle"] = [False]

    # Register blocking: only meaningful for matmul-shaped kernels.
    if is_matmul:
        space["reg_tile"] = [1, 2, 4]

    total = 1
    for v in space.values():
        total *= len(v)
    space["_total_variants"] = total
    space["_kernel"]         = profile.name
    return space


# ── Auto-select parser ─────────────────────────────────────────────────────

def make_parser():
    """
    Return the best available parser instance.

    Prefers LibclangKernelParser when libclang is importable; otherwise
    falls back to RegexKernelParser and emits a RuntimeWarning.
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


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
           Path(__file__).parent / "kernels" / "baseline_kernels.cu"

    profiles = analyze_file(path, verbose=True)
    if not profiles:
        print(f"No kernels found in {path}")
        sys.exit(1)

    for p in profiles:
        space = build_search_space(p)
        print(f"\nKernel: {p.name}   [backend={p.backend}]")
        print(f"  block_dim={p.block_dim}  loop_depth={p.loop_depth}")
        print(f"  shared={p.uses_shared}  strided={p.memory.has_strided_access}  "
              f"reduction={p.memory.has_reduction}  shfl={p.memory.has_warp_shuffle}")
        print(f"  reduction_ops={p.reduction_ops}")
        print(f"  search space: {space['_total_variants']} variants")
        print(f"  params: { {k: v for k, v in space.items() if not k.startswith('_')} }")

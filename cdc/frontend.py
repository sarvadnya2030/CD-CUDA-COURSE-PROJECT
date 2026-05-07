"""
frontend.py — orchestrates the compiler frontend pipeline.

    raw .cu file
         │
         ▼  preprocessor.preprocess
    cleaned source
         │
         ▼  preprocessor.extract_device_functions
    list of (kernel snippet, line)
         │
         ▼  parser.parse           ┐
                                   │  per kernel
         ▼  symbol_table.build_for_function
                                   │
         ▼  type_check.TypeChecker┘
    list of FrontendKernel objects (AST + SymTab + diagnostics)

This is the public entry-point that the rest of the autotuner can call
when `--use-cdc-frontend` is requested.

Maps to syllabus
----------------
* Course Unit I  — Compiler phases / pass orchestration
* Course Unit IV — Front-end output: AST + symbol table + type info
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import ast_nodes as A
from .preprocessor import preprocess, extract_device_functions
from .parser import parse
from .symbol_table import SymbolTable, build_for_function
from .type_check import TypeChecker, Diagnostic, check_function


@dataclass
class FrontendKernel:
    """Result of running the full frontend on one kernel."""
    name: str
    line: int
    source: str
    ast: A.FunctionDef
    symbols: SymbolTable
    scopes: List = field(default_factory=list)        # snapshotted Scope objects
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def ok(self) -> bool:
        return not [d for d in self.diagnostics if d.severity == "error"]

    def dump_scopes(self) -> str:
        lines = []
        for sc in self.scopes:
            lines.append(f"-- scope[{sc.level}] {sc.kind} --")
            if not sc.symbols:
                lines.append("  (empty)")
                continue
            for name in sorted(sc.symbols):
                sym = sc.symbols[name]
                tag = "*" if sym.used else " "
                lines.append(f"  {tag} {sym}")
        return "\n".join(lines)


@dataclass
class FrontendResult:
    """Full-file frontend result — one entry per __global__/__device__ kernel."""
    path: Optional[Path]
    kernels: List[FrontendKernel] = field(default_factory=list)

    @property
    def kernel_names(self) -> list[str]:
        return [k.name for k in self.kernels]

    def by_name(self, name: str) -> Optional[FrontendKernel]:
        for k in self.kernels:
            if k.name == name:
                return k
        return None

    def all_diagnostics(self) -> List[Diagnostic]:
        out = []
        for k in self.kernels:
            out.extend(k.diagnostics)
        return out

    def ok(self) -> bool:
        return all(k.ok() for k in self.kernels)


# ── Public API ─────────────────────────────────────────────────────────────

def run_frontend(path: str | Path | None = None,
                 source: str | None = None) -> FrontendResult:
    """
    Run the full Phase-1 frontend on a CUDA source file or string.

    Either `path` or `source` must be provided.  Returns a FrontendResult
    with one FrontendKernel per __global__/__device__ function found.
    """
    if source is None:
        if path is None:
            raise ValueError("run_frontend: provide path or source")
        p = Path(path)
        source = p.read_text(encoding="utf-8")
        path_obj = p
    else:
        path_obj = Path(path) if path is not None else None

    cleaned = preprocess(source)
    snippets = extract_device_functions(source)

    kernels: List[FrontendKernel] = []
    for snippet, lineno in snippets:
        try:
            tu = parse(snippet)
        except Exception as exc:
            kernels.append(FrontendKernel(
                name="<parse error>", line=lineno, source=snippet,
                ast=A.FunctionDef(),
                symbols=SymbolTable(),
                diagnostics=[Diagnostic("error", lineno, str(exc))],
            ))
            continue

        if not tu.items:
            continue
        fn = tu.items[0]
        # Adjust AST line numbers so they reflect the original file.
        _shift_lines(fn, lineno - 1)

        st, tc = check_function(fn)
        kernels.append(FrontendKernel(
            name=fn.name, line=lineno, source=snippet,
            ast=fn, symbols=st, scopes=list(tc.collected_scopes),
            diagnostics=list(tc.diagnostics),
        ))

    return FrontendResult(path=path_obj, kernels=kernels)


def _shift_lines(node, delta: int) -> None:
    """Walk an AST and offset every node's `line` attribute by `delta`."""
    if isinstance(node, list):
        for n in node:
            _shift_lines(n, delta)
        return
    if not hasattr(node, "__dataclass_fields__"):
        return
    if hasattr(node, "line") and isinstance(node.line, int) and node.line:
        node.line += delta
    for fname in node.__dataclass_fields__:
        val = getattr(node, fname)
        if isinstance(val, list):
            for v in val:
                _shift_lines(v, delta)
        else:
            _shift_lines(val, delta)


# ── Pretty report ──────────────────────────────────────────────────────────

def format_report(result: FrontendResult, show_ast: bool = False,
                  show_symbols: bool = True) -> str:
    """Build a human-readable summary of a FrontendResult."""
    lines = []
    path = str(result.path) if result.path else "<string>"
    lines.append(f"=== Frontend report: {path} ===")
    lines.append(f"  kernels: {len(result.kernels)}")
    for k in result.kernels:
        lines.append("")
        lines.append(f"  -- {k.name}  (line {k.line}, {len(k.ast.params)} params) --")
        lines.append(f"     return type : {k.ast.return_type}")
        lines.append(f"     qualifiers  : {' '.join(k.ast.qualifiers)}")
        for p in k.ast.params:
            lines.append(f"       param: {p.type!s:<30} {p.name}")
        if show_symbols:
            lines.append("     symbols:")
            for line in k.dump_scopes().splitlines():
                lines.append(f"       {line}")
        if k.diagnostics:
            lines.append("     diagnostics:")
            for d in k.diagnostics:
                lines.append(f"       {d}")
        else:
            lines.append("     diagnostics: clean")
        if show_ast:
            lines.append("     AST:")
            for line in A.pretty(k.ast).splitlines():
                lines.append(f"       {line}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    show_ast = "--ast" in args
    args = [a for a in args if a != "--ast"]
    if not args:
        print("usage: python -m cdc.frontend <file.cu> [--ast]", file=sys.stderr)
        sys.exit(1)
    res = run_frontend(args[0])
    print(format_report(res, show_ast=show_ast))

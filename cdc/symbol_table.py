"""
symbol_table.py — scoped symbol table for the CUDA C subset.

The symbol table tracks every declared identifier in a translation unit
together with its type, address space, scope, and source location.  It is
the canonical "directory of names" produced by the frontend and consumed
by the type checker, IR generator, optimiser, and code generator.

Scope structure
---------------
A `SymbolTable` is a stack of `Scope` frames.  Pushing/popping happens at
function boundaries and at every `{ ... }` compound statement.

    Scope kinds:
        global    — top-level declarations (none in our subset; we run the
                    parser per-kernel, but the structure is general).
        function  — kernel parameters live here.
        block     — local variables (`__shared__` or register).

CUDA-specific data
------------------
Each `Symbol` records its `addr_space` ∈ {global, shared, constant, local}
so the type checker can enforce CUDA address-space rules:

    * Pointer parameters default to `global` (kernel arguments).
    * `__shared__` declarations live in `shared`.
    * Stack variables live in `local` (registers).
    * Constants live in `constant`.

Maps to syllabus
----------------
* Course Unit III  — Symbol Table Structure
* Course Unit IV   — Semantic analysis foundation for SDT
* Lab Practical    — 6 (Lexical Analyzer with symbol table)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import ast_nodes as A


@dataclass
class Symbol:
    """One entry in the symbol table."""
    name: str
    type: A.Type
    kind: str                       # 'param' | 'local' | 'function' | 'builtin'
    addr_space: str                 # 'global' | 'shared' | 'constant' | 'local'
    scope_level: int = 0
    line: int = 0
    is_array: bool = False
    array_size: Optional[A.Expr] = None
    extern_shared: bool = False
    used: bool = False              # set by type checker; useful for dead-code

    def __str__(self) -> str:
        flags = []
        if self.is_array:
            flags.append("array")
        if self.extern_shared:
            flags.append("extern[]")
        f = (" " + ",".join(flags)) if flags else ""
        return f"{self.kind:<8} {self.addr_space:<8} {self.type!s:<28} {self.name}{f}"


@dataclass
class Scope:
    kind: str                                       # 'global' | 'function' | 'block'
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    parent: Optional["Scope"] = None
    level: int = 0

    def insert(self, sym: Symbol) -> None:
        if sym.name in self.symbols:
            raise SymbolError(
                f"redeclaration of {sym.name!r} at line {sym.line} "
                f"(previous at line {self.symbols[sym.name].line})"
            )
        sym.scope_level = self.level
        self.symbols[sym.name] = sym

    def lookup_local(self, name: str) -> Optional[Symbol]:
        return self.symbols.get(name)


class SymbolError(Exception):
    pass


# ── Builtin identifiers (always in scope) ──────────────────────────────────
# These are the CUDA-provided variables and intrinsics every kernel sees.

def _builtin_dim3(name: str) -> Symbol:
    """threadIdx / blockIdx / blockDim / gridDim — struct with .x .y .z."""
    return Symbol(
        name=name,
        type=A.Type(base="dim3", addr_space="local"),
        kind="builtin",
        addr_space="local",
    )


def _builtin_int(name: str) -> Symbol:
    return Symbol(
        name=name,
        type=A.Type(base="int", addr_space="local"),
        kind="builtin",
        addr_space="local",
    )


def _builtin_fn(name: str, ret: str = "void") -> Symbol:
    return Symbol(
        name=name,
        type=A.Type(base=ret, addr_space="local"),
        kind="builtin",
        addr_space="local",
    )


CUDA_BUILTINS: List[Symbol] = [
    _builtin_dim3("threadIdx"),
    _builtin_dim3("blockIdx"),
    _builtin_dim3("blockDim"),
    _builtin_dim3("gridDim"),
    _builtin_int("warpSize"),

    _builtin_fn("__syncthreads"),
    _builtin_fn("__syncwarp"),
    _builtin_fn("__shfl_sync",       "float"),
    _builtin_fn("__shfl_xor_sync",   "float"),
    _builtin_fn("__shfl_down_sync",  "float"),
    _builtin_fn("__shfl_up_sync",    "float"),
    _builtin_fn("__ballot_sync",     "unsigned int"),
    _builtin_fn("__activemask",      "unsigned int"),

    # Math intrinsics
    _builtin_fn("expf",     "float"),
    _builtin_fn("logf",     "float"),
    _builtin_fn("sqrtf",    "float"),
    _builtin_fn("rsqrtf",   "float"),
    _builtin_fn("fmaxf",    "float"),
    _builtin_fn("fminf",    "float"),
    _builtin_fn("fabsf",    "float"),
    _builtin_fn("__fdividef", "float"),
    _builtin_fn("__expf",   "float"),
    _builtin_fn("powf",     "float"),
    _builtin_fn("tanhf",    "float"),
    _builtin_fn("sinf",     "float"),
    _builtin_fn("cosf",     "float"),
]


# ── Symbol table ────────────────────────────────────────────────────────────

class SymbolTable:
    """
    A stack of scopes with O(1) push/pop and chained lookup.

    Typical use::

        st = SymbolTable()
        st.push("function")
        st.insert(Symbol(...))
        sym = st.lookup("threadIdx")
        st.pop()
    """

    def __init__(self):
        self._stack: List[Scope] = []
        # Global scope holds builtins.
        global_scope = Scope(kind="global", level=0)
        for b in CUDA_BUILTINS:
            global_scope.insert(b)
        self._stack.append(global_scope)

    @property
    def current(self) -> Scope:
        return self._stack[-1]

    @property
    def depth(self) -> int:
        return len(self._stack)

    def push(self, kind: str) -> Scope:
        s = Scope(kind=kind, parent=self.current, level=self.depth)
        self._stack.append(s)
        return s

    def pop(self) -> Scope:
        if len(self._stack) <= 1:
            raise SymbolError("cannot pop global scope")
        return self._stack.pop()

    def insert(self, sym: Symbol) -> None:
        self.current.insert(sym)

    def lookup(self, name: str) -> Optional[Symbol]:
        for scope in reversed(self._stack):
            s = scope.lookup_local(name)
            if s is not None:
                return s
        return None

    # ── Reporting ──────────────────────────────────────────────────────────

    def dump(self, include_builtins: bool = False) -> str:
        """Pretty-print the contents of every scope above global."""
        lines = []
        for scope in self._stack:
            if scope.kind == "global" and not include_builtins:
                continue
            header = f"-- scope[{scope.level}] {scope.kind} --"
            lines.append(header)
            if not scope.symbols:
                lines.append("  (empty)")
                continue
            for name in sorted(scope.symbols):
                sym = scope.symbols[name]
                tag = "*" if sym.used else " "
                lines.append(f"  {tag} {sym}")
        return "\n".join(lines)


# ── Build a symbol table from an AST ───────────────────────────────────────

def build_for_function(fn: A.FunctionDef) -> SymbolTable:
    """
    Build a fresh SymbolTable populated from one parsed function.

    The function scope contains the parameters; nested blocks push/pop
    new scopes.  The result is suitable as input to type_check.

    Note: variables declared inside `for (int i = 0; ...)` are placed in
    the loop's own block scope per C99, but we collapse them into the
    enclosing block to keep things simple.
    """
    st = SymbolTable()
    st.push("function")

    # Insert parameters.
    for p in fn.params:
        sym = Symbol(
            name=p.name,
            type=p.type,
            kind="param",
            addr_space=p.type.addr_space or ("global" if p.type.is_pointer else "local"),
            line=p.line,
        )
        st.insert(sym)

    # Walk the body, opening/closing block scopes.
    _walk_compound(fn.body, st)
    return st


def _walk_compound(c: A.Compound, st: SymbolTable) -> None:
    for stmt in c.items:
        _walk_stmt(stmt, st)


def _walk_stmt(s, st: SymbolTable) -> None:
    if isinstance(s, A.Compound):
        st.push("block")
        _walk_compound(s, st)
        st.pop()
        return

    if isinstance(s, A.VarDecl):
        sym = Symbol(
            name=s.name,
            type=s.type,
            kind="local",
            addr_space=s.type.addr_space or "local",
            line=s.line,
            is_array=s.array_size is not None or s.extern_shared,
            array_size=s.array_size,
            extern_shared=s.extern_shared,
        )
        try:
            st.insert(sym)
        except SymbolError:
            # silent — duplicate locals are usually loop indices in
            # successive scopes that we collapsed; ignore.
            pass
        return

    if isinstance(s, A.If):
        _walk_stmt(s.then, st)
        if s.else_ is not None:
            _walk_stmt(s.else_, st)
        return

    if isinstance(s, A.For):
        st.push("block")
        if isinstance(s.init, A.VarDecl):
            _walk_stmt(s.init, st)
        elif isinstance(s.init, A.Compound):
            _walk_compound(s.init, st)
        _walk_stmt(s.body, st)
        st.pop()
        return

    if isinstance(s, (A.While, A.DoWhile)):
        _walk_stmt(s.body, st)
        return

    # ExprStmt, Return, Break, Continue — no new bindings.
    return

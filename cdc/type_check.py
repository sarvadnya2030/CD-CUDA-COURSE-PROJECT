"""
type_check.py — semantic analysis pass for the CUDA C subset.

Walks the AST in a single combined pass with `SymbolTable`:

  * pushes a function scope for each `FunctionDef`,
  * pushes a block scope for each `Compound` and each `For`,
  * inserts `VarDecl`s into the current scope,
  * resolves every identifier reference,
  * computes a type for every expression,
  * enforces CUDA address-space and type rules.

The pass also fills `_collected_symbols` so callers can recover the full
SymbolTable contents after walking — every scope's symbol map is captured
before the scope is popped.

Maps to syllabus
----------------
* Course Unit III — Type Checking, Type Conversion, Symbol Table
* Course Unit IV  — Semantic errors, Error Detection & Recovery
* Lab Practical   — 6 (Lexical Analyzer with symbol table) and 10 (SDT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import ast_nodes as A
from .symbol_table import (
    Symbol, SymbolTable, SymbolError, Scope,
)


# ── Diagnostics ────────────────────────────────────────────────────────────

@dataclass
class Diagnostic:
    severity: str          # 'error' | 'warning' | 'note'
    line: int
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] line {self.line}: {self.message}"


# ── Type predicates ────────────────────────────────────────────────────────

_NUMERIC = {"int", "unsigned int", "float", "bool"}


def _is_numeric(t: A.Type) -> bool:
    return (not t.is_pointer) and t.base in _NUMERIC


def _is_integer(t: A.Type) -> bool:
    return (not t.is_pointer) and t.base in ("int", "unsigned int", "bool")


def _is_pointer(t: A.Type) -> bool:
    return t.is_pointer


def _is_dim3(t: A.Type) -> bool:
    return (not t.is_pointer) and t.base == "dim3"


def _is_assignable(target_t: A.Type, value_t: A.Type) -> bool:
    if target_t.base == value_t.base and target_t.is_pointer == value_t.is_pointer:
        return True
    if _is_numeric(target_t) and _is_numeric(value_t):
        return True
    if _is_pointer(target_t) and _is_integer(value_t):
        return True   # NULL/0 → pointer
    return False


def _binop_result(op: str, l: A.Type, r: A.Type) -> A.Type:
    if op in ("+", "-") and _is_pointer(l) and _is_integer(r):
        return l
    if op == "+" and _is_pointer(r) and _is_integer(l):
        return r
    if op == "-" and _is_pointer(l) and _is_pointer(r):
        return A.Type(base="int")
    if op in ("==", "!=", "<", ">", "<=", ">=", "&&", "||"):
        return A.Type(base="int")
    if _is_numeric(l) and _is_numeric(r):
        if l.base == "float" or r.base == "float":
            return A.Type(base="float")
        if l.base == "unsigned int" or r.base == "unsigned int":
            return A.Type(base="unsigned int")
        return A.Type(base="int")
    return l


# ── Type checker (combined symbol-table + type pass) ───────────────────────

class TypeChecker:
    """
    Walks an AST while managing a `SymbolTable`.  After `check_function`,
    the snapshotted scopes are stored in `self.collected_scopes` so a
    caller can dump them or attach them to AST nodes for the IR pass.
    """

    def __init__(self, symbols: Optional[SymbolTable] = None):
        self.symbols       = symbols or SymbolTable()
        self.diagnostics:  List[Diagnostic] = []
        self.types: Dict[int, A.Type]       = {}
        # Snapshot of every scope encountered, in order.
        self.collected_scopes: List[Scope]  = []

    # ── Public API ────────────────────────────────────────────────────────

    def check_function(self, fn: A.FunctionDef) -> None:
        if fn.body is None:
            return
        # Function scope holds parameters.
        self.symbols.push("function")
        for p in fn.params:
            sym = Symbol(
                name=p.name,
                type=p.type,
                kind="param",
                addr_space=p.type.addr_space or ("global" if p.type.is_pointer else "local"),
                line=p.line,
            )
            try:
                self.symbols.insert(sym)
            except SymbolError as e:
                self.diagnostics.append(Diagnostic("error", p.line, str(e)))
        # Walk body.  Compound itself pushes a block scope, so the params
        # remain in their own enclosing function scope.
        self._check_compound(fn.body)
        # Snapshot before pop.
        self.collected_scopes.append(self._snapshot(self.symbols.current))
        self.symbols.pop()

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]

    def ok(self) -> bool:
        return not self.errors

    # ── Traversal ─────────────────────────────────────────────────────────

    def _snapshot(self, sc: Scope) -> Scope:
        """Make a shallow copy of a scope (so we can dump after pop)."""
        return Scope(kind=sc.kind, level=sc.level,
                     symbols=dict(sc.symbols))

    def _check_compound(self, c: A.Compound) -> None:
        self.symbols.push("block")
        for stmt in c.items:
            self._check_stmt(stmt)
        self.collected_scopes.append(self._snapshot(self.symbols.current))
        self.symbols.pop()

    def _check_stmt(self, s) -> None:
        if isinstance(s, A.Compound):
            self._check_compound(s); return
        if isinstance(s, A.VarDecl):
            self._check_vardecl(s); return
        if isinstance(s, A.ExprStmt):
            if s.expr is not None:
                self._check_expr(s.expr)
            return
        if isinstance(s, A.If):
            ct = self._check_expr(s.cond)
            if ct is not None and not _is_numeric(ct) and not _is_pointer(ct):
                self._err(s.cond, "if-condition must be numeric or pointer")
            self._check_stmt(s.then)
            if s.else_ is not None:
                self._check_stmt(s.else_)
            return
        if isinstance(s, A.For):
            self.symbols.push("block")
            if isinstance(s.init, A.VarDecl):
                self._check_vardecl(s.init)
            elif isinstance(s.init, A.ExprStmt):
                if s.init.expr is not None:
                    self._check_expr(s.init.expr)
            elif isinstance(s.init, A.Compound):
                # for(int i=0,j=0;...) — walk inline without an extra push.
                for d in s.init.items:
                    self._check_stmt(d)
            if s.cond is not None:
                ct = self._check_expr(s.cond)
                if ct is not None and not _is_numeric(ct) and not _is_pointer(ct):
                    self._err(s.cond, "for-condition must be numeric or pointer")
            if s.step is not None:
                self._check_expr(s.step)
            self._check_stmt(s.body)
            self.collected_scopes.append(self._snapshot(self.symbols.current))
            self.symbols.pop()
            return
        if isinstance(s, (A.While, A.DoWhile)):
            ct = self._check_expr(s.cond)
            if ct is not None and not _is_numeric(ct) and not _is_pointer(ct):
                self._err(s.cond, "loop condition must be numeric or pointer")
            self._check_stmt(s.body)
            return
        if isinstance(s, A.Return):
            if s.value is not None:
                self._check_expr(s.value)
            return
        # Break, Continue — nothing to check.

    def _check_vardecl(self, d: A.VarDecl) -> None:
        # Coerce shared/array declarations into pointer-typed symbols so
        # subscripting works in the type checker.  In CUDA, `__shared__
        # float sdata[N]` and `extern __shared__ float sdata[]` decay to
        # `float*` for indexing.
        is_array = d.array_size is not None or d.extern_shared
        sym_type = d.type
        if is_array:
            sym_type = A.Type(
                base=d.type.base, is_pointer=True,
                is_const=d.type.is_const,
                is_restrict=d.type.is_restrict,
                addr_space=d.type.addr_space or "shared" if d.extern_shared else d.type.addr_space,
            )
        sym = Symbol(
            name=d.name,
            type=sym_type,
            kind="local",
            addr_space=sym_type.addr_space or "local",
            line=d.line,
            is_array=is_array,
            array_size=d.array_size,
            extern_shared=d.extern_shared,
        )
        try:
            self.symbols.insert(sym)
        except SymbolError as e:
            self._err(d, str(e))

        if d.array_size is not None:
            self._check_expr(d.array_size)
        if d.init is not None:
            it = self._check_expr(d.init)
            if it is not None and not _is_assignable(d.type, it):
                self._warn(d, f"narrowing initializer: {it!s} -> {d.type!s}")

    # ── Expression typing ─────────────────────────────────────────────────

    def _check_expr(self, e) -> Optional[A.Type]:
        if e is None:
            return None
        t = self._compute(e)
        self.types[id(e)] = t
        return t

    def _compute(self, e) -> Optional[A.Type]:
        if isinstance(e, A.IntLit):
            return A.Type(base="int")
        if isinstance(e, A.FloatLit):
            return A.Type(base="float")
        if isinstance(e, A.BoolLit):
            return A.Type(base="bool")

        if isinstance(e, A.Ident):
            sym = self.symbols.lookup(e.name)
            if sym is None:
                self._err(e, f"use of undeclared identifier {e.name!r}")
                return None
            sym.used = True
            return sym.type

        if isinstance(e, A.Member):
            obj_t = self._check_expr(e.obj)
            if obj_t is None:
                return None
            if _is_dim3(obj_t):
                if e.field not in ("x", "y", "z"):
                    self._err(e, f"dim3 has no member {e.field!r}")
                return A.Type(base="unsigned int")
            self._err(e, f"member access on non-struct type {obj_t!s}")
            return None

        if isinstance(e, A.Subscript):
            arr_t = self._check_expr(e.array)
            idx_t = self._check_expr(e.index)
            if idx_t is not None and not _is_integer(idx_t):
                self._err(e, f"array index has non-integer type {idx_t!s}")
            if arr_t is None:
                return None
            if not _is_pointer(arr_t):
                self._err(e, f"subscript on non-pointer type {arr_t!s}")
                return None
            return A.Type(
                base=arr_t.base, is_pointer=False,
                is_const=arr_t.is_const,
                addr_space=arr_t.addr_space,
            )

        if isinstance(e, A.UnaryOp):
            ot = self._check_expr(e.operand)
            if ot is None:
                return None
            if e.op in ("-", "+", "~"):
                if not _is_numeric(ot):
                    self._err(e, f"unary {e.op} on non-numeric {ot!s}")
                return ot
            if e.op == "!":
                return A.Type(base="int")
            if e.op in ("++", "--"):
                if not (_is_numeric(ot) or _is_pointer(ot)):
                    self._err(e, f"++/-- on non-numeric {ot!s}")
                return ot
            if e.op == "*":
                if not _is_pointer(ot):
                    self._err(e, f"cannot dereference non-pointer {ot!s}")
                    return None
                return A.Type(
                    base=ot.base, is_pointer=False,
                    is_const=ot.is_const, addr_space=ot.addr_space,
                )
            if e.op == "&":
                return A.Type(base=ot.base, is_pointer=True,
                              addr_space=ot.addr_space or "local")
            return ot

        if isinstance(e, A.BinOp):
            lt = self._check_expr(e.left)
            rt = self._check_expr(e.right)
            if lt is None or rt is None:
                return None
            return _binop_result(e.op, lt, rt)

        if isinstance(e, A.Cond):
            self._check_expr(e.cond)
            tt = self._check_expr(e.then)
            ft = self._check_expr(e.else_)
            if tt is not None and ft is not None:
                if tt.base != ft.base or tt.is_pointer != ft.is_pointer:
                    if not (_is_numeric(tt) and _is_numeric(ft)):
                        self._warn(e, f"ternary branches differ: {tt!s} vs {ft!s}")
            return tt

        if isinstance(e, A.Assign):
            lt = self._check_expr(e.target)
            rt = self._check_expr(e.value)
            if lt is None or rt is None:
                return lt
            if lt.is_const and not lt.is_pointer:
                self._err(e, "assignment to const variable")
            if not _is_assignable(lt, rt):
                self._warn(e, f"assignment narrows: {rt!s} -> {lt!s}")
            return lt

        if isinstance(e, A.Call):
            for a in e.args:
                self._check_expr(a)
            if isinstance(e.fn, A.Ident):
                sym = self.symbols.lookup(e.fn.name)
                if sym is None:
                    self._warn(e, f"call to undeclared function {e.fn.name!r}")
                    return A.Type(base="float")
                return sym.type
            return self._check_expr(e.fn)

        if isinstance(e, A.Cast):
            self._check_expr(e.operand)
            return e.type

        return None

    # ── Diagnostic helpers ────────────────────────────────────────────────

    def _err(self, node, msg: str):
        line = getattr(node, "line", 0)
        self.diagnostics.append(Diagnostic("error", line, msg))

    def _warn(self, node, msg: str):
        line = getattr(node, "line", 0)
        self.diagnostics.append(Diagnostic("warning", line, msg))


# ── Convenience driver (kept for callers that want both objects) ───────────

def check_function(fn: A.FunctionDef) -> tuple[SymbolTable, TypeChecker]:
    """Run the combined symbol-table + type-check pass for one function."""
    st = SymbolTable()
    tc = TypeChecker(st)
    tc.check_function(fn)
    return st, tc

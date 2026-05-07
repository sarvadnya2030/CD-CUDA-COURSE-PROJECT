"""
ast_nodes.py — AST class hierarchy produced by the PLY parser.

Each node carries (line, col) source coordinates for diagnostic output.
The hierarchy is intentionally close to a textbook abstract syntax tree:

    TranslationUnit
        FunctionDef           __global__ / __device__ / __host__
            Param
            Compound
                VarDecl       includes __shared__ declarations
                ExprStmt
                If / For / While / DoWhile / Return / Break / Continue
                Compound      nested
            (statement bodies recurse)
        Expr nodes:
            BinOp, UnaryOp, Assign, Cond (?:), Subscript, Member,
            Call, Ident, IntLit, FloatLit, BoolLit

Nodes are dataclasses for easy printing and visitor traversal.
A `pretty(node)` helper produces an indented dump.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


# ── Type system ─────────────────────────────────────────────────────────────

# Address spaces relevant to CUDA semantics:
#   "global"   default device-pointer (kernel parameter)
#   "shared"   __shared__
#   "constant" __constant__
#   "local"    register / stack
#   "host"     host-side
ADDR_SPACES = ("global", "shared", "constant", "local", "host")

# Function qualifiers
FN_QUALIFIERS = ("__global__", "__device__", "__host__")


@dataclass
class Type:
    """Represents a (possibly pointer/qualified) type."""
    base: str                       # int, float, void, bool, unsigned int
    is_pointer: bool = False
    is_const: bool = False
    is_restrict: bool = False
    addr_space: Optional[str] = None  # only set on declarations

    def __str__(self) -> str:
        parts = []
        if self.addr_space and self.addr_space != "local":
            parts.append(f"__{self.addr_space}__")
        if self.is_const:
            parts.append("const")
        parts.append(self.base)
        if self.is_pointer:
            parts.append("*")
            if self.is_restrict:
                parts.append("__restrict__")
        return " ".join(parts)


# ── Base node ──────────────────────────────────────────────────────────────

@dataclass
class Node:
    line: int = 0
    col: int = 0


# ── Top level ──────────────────────────────────────────────────────────────

@dataclass
class TranslationUnit(Node):
    items: List["FunctionDef"] = field(default_factory=list)


@dataclass
class FunctionDef(Node):
    qualifiers: List[str] = field(default_factory=list)   # __global__, __device__, __host__
    return_type: Optional[Type] = None
    name: str = ""
    params: List["Param"] = field(default_factory=list)
    body: Optional["Compound"] = None


@dataclass
class Param(Node):
    type: Optional[Type] = None
    name: str = ""


# ── Statements ─────────────────────────────────────────────────────────────

@dataclass
class Compound(Node):
    items: List["Stmt"] = field(default_factory=list)


@dataclass
class VarDecl(Node):
    type: Optional[Type] = None
    name: str = ""
    init: Optional["Expr"] = None
    array_size: Optional["Expr"] = None    # for `float sdata[N]` or extern[]
    extern_shared: bool = False             # for `extern __shared__ float foo[]`


@dataclass
class ExprStmt(Node):
    expr: Optional["Expr"] = None


@dataclass
class If(Node):
    cond: Optional["Expr"] = None
    then: Optional["Stmt"] = None
    else_: Optional["Stmt"] = None


@dataclass
class For(Node):
    init: Optional[Union["Stmt", "Expr"]] = None
    cond: Optional["Expr"] = None
    step: Optional["Expr"] = None
    body: Optional["Stmt"] = None


@dataclass
class While(Node):
    cond: Optional["Expr"] = None
    body: Optional["Stmt"] = None


@dataclass
class DoWhile(Node):
    body: Optional["Stmt"] = None
    cond: Optional["Expr"] = None


@dataclass
class Return(Node):
    value: Optional["Expr"] = None


@dataclass
class Break(Node):
    pass


@dataclass
class Continue(Node):
    pass


# ── Expressions ────────────────────────────────────────────────────────────

@dataclass
class BinOp(Node):
    op: str = ""
    left: Optional["Expr"] = None
    right: Optional["Expr"] = None


@dataclass
class UnaryOp(Node):
    op: str = ""
    operand: Optional["Expr"] = None
    postfix: bool = False     # True for x++ / x--, False for ++x / --x / -x / etc.


@dataclass
class Assign(Node):
    op: str = "="          # =, +=, -=, *=, /=, %=, <<=, >>=, &=, ^=, |=
    target: Optional["Expr"] = None
    value: Optional["Expr"] = None


@dataclass
class Cond(Node):           # ternary ?:
    cond: Optional["Expr"] = None
    then: Optional["Expr"] = None
    else_: Optional["Expr"] = None


@dataclass
class Subscript(Node):       # arr[idx]
    array: Optional["Expr"] = None
    index: Optional["Expr"] = None


@dataclass
class Member(Node):           # obj.field   (used for threadIdx.x etc.)
    obj: Optional["Expr"] = None
    field: str = ""


@dataclass
class Call(Node):
    fn: Optional["Expr"] = None
    args: List["Expr"] = field(default_factory=list)


@dataclass
class Cast(Node):
    type: Optional[Type] = None
    operand: Optional["Expr"] = None


@dataclass
class Ident(Node):
    name: str = ""


@dataclass
class IntLit(Node):
    value: int = 0


@dataclass
class FloatLit(Node):
    value: float = 0.0


@dataclass
class BoolLit(Node):
    value: bool = False


# Type aliases for grammar code clarity.
Stmt = Union[Compound, VarDecl, ExprStmt, If, For, While, DoWhile,
             Return, Break, Continue]
Expr = Union[BinOp, UnaryOp, Assign, Cond, Subscript, Member,
             Call, Cast, Ident, IntLit, FloatLit, BoolLit]


# ── Pretty printer ─────────────────────────────────────────────────────────

def pretty(node, indent: int = 0) -> str:
    """Return a multi-line indented string representation of an AST."""
    pad = "  " * indent
    if node is None:
        return f"{pad}<None>"
    if isinstance(node, list):
        if not node:
            return f"{pad}[]"
        return "\n".join(pretty(n, indent) for n in node)

    cls = type(node).__name__

    # Leaf-ish nodes — print on one line.
    if isinstance(node, Ident):
        return f"{pad}Ident({node.name!r}) @ {node.line}:{node.col}"
    if isinstance(node, IntLit):
        return f"{pad}IntLit({node.value})"
    if isinstance(node, FloatLit):
        return f"{pad}FloatLit({node.value})"
    if isinstance(node, BoolLit):
        return f"{pad}BoolLit({node.value})"
    if isinstance(node, Type):
        return f"{pad}Type({node})"

    # Compound nodes — print children.
    lines = [f"{pad}{cls}"]
    if isinstance(node, FunctionDef):
        lines.append(f"{pad}  qualifiers={node.qualifiers}")
        lines.append(f"{pad}  return_type={node.return_type}")
        lines.append(f"{pad}  name={node.name!r}")
        lines.append(f"{pad}  params:")
        for p in node.params:
            lines.append(f"{pad}    {p.type} {p.name}")
        lines.append(f"{pad}  body:")
        lines.append(pretty(node.body, indent + 2))
        return "\n".join(lines)

    if isinstance(node, Param):
        return f"{pad}Param({node.type} {node.name})"

    if isinstance(node, Compound):
        for s in node.items:
            lines.append(pretty(s, indent + 1))
        return "\n".join(lines)

    if isinstance(node, VarDecl):
        size = ""
        if node.array_size is not None:
            size = f"[size={pretty(node.array_size).strip()}]"
        elif node.extern_shared:
            size = "[extern]"
        lines.append(f"{pad}  type={node.type}")
        lines.append(f"{pad}  name={node.name!r}{size}")
        if node.init is not None:
            lines.append(f"{pad}  init:")
            lines.append(pretty(node.init, indent + 2))
        return "\n".join(lines)

    if isinstance(node, ExprStmt):
        lines.append(pretty(node.expr, indent + 1))
        return "\n".join(lines)

    if isinstance(node, If):
        lines.append(f"{pad}  cond:")
        lines.append(pretty(node.cond, indent + 2))
        lines.append(f"{pad}  then:")
        lines.append(pretty(node.then, indent + 2))
        if node.else_ is not None:
            lines.append(f"{pad}  else:")
            lines.append(pretty(node.else_, indent + 2))
        return "\n".join(lines)

    if isinstance(node, For):
        lines.append(f"{pad}  init:")
        lines.append(pretty(node.init, indent + 2))
        lines.append(f"{pad}  cond:")
        lines.append(pretty(node.cond, indent + 2))
        lines.append(f"{pad}  step:")
        lines.append(pretty(node.step, indent + 2))
        lines.append(f"{pad}  body:")
        lines.append(pretty(node.body, indent + 2))
        return "\n".join(lines)

    if isinstance(node, (While, DoWhile)):
        lines.append(f"{pad}  cond:")
        lines.append(pretty(node.cond, indent + 2))
        lines.append(f"{pad}  body:")
        lines.append(pretty(node.body, indent + 2))
        return "\n".join(lines)

    if isinstance(node, Return):
        if node.value is not None:
            lines.append(pretty(node.value, indent + 1))
        return "\n".join(lines)

    if isinstance(node, BinOp):
        lines.append(f"{pad}  op={node.op!r}")
        lines.append(pretty(node.left, indent + 1))
        lines.append(pretty(node.right, indent + 1))
        return "\n".join(lines)

    if isinstance(node, UnaryOp):
        kind = "postfix" if node.postfix else "prefix"
        lines.append(f"{pad}  op={node.op!r} ({kind})")
        lines.append(pretty(node.operand, indent + 1))
        return "\n".join(lines)

    if isinstance(node, Assign):
        lines.append(f"{pad}  op={node.op!r}")
        lines.append(f"{pad}  target:")
        lines.append(pretty(node.target, indent + 2))
        lines.append(f"{pad}  value:")
        lines.append(pretty(node.value, indent + 2))
        return "\n".join(lines)

    if isinstance(node, Cond):
        lines.append(f"{pad}  cond:")
        lines.append(pretty(node.cond, indent + 2))
        lines.append(f"{pad}  then:")
        lines.append(pretty(node.then, indent + 2))
        lines.append(f"{pad}  else:")
        lines.append(pretty(node.else_, indent + 2))
        return "\n".join(lines)

    if isinstance(node, Subscript):
        lines.append(f"{pad}  array:")
        lines.append(pretty(node.array, indent + 2))
        lines.append(f"{pad}  index:")
        lines.append(pretty(node.index, indent + 2))
        return "\n".join(lines)

    if isinstance(node, Member):
        lines.append(f"{pad}  field={node.field!r}")
        lines.append(pretty(node.obj, indent + 1))
        return "\n".join(lines)

    if isinstance(node, Call):
        lines.append(f"{pad}  fn:")
        lines.append(pretty(node.fn, indent + 2))
        lines.append(f"{pad}  args:")
        for a in node.args:
            lines.append(pretty(a, indent + 2))
        return "\n".join(lines)

    if isinstance(node, Cast):
        lines.append(f"{pad}  type={node.type}")
        lines.append(pretty(node.operand, indent + 1))
        return "\n".join(lines)

    if isinstance(node, TranslationUnit):
        for it in node.items:
            lines.append(pretty(it, indent + 1))
        return "\n".join(lines)

    return f"{pad}{cls}{node!r}"

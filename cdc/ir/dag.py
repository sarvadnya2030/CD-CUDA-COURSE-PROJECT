"""
dag.py — DAG representation of a basic block (Aho/Ullman §8.5).

A *DAG-of-basic-block* is a directed acyclic graph whose interior nodes
represent operations and whose leaves represent initial values (variables
or constants).  Every variable defined in the block is attached to the
node holding its current value; common subexpressions naturally collapse
because we look up an existing node before creating a new one.

Why we care
-----------
The DAG is the canonical structure for:

  * **Common-subexpression elimination** (CSE) — if `a + b` already has a
    node, the next occurrence of `a + b` reuses it.
  * **Dead-code detection** — interior nodes with no attached identifier
    are dead.
  * **Generating efficient code from DAGs** (Course Unit V) — the order
    we walk the DAG influences register pressure.

What we do NOT do here
----------------------
This module *constructs* the DAG and prints it.  Actual elimination /
code emission lives in `cdc/opt/` (Phase 3).

Maps to syllabus
----------------
* Course Unit V — DAG representation of basic blocks, generating code from DAGs
* Tutorial 13   — Examples of DAG representation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Optional, Tuple

from .basic_block import BasicBlock
from .tac import Quad


# ── DAG node ───────────────────────────────────────────────────────────────

@dataclass
class DagNode:
    """One node in a basic-block DAG.

    Leaves have op='leaf' and value set to a variable name or literal.
    Interior nodes have op set to a TAC opcode and one or two children.
    `attached` lists every variable name whose current value the node
    represents — the *first* attached name is the node's canonical
    identifier when generating code.
    """
    id: int
    op: str
    value: Optional[str] = None      # for leaves
    left: Optional["DagNode"] = None
    right: Optional["DagNode"] = None
    extra: Optional[str] = None      # e.g. field name for member ops
    attached: List[str] = field(default_factory=list)
    line: int = 0

    def is_leaf(self) -> bool:
        return self.op == "leaf"

    def label(self) -> str:
        if self.is_leaf():
            return f"{self.value}"
        if self.op in ("[]=", "=[]", "=*", "*=", "member", "cast"):
            return self.op
        return self.op

    def __str__(self) -> str:
        names = "{" + ",".join(self.attached) + "}" if self.attached else ""
        if self.is_leaf():
            return f"n{self.id}: leaf({self.value}) {names}"
        l = f"n{self.left.id}" if self.left else "_"
        r = f"n{self.right.id}" if self.right else "_"
        extra = f" extra={self.extra!r}" if self.extra is not None else ""
        return f"n{self.id}: {self.label()}({l}, {r}){extra} {names}"


# ── DAG builder ────────────────────────────────────────────────────────────

class DagBuilder:
    """
    Build a DAG from a single basic block.

    The classic algorithm (Aho/Ullman Algorithm 8.5):

        for each TAC instruction `x = y op z`:
            n_y = lookup_or_make_leaf(y)
            n_z = lookup_or_make_leaf(z)
            existing = find_interior(op, n_y, n_z)
            if existing:
                node = existing
            else:
                node = new_interior(op, n_y, n_z)
            detach `x` from any other node and attach to `node`.

    We treat copy (`=`), unary, member, cast, and load/store the same
    way, with the obvious shape adjustments.
    """

    def __init__(self):
        self._nodes: List[DagNode] = []
        self._ids = count()
        self._var_node: Dict[str, DagNode] = {}                 # name → current node
        self._interior_cache: Dict[Tuple, DagNode] = {}         # (op, lid, rid, extra) → node

    # ── Public API ─────────────────────────────────────────────────────

    def build(self, block: BasicBlock) -> List[DagNode]:
        for q in block.quads:
            self._consume(q)
        return self._nodes

    # ── Lookups ────────────────────────────────────────────────────────

    def _leaf(self, name: str) -> DagNode:
        n = self._var_node.get(name)
        if n is not None:
            return n
        n = DagNode(id=next(self._ids), op="leaf", value=name, attached=[])
        self._nodes.append(n)
        self._var_node[name] = n
        return n

    def _interior(self, op: str, l: DagNode, r: Optional[DagNode],
                  extra: Optional[str] = None, line: int = 0) -> DagNode:
        key = (op, l.id, r.id if r else None, extra)
        cached = self._interior_cache.get(key)
        if cached is not None:
            return cached
        n = DagNode(id=next(self._ids), op=op, left=l, right=r,
                    extra=extra, line=line, attached=[])
        self._nodes.append(n)
        self._interior_cache[key] = n
        return n

    def _attach(self, name: str, node: DagNode) -> None:
        prev = self._var_node.get(name)
        if prev is not None and prev is not node:
            if name in prev.attached:
                prev.attached.remove(name)
        if name not in node.attached:
            node.attached.append(name)
        self._var_node[name] = node

    # ── Per-quadruple semantics ────────────────────────────────────────

    def _consume(self, q: Quad) -> None:
        if q.op == "label" or q.is_branch() or q.op == "return" or q.op == "param":
            return                              # control / ABI: ignored

        if q.op == "=":                         # copy / move
            src_name = str(q.arg1)
            src_node = self._leaf(src_name)
            self._attach(str(q.result), src_node)
            return

        if q.op in ("uminus", "uplus", "unot", "bnot"):
            l = self._leaf(str(q.arg1))
            n = self._interior(q.op, l, None, line=q.line)
            self._attach(str(q.result), n)
            return

        if q.op == "[]=":                       # load: r = a[b]
            l = self._leaf(str(q.arg1))
            r = self._leaf(str(q.arg2))
            n = self._interior("[]=", l, r, line=q.line)
            self._attach(str(q.result), n)
            return

        if q.op == "=[]":                       # store: a[b] = r  (kills nothing in-block)
            l = self._leaf(str(q.arg1))
            r = self._leaf(str(q.arg2))
            v = self._leaf(str(q.result))
            n = self._interior("=[]", l, r, extra=str(q.result), line=q.line)
            n.attached.append(f"*{q.arg1}[{q.arg2}]")
            return

        if q.op == "=*":                        # ptr load
            l = self._leaf(str(q.arg1))
            n = self._interior("=*", l, None, line=q.line)
            self._attach(str(q.result), n)
            return

        if q.op == "*=":                        # ptr store
            l = self._leaf(str(q.arg1))
            v = self._leaf(str(q.result))
            n = self._interior("*=", l, None, extra=str(q.result), line=q.line)
            n.attached.append(f"*{q.arg1}")
            return

        if q.op == "member":
            l = self._leaf(str(q.arg1))
            n = self._interior("member", l, None, extra=str(q.arg2), line=q.line)
            self._attach(str(q.result), n)
            return

        if q.op == "cast":
            l = self._leaf(str(q.arg1))
            n = self._interior("cast", l, None, extra=str(q.arg2), line=q.line)
            self._attach(str(q.result), n)
            return

        if q.op == "call":
            # Calls have side effects; do not CSE.  Allocate a unique node.
            l = self._leaf(str(q.arg1)) if q.arg1 is not None else None
            n = DagNode(id=next(self._ids), op="call", left=l, right=None,
                        extra=str(q.arg2), line=q.line, attached=[])
            self._nodes.append(n)
            self._attach(str(q.result), n)
            return

        # Fall-through: binary operator (+, -, *, /, %, ==, !=, <, >, …)
        if q.arg1 is not None and q.arg2 is not None and q.result is not None:
            l = self._leaf(str(q.arg1))
            r = self._leaf(str(q.arg2))
            n = self._interior(q.op, l, r, line=q.line)
            self._attach(str(q.result), n)
            return


# ── Public façade & rendering ──────────────────────────────────────────────

def build_dag(block: BasicBlock) -> List[DagNode]:
    """Build a DAG for one basic block; return the list of DAG nodes."""
    return DagBuilder().build(block)


def format_dag(nodes: List[DagNode]) -> str:
    if not nodes:
        return "(empty DAG)"
    lines = []
    for n in nodes:
        lines.append(f"  {n}")
    return "\n".join(lines)


def to_dot(nodes: List[DagNode], block_label: str = "BB") -> str:
    """Render a DAG as a Graphviz DOT graph for documentation use."""
    out = [f'digraph "{block_label}" {{', '  node [shape=ellipse, fontname="monospace"];']
    for n in nodes:
        if n.is_leaf():
            shape = "box"
            txt = n.value
        else:
            shape = "ellipse"
            txt = n.label()
            if n.extra:
                txt += f"\\n[{n.extra}]"
        attached = ("\\n{" + ",".join(n.attached) + "}") if n.attached else ""
        out.append(f'  n{n.id} [shape={shape}, label="{txt}{attached}"];')
        if n.left is not None:
            out.append(f'  n{n.id} -> n{n.left.id};')
        if n.right is not None:
            out.append(f'  n{n.id} -> n{n.right.id};')
    out.append("}")
    return "\n".join(out)

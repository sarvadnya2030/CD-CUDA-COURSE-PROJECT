"""
cfg.py — Control-Flow Graph from a list of basic blocks.

Algorithm
---------
For every block `B`:

  * Inspect its terminator quadruple (the last one).
  * If terminator is `goto L`         → succ = block whose label is L.
  * If terminator is `ifgoto L` /
                       `iffalse L`    → succ = {block(L), fall-through B+1}.
  * If terminator is `return`         → no successors (exit edge).
  * Otherwise (no terminator)         → fall-through to B+1.

Predecessors are the inverse map.  A virtual ENTRY node points to the
first block; a virtual EXIT node collects edges from `return`-terminated
blocks.

Dominator tree
--------------
We compute dominators with the simple iterative dataflow formulation
(Cooper, Harvey, Kennedy 2001 — *A Simple, Fast Dominance Algorithm*).
The result is exposed as `cfg.idom` (immediate-dominator map) and
`cfg.dom_tree` (parent → children).  Useful inputs to the planned
loop-invariant code-motion pass.

Maps to syllabus
----------------
* Course Unit V — Flow Graphs
* Course Unit VI — Global Data Flow Analysis foundations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .basic_block import BasicBlock


ENTRY_ID = -1   # virtual entry node id
EXIT_ID  = -2   # virtual exit node id


@dataclass
class ControlFlowGraph:
    blocks: List[BasicBlock] = field(default_factory=list)
    succ:   Dict[int, List[int]] = field(default_factory=dict)
    pred:   Dict[int, List[int]] = field(default_factory=dict)
    idom:   Dict[int, Optional[int]] = field(default_factory=dict)
    dom_tree: Dict[int, List[int]]   = field(default_factory=dict)

    # Quick lookups -----------------------------------------------------------

    def block(self, bid: int) -> Optional[BasicBlock]:
        if bid in (ENTRY_ID, EXIT_ID):
            return None
        if 0 <= bid < len(self.blocks):
            return self.blocks[bid]
        return None

    def label_of(self, bid: int) -> str:
        if bid == ENTRY_ID:
            return "ENTRY"
        if bid == EXIT_ID:
            return "EXIT"
        bb = self.block(bid)
        return f"BB{bid}" + (f" ({bb.label})" if bb and bb.label else "")

    # ── Pretty-printers ─────────────────────────────────────────────────

    def format_edges(self) -> str:
        lines = ["edges:"]
        for src in sorted(self.succ):
            for dst in self.succ[src]:
                lines.append(f"  {self.label_of(src):>10}  ->  {self.label_of(dst)}")
        return "\n".join(lines)

    def format_dominators(self) -> str:
        lines = ["dominator tree:"]
        for parent in sorted(self.dom_tree):
            kids = self.dom_tree[parent]
            if not kids:
                continue
            lines.append(f"  {self.label_of(parent)} -> {[self.label_of(c) for c in kids]}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the CFG (without dom tree)."""
        out = ["digraph CFG {", '  node [shape=box, fontname="monospace"];']
        for bb in self.blocks:
            label = (f"BB{bb.id}" +
                     (f" ({bb.label})" if bb.label else "") +
                     f"\\l{len(bb.quads)} quads\\l")
            out.append(f'  bb{bb.id} [label="{label}"];')
        for src, dsts in self.succ.items():
            for dst in dsts:
                if src == ENTRY_ID:
                    out.append(f'  ENTRY -> bb{dst};')
                elif dst == EXIT_ID:
                    out.append(f'  bb{src} -> EXIT;')
                else:
                    out.append(f'  bb{src} -> bb{dst};')
        out.append("}")
        return "\n".join(out)


# ── CFG construction ───────────────────────────────────────────────────────

def build_cfg(blocks: List[BasicBlock]) -> ControlFlowGraph:
    """Build a `ControlFlowGraph` from a flat list of basic blocks."""
    cfg = ControlFlowGraph(blocks=blocks)
    cfg.succ = {ENTRY_ID: [], EXIT_ID: []}
    cfg.pred = {ENTRY_ID: [], EXIT_ID: []}
    for bb in blocks:
        cfg.succ[bb.id] = []
        cfg.pred[bb.id] = []

    if not blocks:
        return cfg

    # ENTRY → first block.
    cfg.succ[ENTRY_ID].append(blocks[0].id)
    cfg.pred[blocks[0].id].append(ENTRY_ID)

    # Map block-label → block id for branch resolution.
    label_to_id: Dict[str, int] = {bb.label: bb.id for bb in blocks if bb.label}

    for i, bb in enumerate(blocks):
        term = bb.terminator()
        next_id = blocks[i + 1].id if i + 1 < len(blocks) else None

        if term is None:
            # Fall through to next block (or EXIT).
            if next_id is not None:
                _add_edge(cfg, bb.id, next_id)
            else:
                _add_edge(cfg, bb.id, EXIT_ID)
            continue

        if term.op == "goto":
            tgt = label_to_id.get(str(term.result))
            if tgt is not None:
                _add_edge(cfg, bb.id, tgt)
            continue

        if term.op in ("ifgoto", "iffalse"):
            tgt = label_to_id.get(str(term.result))
            if tgt is not None:
                _add_edge(cfg, bb.id, tgt)
            if next_id is not None:
                _add_edge(cfg, bb.id, next_id)
            else:
                _add_edge(cfg, bb.id, EXIT_ID)
            continue

        if term.op == "return":
            _add_edge(cfg, bb.id, EXIT_ID)
            continue

    _compute_dominators(cfg)
    return cfg


def _add_edge(cfg: ControlFlowGraph, src: int, dst: int) -> None:
    if dst not in cfg.succ[src]:
        cfg.succ[src].append(dst)
    if src not in cfg.pred[dst]:
        cfg.pred[dst].append(src)


# ── Dominator computation (iterative) ──────────────────────────────────────

def _compute_dominators(cfg: ControlFlowGraph) -> None:
    """
    Iterative dominator computation.

    Initialise dom[entry] = {entry} and dom[v] = all_nodes for v != entry.
    Repeatedly set dom[v] = {v} ∪ (∩ over preds p: dom[p]) until fixed
    point.  Then idom[v] = the unique dominator in dom[v]\\{v} that is
    dominated by every other element.
    """
    nodes = [ENTRY_ID] + [bb.id for bb in cfg.blocks]
    if EXIT_ID in cfg.pred:
        nodes.append(EXIT_ID)
    all_set = set(nodes)
    dom: Dict[int, Set[int]] = {n: set(all_set) for n in nodes}
    dom[ENTRY_ID] = {ENTRY_ID}

    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n == ENTRY_ID:
                continue
            preds = cfg.pred.get(n, [])
            if not preds:
                new = {n}
            else:
                new = set(all_set)
                for p in preds:
                    new &= dom[p]
                new.add(n)
            if new != dom[n]:
                dom[n] = new
                changed = True

    # idom[v] = the strict dominator of v that is dominated by every other
    # strict dominator (i.e. the closest ancestor in the dominator tree).
    idom: Dict[int, Optional[int]] = {n: None for n in nodes}
    for n in nodes:
        if n == ENTRY_ID:
            continue
        strict = dom[n] - {n}
        best: Optional[int] = None
        for d in strict:
            # d is the immediate dominator if every other strict dominator
            # also dominates d (i.e. dom[d] ⊇ strict).
            if all((other in dom[d]) for other in strict):
                if best is None or len(dom[d]) > len(dom[best]):
                    best = d
        idom[n] = best

    cfg.idom = idom
    tree: Dict[int, List[int]] = {n: [] for n in nodes}
    for n, p in idom.items():
        if p is not None:
            tree[p].append(n)
    cfg.dom_tree = tree

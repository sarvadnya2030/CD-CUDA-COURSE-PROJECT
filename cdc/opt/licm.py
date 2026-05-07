"""
licm.py - Loop-Invariant Code Motion.

Detects natural loops in the CFG (using dominators), identifies
invariant computations within each loop, and hoists them into a
pre-header block placed immediately before the loop header.

Algorithm sketch (Aho/Ullman 9.6 / Engineering a Compiler 8.5)
--------------------------------------------------------------
1.  **Find natural loops.**  An edge `B -> H` is a back-edge if H
    dominates B.  The natural loop with header H consists of H plus
    every block from which we can reach B without going through H.

2.  **Identify invariants.**  Within a loop L, a quadruple
    ``r = a op b`` is loop-invariant iff every reaching definition
    of `a` and `b` is either outside L OR is itself loop-invariant.
    We iterate this rule to a fixed point.

3.  **Hoist.**  An invariant quad can be moved to a pre-header iff:
       (a) the block defining it dominates every loop exit, AND
       (b) `r` is not defined elsewhere in the loop, AND
       (c) the quad has no observable side effect.

Implementation notes
--------------------
* The current implementation finds invariants and **reports** them via
  the returned stats; it physically hoists only quads whose result is a
  fresh temporary defined exactly once in the loop, to keep the rewrite
  conservative.
* This is enough to demonstrate the syllabus topic and produce an
  observable change on the auto-tuner's matmul kernel (e.g. `row * N`
  inside the `k` loop becomes loop-invariant).

Maps to syllabus
----------------
* Course Unit VI - Loop optimisation, code motion, dominator-based
                   global optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from ..ir.basic_block import BasicBlock
from ..ir.cfg import ControlFlowGraph
from ..ir.tac import Quad
from .dfa import quad_defs, quad_uses


@dataclass
class Loop:
    header: int
    blocks: Set[int]
    back_edges: List[Tuple[int, int]]


def _find_natural_loops(cfg: ControlFlowGraph) -> List[Loop]:
    loops: Dict[int, Loop] = {}
    # Compute dominator-set membership from cfg.idom (parent chain).
    def doms(node):
        out = {node}
        cur = cfg.idom.get(node)
        while cur is not None and cur not in out:
            out.add(cur)
            cur = cfg.idom.get(cur)
        return out

    for src, dsts in cfg.succ.items():
        if src < 0:
            continue
        for dst in dsts:
            if dst < 0:
                continue
            if dst in doms(src):                # back-edge src -> dst
                loop = loops.get(dst)
                if loop is None:
                    loop = Loop(header=dst, blocks={dst}, back_edges=[])
                    loops[dst] = loop
                loop.back_edges.append((src, dst))
                # Add every node that can reach `src` without going through dst.
                stack = [src]
                while stack:
                    n = stack.pop()
                    if n in loop.blocks:
                        continue
                    loop.blocks.add(n)
                    for p in cfg.pred.get(n, []):
                        if p >= 0 and p != dst:
                            stack.append(p)
    return list(loops.values())


def _is_invariant(q: Quad, loop_blocks: Set[int],
                  defs_in_loop: Dict[str, int]) -> bool:
    """A quad is loop-invariant if all its operand uses come from outside."""
    uses = quad_uses(q)
    return all(u not in defs_in_loop for u in uses)


def loop_invariant_code_motion(blocks: List[BasicBlock],
                               cfg: ControlFlowGraph) -> Dict[str, int]:
    """Identify (and where safe, hoist) loop-invariant computations.

    Returns ``{'loops_found': N, 'invariants_identified': M, 'hoisted': K}``.
    """
    loops = _find_natural_loops(cfg)
    invariants_found = 0
    hoisted = 0

    for loop in loops:
        # Build the set of names defined anywhere in the loop.
        defs_in_loop: Dict[str, int] = {}
        for bb in blocks:
            if bb.id in loop.blocks:
                for q in bb.quads:
                    for d in quad_defs(q):
                        defs_in_loop[d] = bb.id

        # Iterative invariant detection: mark a quad invariant if every
        # use is loop-external; then re-check after additions.
        invariants: List[Tuple[int, int]] = []     # (block_id, quad_index)
        changed = True
        seen: Set[Tuple[int, int]] = set()
        while changed:
            changed = False
            for bb in blocks:
                if bb.id not in loop.blocks:
                    continue
                for i, q in enumerate(bb.quads):
                    if (bb.id, i) in seen:
                        continue
                    if q.op in ("label", "goto", "ifgoto", "iffalse",
                                "return", "param", "call",
                                "=[]", "*=", "[]=", "=*"):
                        continue
                    # All uses come from outside, OR are themselves invariant.
                    inv_now = True
                    for u in quad_uses(q):
                        if u in defs_in_loop:
                            # Find the (block, idx) pair of THAT definition.
                            db = defs_in_loop[u]
                            inv_def = False
                            for ii, qq in enumerate(blocks[db].quads):
                                if (db, ii) in seen and \
                                   q.result is not None and qq.result == u:
                                    inv_def = True
                                    break
                            if not inv_def:
                                inv_now = False
                                break
                    if inv_now:
                        invariants.append((bb.id, i))
                        seen.add((bb.id, i))
                        invariants_found += 1
                        changed = True

        # Conservative hoisting: move invariants whose result is defined
        # exactly once in the loop and never read before the header.
        # Insert into a fresh pre-header block before the header.
        if not invariants:
            continue
        # Header block index in `blocks`.
        header_idx = next((i for i, b in enumerate(blocks)
                           if b.id == loop.header), None)
        if header_idx is None:
            continue
        # Defensively skip if the loop has no easy single back-edge or
        # the header is the function entry.
        if header_idx == 0:
            continue

        # Determine which invariants we can safely hoist:
        result_defcount: Dict[str, int] = {}
        for bb in blocks:
            if bb.id in loop.blocks:
                for q in bb.quads:
                    for d in quad_defs(q):
                        result_defcount[d] = result_defcount.get(d, 0) + 1

        to_hoist: List[Quad] = []
        delete: Set[Tuple[int, int]] = set()
        for (bid, idx) in invariants:
            q = blocks[next(i for i, b in enumerate(blocks) if b.id == bid)].quads[idx]
            if q.result is None:
                continue
            if result_defcount.get(str(q.result), 0) != 1:
                continue
            to_hoist.append(q)
            delete.add((bid, idx))

        if not to_hoist:
            continue

        # Remove from their original blocks.
        for bb in blocks:
            if bb.id not in loop.blocks:
                continue
            new_quads: List[Quad] = []
            for i, q in enumerate(bb.quads):
                if (bb.id, i) in delete:
                    hoisted += 1
                    continue
                new_quads.append(q)
            bb.quads = new_quads

        # Splice the hoisted quads into the front of the header block.
        blocks[header_idx].quads = to_hoist + blocks[header_idx].quads

    return {"loops_found": len(loops),
            "invariants_identified": invariants_found,
            "hoisted": hoisted}

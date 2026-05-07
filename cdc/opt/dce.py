"""
dce.py - Dead Code Elimination via Live-Variable Analysis.

A quadruple ``r = ...`` is dead at point P if `r` is not live on any
path leaving P **and** the quad has no observable side effect.

Side-effect-bearing opcodes that we never delete:

    =[]   *=     - memory stores
    call          - function call (may have side effects)
    return        - control transfer
    label         - block boundary
    goto, ifgoto, iffalse - control flow
    param         - argument staging

The pass walks each basic block backwards using the OUT[B] live set
from `LiveVariables`.  It maintains a "currently live" set, and
deletes any pure quad whose `result` is not in that set; otherwise the
quad's USE names join the live set.

Maps to syllabus
----------------
* Course Unit VI - Live range analysis, optimisation of basic blocks,
                   global data-flow analysis applications.
"""

from __future__ import annotations

from typing import Dict, List

from ..ir.basic_block import BasicBlock
from ..ir.cfg import ControlFlowGraph
from .dfa import LiveVariables, solve, quad_defs, quad_uses


_PURE_OPCODES = {
    "=", "uminus", "uplus", "unot", "bnot",
    "+", "-", "*", "/", "%",
    "&", "|", "^", "<<", ">>",
    "==", "!=", "<", ">", "<=", ">=",
    "[]=", "=*", "member", "cast",
}


def dead_code_elimination(blocks: List[BasicBlock],
                          cfg: ControlFlowGraph) -> Dict[str, int]:
    """Remove dead pure quadruples driven by live-variable analysis.

    Mutates `blocks` in place.  Returns ``{'removed': N}``.
    """
    lv = LiveVariables()
    in_sets, out_sets = solve(cfg, lv)

    removed = 0
    for bb in blocks:
        live = set(out_sets.get(bb.id, set()))
        new_rev: List = []
        for q in reversed(bb.quads):
            if q.op in _PURE_OPCODES and q.result is not None and \
               str(q.result) not in live and not str(q.result).startswith("*"):
                removed += 1
                continue                      # delete
            # Keep this quad.
            for d in quad_defs(q):
                live.discard(d)
            for u in quad_uses(q):
                live.add(u)
            new_rev.append(q)
        bb.quads = list(reversed(new_rev))

    return {"removed": removed}

"""
cse.py - Common Subexpression Elimination.

Two implementations are provided:

  * **Local CSE** (per basic block) using the DAG built by
    `cdc.ir.dag.DagBuilder`.  When a binary operation has the same
    operands as an earlier operation in the block, the earlier
    temporary is reused and the duplicate quad is replaced with a copy.
  * **Global CSE** stub (using the `AvailableExpressions` analysis from
    `cdc.opt.dfa`) - left as a hook so the code-generation pass can
    consume the analysis output without re-running it.

Maps to syllabus
----------------
* Course Unit V  - DAG representation, generating code from DAGs.
* Course Unit VI - Common subexpression elimination, optimisation of
                   basic blocks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..ir.basic_block import BasicBlock
from ..ir.tac import Quad


def _binary_key(q: Quad):
    """Return a normalised key for a CSE-able binary quad, else None."""
    if q.op in ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>",
                "==", "!=", "<", ">", "<=", ">="):
        a, b = str(q.arg1), str(q.arg2)
        if q.op in ("+", "*", "&", "|", "^", "==", "!="):
            a, b = sorted((a, b))
        return (q.op, a, b)
    return None


def common_subexpression_elimination(blocks: List[BasicBlock]) -> Dict[str, int]:
    """Eliminate CSEs locally in each basic block.

    Mutates `blocks` in place.  Returns ``{'eliminated': N}``.
    """
    eliminated = 0

    for bb in blocks:
        # expr_table: (op, a, b) -> the *first* result name that computed it,
        # provided neither operand has been redefined since.
        expr_table: Dict[Tuple[str, str, str], str] = {}
        # Track which names have been redefined inside this block.
        defined_after: set = set()
        new_quads: List[Quad] = []
        for q in bb.quads:
            key = _binary_key(q)
            if key is not None and q.result is not None:
                if key in expr_table and \
                   key[1] not in defined_after and key[2] not in defined_after:
                    # Reuse: replace `r = a op b` with `r = <prev>`.
                    new_q = Quad(op="=", arg1=expr_table[key],
                                 result=q.result, line=q.line)
                    new_quads.append(new_q)
                    eliminated += 1
                else:
                    expr_table[key] = str(q.result)
                    new_quads.append(q)
            else:
                new_quads.append(q)
            # Update redefined-name tracking.
            if q.result is not None:
                # Any expression that referred to this name as an operand is
                # killed.
                killed_keys = [k for k in expr_table
                               if k[1] == str(q.result) or k[2] == str(q.result)]
                for k in killed_keys:
                    expr_table.pop(k, None)
                defined_after.add(str(q.result))

        bb.quads = new_quads

    return {"eliminated": eliminated}

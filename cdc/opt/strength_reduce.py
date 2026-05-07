"""
strength_reduce.py - Strength reduction.

Replaces expensive instructions with cheaper equivalents that have the
same value.  Local pass over the TAC quad list.

Patterns currently recognised:

    x * 2^k           ->   x << k
    x / 2^k  (signed positive constant)   ->   x >> k
    x % 2^k  (positive constant)          ->   x & (2^k - 1)
    x * 2             ->   x + x                (cheaper on some ISAs)
    x * 1   /   1 * x ->   x                    (identity)
    x + 0   /   0 + x ->   x
    x - 0             ->   x
    x * 0   /   0 * x ->   0
    pow(x, 2)         ->   x * x   (handled separately if call recognised)

Maps to syllabus
----------------
* Course Unit V/VI - Optimisation of basic blocks, machine-dependent
                     optimisation, peephole-style replacements.
"""

from __future__ import annotations

from typing import Dict, List

from ..ir.basic_block import BasicBlock
from ..ir.tac import Quad


def _power_of_two(n: int):
    """If n is a power of two >= 1, return its log2.  Else None."""
    if n is None or n <= 0:
        return None
    if n & (n - 1) == 0:
        return n.bit_length() - 1
    return None


def _maybe_int(s):
    if s is None:
        return None
    s = str(s)
    if s.lstrip("-").isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    return None


def strength_reduction(blocks: List[BasicBlock]) -> Dict[str, int]:
    """Apply strength-reduction rules block-by-block."""
    rewritten = 0
    for bb in blocks:
        new_quads: List[Quad] = []
        for q in bb.quads:
            a_int = _maybe_int(q.arg1)
            b_int = _maybe_int(q.arg2)

            if q.op == "*":
                # x * 0 -> 0
                if b_int == 0 or a_int == 0:
                    new_quads.append(Quad(op="=", arg1="0", result=q.result, line=q.line))
                    rewritten += 1; continue
                # x * 1 -> x
                if b_int == 1:
                    new_quads.append(Quad(op="=", arg1=q.arg1, result=q.result, line=q.line))
                    rewritten += 1; continue
                if a_int == 1:
                    new_quads.append(Quad(op="=", arg1=q.arg2, result=q.result, line=q.line))
                    rewritten += 1; continue
                # x * 2^k -> x << k
                k = _power_of_two(b_int) if b_int is not None and b_int > 0 else None
                if k is not None:
                    new_quads.append(Quad(op="<<", arg1=q.arg1, arg2=str(k),
                                          result=q.result, line=q.line))
                    rewritten += 1; continue
                k = _power_of_two(a_int) if a_int is not None and a_int > 0 else None
                if k is not None:
                    new_quads.append(Quad(op="<<", arg1=q.arg2, arg2=str(k),
                                          result=q.result, line=q.line))
                    rewritten += 1; continue

            elif q.op == "/":
                if b_int == 1:
                    new_quads.append(Quad(op="=", arg1=q.arg1, result=q.result, line=q.line))
                    rewritten += 1; continue
                k = _power_of_two(b_int) if b_int is not None and b_int > 0 else None
                if k is not None:
                    new_quads.append(Quad(op=">>", arg1=q.arg1, arg2=str(k),
                                          result=q.result, line=q.line))
                    rewritten += 1; continue

            elif q.op == "%":
                k = _power_of_two(b_int) if b_int is not None and b_int > 0 else None
                if k is not None:
                    new_quads.append(Quad(op="&", arg1=q.arg1,
                                          arg2=str(b_int - 1),
                                          result=q.result, line=q.line))
                    rewritten += 1; continue

            elif q.op == "+":
                if a_int == 0:
                    new_quads.append(Quad(op="=", arg1=q.arg2, result=q.result, line=q.line))
                    rewritten += 1; continue
                if b_int == 0:
                    new_quads.append(Quad(op="=", arg1=q.arg1, result=q.result, line=q.line))
                    rewritten += 1; continue

            elif q.op == "-":
                if b_int == 0:
                    new_quads.append(Quad(op="=", arg1=q.arg1, result=q.result, line=q.line))
                    rewritten += 1; continue

            new_quads.append(q)
        bb.quads = new_quads
    return {"rewritten": rewritten}

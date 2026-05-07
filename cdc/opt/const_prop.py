"""
const_prop.py - Constant Propagation + Constant Folding.

Two related local/global passes:

  * **Constant folding** evaluates compile-time-known operations
    (`5 * 4 -> 20`, `1 + 2 -> 3`, `true && x -> x`, etc.).
  * **Constant propagation** replaces uses of variables whose definition
    is a single constant assignment with the constant directly.

Implementation
--------------
The pass operates on a flat `TacProgram`.  We perform a **local
constant propagation** within each basic block (the textbook starting
point), with three rules:

  1. If we see ``x = c`` where `c` is a literal, record `x -> c`.
  2. On any other definition of `x`, drop the binding.
  3. When evaluating ``r = a op b``, if both `a` and `b` resolve to
     constants, replace the quad with ``r = (folded value)``.

A second pass then removes useless copies introduced by folding.

Maps to syllabus
----------------
* Course Unit VI - Principle Sources of Optimization, Constant
                   propagation, Optimization of basic blocks.
* Lab Practical 12 - Code optimiser.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..ir.basic_block import BasicBlock
from ..ir.tac import Quad, TacProgram


_NUMERIC_OPS = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>",
                "==", "!=", "<", ">", "<=", ">="}


def _as_const(s: Optional[str]):
    """If `s` is a string-form literal, return the Python value; else None."""
    if s is None:
        return None
    s = str(s)
    if s in ("True", "1") or s == "true":
        return True if s == "true" else 1
    if s in ("False", "0") or s == "false":
        return False if s == "false" else 0
    try:
        if "." in s or "e" in s or "E" in s:
            return float(s)
        return int(s, 0)
    except ValueError:
        return None


def _fmt_const(v):
    """Render a Python constant in the same form the lexer would emit."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _fold(op: str, a, b):
    """Apply a binary op to two constants; returns a Python value or None."""
    try:
        if op == "+":  return a + b
        if op == "-":  return a - b
        if op == "*":  return a * b
        if op == "/":
            return (a // b) if isinstance(a, int) and isinstance(b, int) else (a / b)
        if op == "%":  return a % b
        if op == "&":  return a & b
        if op == "|":  return a | b
        if op == "^":  return a ^ b
        if op == "<<": return a << b
        if op == ">>": return a >> b
        if op == "==": return 1 if a == b else 0
        if op == "!=": return 1 if a != b else 0
        if op == "<":  return 1 if a < b else 0
        if op == ">":  return 1 if a > b else 0
        if op == "<=": return 1 if a <= b else 0
        if op == ">=": return 1 if a >= b else 0
    except (ZeroDivisionError, TypeError):
        return None
    return None


def _fold_unary(op: str, a):
    try:
        if op == "uminus": return -a
        if op == "uplus":  return +a
        if op == "unot":   return 0 if a else 1
        if op == "bnot":   return ~int(a)
    except TypeError:
        return None
    return None


# ── Pass driver ────────────────────────────────────────────────────────────

def constant_propagation(prog: TacProgram, blocks: List[BasicBlock]) -> Dict[str, int]:
    """Run local constant propagation + folding over every basic block.

    Mutates `blocks` and `prog.quads` in-place.  Returns a stats dict:
    {'folded': N, 'propagated': M, 'copies_removed': K}.
    """
    folded = 0
    propagated = 0

    for bb in blocks:
        env: Dict[str, object] = {}                 # var name -> Python const
        new_quads: List[Quad] = []
        for q in bb.quads:
            # Resolve operands through the environment.
            def _resolve(x):
                nonlocal propagated
                if x is None:
                    return None
                v = _as_const(x)
                if v is not None:
                    return v
                if str(x) in env:
                    propagated += 1
                    return env[str(x)]
                return None

            if q.op == "=":
                v = _resolve(q.arg1)
                if v is not None:
                    env[str(q.result)] = v
                    new_q = Quad(op="=", arg1=_fmt_const(v), result=q.result, line=q.line)
                    new_quads.append(new_q)
                else:
                    env.pop(str(q.result), None)
                    new_quads.append(q)
                continue

            if q.op in _NUMERIC_OPS and q.arg1 is not None and q.arg2 is not None:
                a = _resolve(q.arg1); b = _resolve(q.arg2)
                if a is not None and b is not None:
                    v = _fold(q.op, a, b)
                    if v is not None:
                        env[str(q.result)] = v
                        folded += 1
                        new_quads.append(Quad(op="=", arg1=_fmt_const(v),
                                              result=q.result, line=q.line))
                        continue
                env.pop(str(q.result), None)
                new_quads.append(q)
                continue

            if q.op in ("uminus", "uplus", "unot", "bnot"):
                a = _resolve(q.arg1)
                if a is not None:
                    v = _fold_unary(q.op, a)
                    if v is not None:
                        env[str(q.result)] = v
                        folded += 1
                        new_quads.append(Quad(op="=", arg1=_fmt_const(v),
                                              result=q.result, line=q.line))
                        continue
                env.pop(str(q.result), None)
                new_quads.append(q)
                continue

            # Default: drop env entry for the result, keep the quad.
            if q.result is not None:
                env.pop(str(q.result), None)
            new_quads.append(q)

        bb.quads = new_quads

    # Rebuild prog.quads from the (now possibly shorter) blocks.
    prog.quads = [q for bb in blocks for q in bb.quads]
    return {"folded": folded, "propagated": propagated}

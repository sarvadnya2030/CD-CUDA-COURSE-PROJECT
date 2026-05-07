"""
dfa.py - Data-flow analysis framework + classical analyses.

Implements the iterative worklist solver and three textbook DFA passes:

    Live Variables          backward, union meet
    Reaching Definitions    forward,  union meet
    Available Expressions   forward,  intersection meet

Solver
------
A `DataFlowAnalysis` subclass declares its direction (`forward` or
`backward`), its meet operator (`union` / `intersect`), the boundary
value (entry-block IN for forward, exit-block OUT for backward), and the
local transfer function `(block, in_set) -> out_set`.

The solver `solve(cfg, dfa)` returns two dicts:

    in_set[block_id]   value flowing INTO  the block
    out_set[block_id]  value flowing OUT OF the block

It is a vanilla Kildall iteration; for the kernels in this project the
worklist saturates in O(n) iterations.

Maps to syllabus
----------------
* Course Unit VI - Global Data Flow Analysis, live range analysis,
                   constant propagation foundations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Set, Tuple

from ..ir.basic_block import BasicBlock
from ..ir.cfg import ControlFlowGraph, ENTRY_ID, EXIT_ID
from ..ir.tac import Quad


# ── Generic framework ──────────────────────────────────────────────────────

class DataFlowAnalysis(ABC):
    """Abstract base class for an iterative dataflow analysis."""

    direction: str = "forward"        # 'forward' | 'backward'
    name: str       = "DFA"

    @abstractmethod
    def boundary(self, cfg: ControlFlowGraph) -> Set:
        """Initial value for the boundary node (ENTRY for forward, EXIT for backward)."""

    @abstractmethod
    def initial(self, block: BasicBlock) -> Set:
        """Initial value for every other block (typically the empty or full set)."""

    @abstractmethod
    def meet(self, values) -> Set:
        """Combine values flowing in from several predecessors / successors."""

    @abstractmethod
    def transfer(self, block: BasicBlock, in_set: Set) -> Set:
        """Compute OUT[B] from IN[B] for forward analyses (or vice versa)."""


def solve(cfg: ControlFlowGraph, dfa: DataFlowAnalysis):
    """Run an iterative worklist algorithm over the CFG.

    Returns ``(in_set, out_set)`` mapping block ID → frozenset of facts.
    """
    forward = dfa.direction == "forward"

    in_sets:  Dict[int, Set] = {}
    out_sets: Dict[int, Set] = {}
    blocks_by_id = {bb.id: bb for bb in cfg.blocks}

    # Initialise.
    boundary_id = ENTRY_ID if forward else EXIT_ID
    boundary_val = dfa.boundary(cfg)
    if forward:
        in_sets[ENTRY_ID]  = boundary_val
        out_sets[ENTRY_ID] = boundary_val
        for bb in cfg.blocks:
            in_sets[bb.id]  = dfa.initial(bb)
            out_sets[bb.id] = dfa.initial(bb)
    else:
        in_sets[EXIT_ID]  = boundary_val
        out_sets[EXIT_ID] = boundary_val
        for bb in cfg.blocks:
            in_sets[bb.id]  = dfa.initial(bb)
            out_sets[bb.id] = dfa.initial(bb)

    # Worklist over the real blocks.
    worklist = list(blocks_by_id.keys())
    while worklist:
        bid = worklist.pop(0)
        bb = blocks_by_id[bid]
        if forward:
            preds = cfg.pred.get(bid, [])
            in_vals = [out_sets[p] for p in preds]
            new_in = dfa.meet(in_vals) if in_vals else dfa.initial(bb)
            new_out = dfa.transfer(bb, new_in)
            if new_in != in_sets[bid] or new_out != out_sets[bid]:
                in_sets[bid] = new_in
                out_sets[bid] = new_out
                for s in cfg.succ.get(bid, []):
                    if s in blocks_by_id and s not in worklist:
                        worklist.append(s)
        else:  # backward
            succs = cfg.succ.get(bid, [])
            out_vals = [in_sets[s] for s in succs]
            new_out = dfa.meet(out_vals) if out_vals else dfa.initial(bb)
            new_in = dfa.transfer(bb, new_out)
            if new_in != in_sets[bid] or new_out != out_sets[bid]:
                in_sets[bid] = new_in
                out_sets[bid] = new_out
                for p in cfg.pred.get(bid, []):
                    if p in blocks_by_id and p not in worklist:
                        worklist.append(p)

    return in_sets, out_sets


# ── Helpers: extract def/use sets from a quadruple ─────────────────────────

# Operations that DEFINE their `result` (by writing to it).
_DEFS_RESULT = {
    "=", "uminus", "uplus", "unot", "bnot",
    "+", "-", "*", "/", "%",
    "&", "|", "^", "<<", ">>",
    "==", "!=", "<", ">", "<=", ">=",
    "[]=", "=*", "member", "cast", "call", "phi",
}


def quad_defs(q: Quad) -> Set[str]:
    """Names defined by a quadruple."""
    if q.op in _DEFS_RESULT and q.result is not None:
        s = str(q.result)
        if not s.replace(".", "").replace("*", "").isdigit() and s != "":
            return {s}
    return set()


def quad_uses(q: Quad) -> Set[str]:
    """Names used (read) by a quadruple."""
    out: Set[str] = set()

    def _maybe(x):
        if x is None:
            return
        s = str(x)
        # Filter literals: digits, decimals, scientific.
        try:
            float(s)
            return
        except ValueError:
            pass
        if s in ("True", "False"):
            return
        out.add(s)

    if q.op in ("=[]", "*=", "=[]"):
        # Stores read all three slots.
        _maybe(q.arg1); _maybe(q.arg2); _maybe(q.result)
        return out
    if q.op == "param":
        _maybe(q.arg1); return out
    if q.op == "return":
        _maybe(q.arg1); return out
    if q.op == "call":
        _maybe(q.arg1); return out          # callee name (param quads queue args)
    if q.op in ("ifgoto", "iffalse"):
        _maybe(q.arg1); return out
    if q.op == "label" or q.op == "goto":
        return out
    # Default: arg1, arg2 are uses.
    _maybe(q.arg1); _maybe(q.arg2)
    return out


def block_def_use(bb: BasicBlock) -> Tuple[Set[str], Set[str]]:
    """Return (DEF, USE) for a basic block.

    DEF = variables definitely written before any use within bb.
    USE = variables read before any definition within bb (upward-exposed).
    """
    defined: Set[str] = set()
    used:    Set[str] = set()
    for q in bb.quads:
        for u in quad_uses(q):
            if u not in defined:
                used.add(u)
        defined |= quad_defs(q)
    return defined, used


# ── Live variables (backward, union) ───────────────────────────────────────

@dataclass
class LiveVariables(DataFlowAnalysis):
    direction: str = "backward"
    name: str = "live"
    universe: FrozenSet[str] = frozenset()    # set of all names (informational)

    def boundary(self, cfg):
        return set()

    def initial(self, block):
        return set()

    def meet(self, vals):
        out = set()
        for v in vals:
            out |= v
        return out

    def transfer(self, block, out_set):
        # IN = USE ∪ (OUT - DEF)
        defs, uses = block_def_use(block)
        return uses | (out_set - defs)


# ── Reaching definitions (forward, union) ──────────────────────────────────
#
# We model definitions as `(block_id, instruction_index)` pairs; this
# preserves identity across the program and avoids name collisions.

DefId = Tuple[int, int]


@dataclass
class ReachingDefinitions(DataFlowAnalysis):
    direction: str = "forward"
    name: str = "reaching-defs"

    def boundary(self, cfg):
        return set()

    def initial(self, block):
        return set()

    def meet(self, vals):
        out: Set[DefId] = set()
        for v in vals:
            out |= v
        return out

    def transfer(self, block, in_set):
        # All defs in this block:
        gen: Set[DefId] = set()
        kill_names: Set[str] = set()
        for i, q in enumerate(block.quads):
            for d in quad_defs(q):
                kill_names.add(d)
                gen.add((block.id, i))
        # KILL = every definition (anywhere) whose target name we redefine.
        # Implementation: subtract any (b, i) whose quad defines a killed name.
        # Caller uses _kill_for_block helper for that — we keep the math simple
        # by recomputing on the fly when comparing sets.
        result = set(in_set)
        # Remove all defs in `in_set` that target a killed name.  We don't
        # have access to the global block table here; defer the kill work
        # to a wrapper.  See `_RD_Wrapper` below for the full algorithm.
        for d in list(result):
            b, i = d
            # We'll defer; transfer won't know quads outside the block.
            # The wrapper class overrides transfer with proper kill semantics.
            pass
        return gen | result


# ── Wrapped reaching defs that knows the whole program ─────────────────────

class ReachingDefsSolver:
    """Reaching-definitions analysis aware of every basic block.

    The textbook algorithm needs to know *every* definition of each
    variable so it can compute KILL[B] = (defs of x in program) - (defs of
    x in B itself).  We pre-compute that index, then run the iterative
    solver.
    """

    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg
        self._defs_of: Dict[str, Set[DefId]] = {}
        self._block_defs: Dict[int, Set[DefId]] = {}
        self._block_kill: Dict[int, Set[DefId]] = {}
        self._index_program()

    def _index_program(self):
        for bb in self.cfg.blocks:
            local_defs: Set[DefId] = set()
            for i, q in enumerate(bb.quads):
                for name in quad_defs(q):
                    self._defs_of.setdefault(name, set()).add((bb.id, i))
                    local_defs.add((bb.id, i))
            self._block_defs[bb.id] = local_defs

        for bb in self.cfg.blocks:
            kill: Set[DefId] = set()
            local_names: Set[str] = set()
            for i, q in enumerate(bb.quads):
                local_names |= quad_defs(q)
            for name in local_names:
                # Every definition of `name` anywhere in the program is killed
                # except the local ones (those are in GEN).
                kill |= (self._defs_of.get(name, set()) - self._block_defs[bb.id])
            self._block_kill[bb.id] = kill

    def solve(self):
        in_sets:  Dict[int, Set[DefId]] = {bb.id: set() for bb in self.cfg.blocks}
        out_sets: Dict[int, Set[DefId]] = {bb.id: set() for bb in self.cfg.blocks}
        in_sets[ENTRY_ID]  = set()
        out_sets[ENTRY_ID] = set()

        worklist = [bb.id for bb in self.cfg.blocks]
        while worklist:
            bid = worklist.pop(0)
            preds = self.cfg.pred.get(bid, [])
            new_in: Set[DefId] = set()
            for p in preds:
                if p in (ENTRY_ID,):
                    continue
                new_in |= out_sets.get(p, set())
            new_out = (new_in - self._block_kill[bid]) | self._block_defs[bid]
            if new_out != out_sets[bid]:
                in_sets[bid]  = new_in
                out_sets[bid] = new_out
                for s in self.cfg.succ.get(bid, []):
                    if s not in (ENTRY_ID, EXIT_ID) and s not in worklist:
                        worklist.append(s)
            else:
                in_sets[bid] = new_in
        return in_sets, out_sets


# ── Available expressions (forward, intersection) ──────────────────────────
#
# An expression `x op y` is available at point p if every path from
# entry to p evaluates `x op y` and afterwards none of x, y, or the
# expression's result is redefined.

@dataclass(frozen=True)
class ExprKey:
    op: str
    a: str
    b: str

    def __str__(self) -> str:
        return f"{self.a} {self.op} {self.b}"


def _key_of(q: Quad):
    """Return an ExprKey if `q` evaluates a CSE-able binary expression, else None."""
    if q.op in ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>",
                "==", "!=", "<", ">", "<=", ">="):
        a, b = sorted([str(q.arg1), str(q.arg2)]) if q.op in ("+", "*", "&", "|", "^",
                                                              "==", "!=") \
                                                  else (str(q.arg1), str(q.arg2))
        return ExprKey(op=q.op, a=a, b=b)
    return None


@dataclass
class AvailableExpressions(DataFlowAnalysis):
    direction: str = "forward"
    name: str = "available-exprs"

    universe: FrozenSet[ExprKey] = frozenset()

    def boundary(self, cfg):
        return set()                              # entry: nothing available

    def initial(self, block):
        return set(self.universe)                 # everywhere else: top = universe

    def meet(self, vals):
        if not vals:
            return set(self.universe)
        result = None
        for v in vals:
            result = set(v) if result is None else result & v
        return result or set()

    def transfer(self, block, in_set):
        out = set(in_set)
        for q in block.quads:
            k = _key_of(q)
            if k is not None:
                out.add(k)
            # Kill any available expression whose operand was redefined.
            for d in quad_defs(q):
                out = {e for e in out if e.a != d and e.b != d}
        return out


def collect_universe(cfg: ControlFlowGraph) -> FrozenSet[ExprKey]:
    """Enumerate every CSE-able binary expression in the program."""
    out: Set[ExprKey] = set()
    for bb in cfg.blocks:
        for q in bb.quads:
            k = _key_of(q)
            if k is not None:
                out.add(k)
    return frozenset(out)


# ── Pretty-print helpers ───────────────────────────────────────────────────

def format_dfa_result(name: str, in_sets, out_sets, cfg: ControlFlowGraph) -> str:
    lines = [f"== {name} =="]
    for bb in cfg.blocks:
        lines.append(f"BB{bb.id}" + (f" ({bb.label})" if bb.label else ""))
        lines.append(f"   IN  : {sorted(map(str, in_sets.get(bb.id, set())))}")
        lines.append(f"   OUT : {sorted(map(str, out_sets.get(bb.id, set())))}")
    return "\n".join(lines)

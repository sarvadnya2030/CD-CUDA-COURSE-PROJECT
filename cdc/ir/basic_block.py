"""
basic_block.py — partition a TAC program into basic blocks.

A *basic block* is a maximal sequence of consecutive quadruples such that
control enters at the first instruction and leaves at the last, with no
internal branches.  Basic blocks are the units operated on by

  * the control-flow graph (cfg.py),
  * data-flow analysis (opt/dfa.py, planned for Phase 3),
  * the DAG-of-basic-block (dag.py),
  * peephole and code-generation passes.

Algorithm (Aho/Ullman §8.4)
---------------------------
A quadruple Q is a *leader* if any of:

  1. Q is the first quadruple of the program.
  2. Q is the target of a goto/ifgoto/iffalse.
  3. Q immediately follows a goto/ifgoto/iffalse/return.

Every basic block starts at a leader and continues to (but not including)
the next leader.  Labels are absorbed into the block they begin; the
control instruction that ends a block is included as its last quadruple.

Maps to syllabus
----------------
* Course Unit V — Basic Blocks and Flow Graphs
* Tutorial      — 13 (DAG examples; basic-block step is its prerequisite)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .tac import Quad, TacProgram


# ── Basic block ────────────────────────────────────────────────────────────

@dataclass
class BasicBlock:
    """A straight-line sequence of quadruples."""
    id: int
    label: Optional[str] = None             # name of the leading label, if any
    start: int = 0                          # index of the leader in prog.quads
    end: int = 0                            # exclusive end index
    quads: List[Quad] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.quads

    def terminator(self) -> Optional[Quad]:
        if not self.quads:
            return None
        last = self.quads[-1]
        return last if last.is_terminator() else None

    def __str__(self) -> str:
        head = f"BB{self.id}" + (f"  ({self.label}:)" if self.label else "")
        body = "\n".join(f"    {q}" for q in self.quads)
        return f"{head}\n{body}"


# ── Partitioner ────────────────────────────────────────────────────────────

def _find_leaders(prog: TacProgram) -> List[int]:
    """Return sorted list of indices in `prog.quads` that begin a basic block."""
    quads = prog.quads
    if not quads:
        return []

    leaders: Set[int] = {0}

    # Map every label name to the index of its `label` quadruple.
    label_to_idx: Dict[str, int] = {}
    for i, q in enumerate(quads):
        if q.is_label():
            label_to_idx[str(q.result)] = i

    for i, q in enumerate(quads):
        if q.is_branch():
            tgt = str(q.result)
            if tgt in label_to_idx:
                leaders.add(label_to_idx[tgt])
            # Quad immediately after any branch is also a leader.
            if i + 1 < len(quads):
                leaders.add(i + 1)
        elif q.op == "return":
            if i + 1 < len(quads):
                leaders.add(i + 1)
        elif q.is_label():
            leaders.add(i)

    return sorted(leaders)


def partition_blocks(prog: TacProgram) -> List[BasicBlock]:
    """
    Partition `prog.quads` into a list of `BasicBlock`s.

    Empty programs return an empty list.  Otherwise the returned blocks
    cover every quadruple exactly once and are ordered by their leader
    index.
    """
    quads = prog.quads
    if not quads:
        return []

    leaders = _find_leaders(prog)
    leaders.append(len(quads))   # sentinel for the final slice

    blocks: List[BasicBlock] = []
    for bi in range(len(leaders) - 1):
        s = leaders[bi]
        e = leaders[bi + 1]
        slice_ = quads[s:e]
        # The block "label" is the leading-label quadruple's `result`,
        # if the first quad happens to be a label.
        label = None
        if slice_ and slice_[0].is_label():
            label = str(slice_[0].result)
        blocks.append(BasicBlock(
            id=bi, label=label, start=s, end=e, quads=list(slice_),
        ))

    return blocks


# ── Reporting ──────────────────────────────────────────────────────────────

def format_blocks(blocks: List[BasicBlock]) -> str:
    """Pretty-print a list of basic blocks."""
    if not blocks:
        return "(no basic blocks)"
    lines = []
    for bb in blocks:
        lines.append(str(bb))
        lines.append("")
    return "\n".join(lines).rstrip()

"""
cdc.ir — Intermediate Representation pipeline.

Phase 2 of the compiler design layer.  Lowers an AST produced by the
frontend into:

    AST  ──► tac.py        ──► linear three-address code (quadruples)
                  │
                  ▼
            basic_block.py  ──► list of BasicBlock objects
                  │
                  ▼
                cfg.py      ──► ControlFlowGraph (succ/pred + dominators)
                  │
                  ▼
                dag.py      ──► per-basic-block DAG (value numbering)

Each module is independently testable; the orchestration entry point is
`cdc.frontend.run_frontend()` extended via `cdc.ir.lower(kernel)` (added
when Phase 3 hooks the optimisation passes).

Maps to syllabus
----------------
* Course Unit IV — Intermediate Code Generation (quadruples form)
* Course Unit V  — Basic Blocks and Flow Graphs, DAG representation,
                   Generating code from DAGs
* Tutorials 12, 13 — Quadruples and DAG examples
"""

from .tac import Quad, TacProgram, emit_tac
from .basic_block import BasicBlock, partition_blocks
from .cfg import ControlFlowGraph, build_cfg
from .dag import DagNode, DagBuilder, build_dag

__all__ = [
    "Quad", "TacProgram", "emit_tac",
    "BasicBlock", "partition_blocks",
    "ControlFlowGraph", "build_cfg",
    "DagNode", "DagBuilder", "build_dag",
]

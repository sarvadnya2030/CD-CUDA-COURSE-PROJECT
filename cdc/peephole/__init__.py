"""
cdc.peephole - PTX peephole optimisation.

Phase 4 of the compiler design layer.  Operates on the PTX assembly
emitted by `nvcc -ptx`, applying small pattern-based rewrites that the
NVIDIA compiler does not always perform.  Each pattern is documented as
a (before, after) pair and is logged when it fires so the auto-tuner's
report can show which rewrites mattered.

This module is the textbook "peephole optimiser" from CD Unit V, but
operating on a real production IR (NVIDIA PTX) rather than a toy ISA.
"""

from .ptx_peephole import (
    PtxPeepholeOptimizer, PeepholePattern, optimise_ptx_file,
    summarise_passes,
)

__all__ = [
    "PtxPeepholeOptimizer",
    "PeepholePattern",
    "optimise_ptx_file",
    "summarise_passes",
]

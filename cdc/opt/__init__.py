"""
cdc.opt - Phase 3 optimisation passes.

Built on top of the CFG and DAG produced by `cdc.ir`.  Each module is an
independently runnable pass:

    dfa.py              Data-flow analysis framework + 3 classical analyses
                        (reaching definitions, live variables,
                        available expressions).  Iterative worklist solver.
    const_prop.py       Constant propagation + constant folding.
    cse.py              Common-subexpression elimination via DAG value
                        numbering.
    dce.py              Dead-code elimination driven by live-variable
                        analysis.
    licm.py             Loop-invariant code motion (uses dominators +
                        reaching definitions).
    strength_reduce.py  Strength reduction (i*4 -> i<<2, i/2 -> i>>1, etc.).
    register_pressure.py Estimate live-range count per program point and
                        feed the auto-tuner's tile/unroll variant pruner.

Maps to syllabus
----------------
* Course Unit VI - Code Optimization, Run-Time Environments, Data Flow
                   Analysis (constant propagation, live range analysis,
                   global DFA, optimisation of basic blocks).
* Lab Practical 12 - Code optimiser for C/C++ subset.
"""

from .dfa import (
    DataFlowAnalysis, LiveVariables, ReachingDefinitions, AvailableExpressions,
    solve,
)
from .const_prop  import constant_propagation
from .cse         import common_subexpression_elimination
from .dce         import dead_code_elimination
from .licm        import loop_invariant_code_motion
from .strength_reduce import strength_reduction
from .register_pressure import estimate_register_pressure, RegisterPressureReport

__all__ = [
    "DataFlowAnalysis", "LiveVariables", "ReachingDefinitions",
    "AvailableExpressions", "solve",
    "constant_propagation",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "loop_invariant_code_motion",
    "strength_reduction",
    "estimate_register_pressure", "RegisterPressureReport",
]

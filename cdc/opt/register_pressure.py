"""
register_pressure.py - Estimate per-program-point register pressure
                       from live-variable analysis, then expose a
                       cost-model hook for the auto-tuner.

Why this matters
----------------
The CD course requires "live range analysis" (Unit VI) and explicitly
mentions "machine-dependent optimisation".  This module ties both into
the original GPU project: by counting how many SSA-style values are
simultaneously live at the busiest point of a kernel, we get an
analytic estimate of register pressure.  That estimate then prunes the
auto-tuner's tile/unroll variant grid so we never compile a variant
that we can predict will spill.

Algorithm
---------
1. Run `LiveVariables` on the kernel CFG (already implemented).
2. For every basic block, simulate the live set instruction by
   instruction (backward) using the OUT[B] live set as starting point.
3. Track the maximum |live set| seen across all program points.  This
   is the lower-bound on registers needed to honour every live range
   without spilling under a perfect register allocator.
4. Apply a small fudge factor for the GPU-specific overhead of
   thread-block scaling (each thread holds its own copy of the live
   ranges).  The exposed budget number is consumed by the auto-tuner
   to decide whether a candidate (tile_x, tile_y, unroll) configuration
   is feasible on RTX 2070 (max 255 fp32 regs/thread before spills).

Maps to syllabus
----------------
* Course Unit VI - Live range analysis, global DFA, machine-dependent
                   optimisation.
* Course case study - Compilation in multicore environment, parallel
                      compilers, deep learning compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..ir.basic_block import BasicBlock
from ..ir.cfg import ControlFlowGraph
from ..ir.tac import Quad
from .dfa import LiveVariables, solve, quad_defs, quad_uses


@dataclass
class RegisterPressureReport:
    kernel: str
    max_live: int                              # peak |live set| in any block
    avg_live: float                             # mean across program points
    by_block: Dict[int, int]                    # peak per basic block
    suggested_max_unroll: int                   # heuristic from max_live
    suggested_max_tile:   int                   # heuristic from max_live
    rtx2070_reg_budget:   int = 65536           # 64K regs/block, sm_75
    rtx2070_threads_per_block_for_full_occ: int = 1024


def estimate_register_pressure(name: str,
                               blocks: List[BasicBlock],
                               cfg: ControlFlowGraph) -> RegisterPressureReport:
    """Compute the live-set peak across every program point in the kernel."""
    lv = LiveVariables()
    in_sets, out_sets = solve(cfg, lv)

    max_live  = 0
    by_block: Dict[int, int] = {}
    pressure_samples: List[int] = []

    for bb in blocks:
        live = set(out_sets.get(bb.id, set()))
        peak = len(live)
        # Walk backward through the block.
        for q in reversed(bb.quads):
            for d in quad_defs(q):
                live.discard(d)
            for u in quad_uses(q):
                live.add(u)
            peak = max(peak, len(live))
            pressure_samples.append(len(live))
        by_block[bb.id] = peak
        max_live = max(max_live, peak)

    avg_live = sum(pressure_samples) / max(len(pressure_samples), 1)

    # Heuristic: convert live-range count into auto-tuner caps.  The RTX
    # 2070 budget is 65,536 registers per block (sm_75).  Aiming for full
    # occupancy of 1024 threads/block leaves 64 fp32 regs/thread.  Each
    # active SSA value in our analysis ≈ 1 register, so:
    #
    #   budget   = 64  (regs/thread, full occupancy target)
    #   reserve  = 16  (compiler/runtime overhead, ld/st, return-address)
    #   live_avail = budget - reserve - max_live  (per-thread headroom)
    #
    # Each unit of unroll roughly doubles the live registers used in the
    # inner loop, so suggested_max_unroll = floor(log2(headroom + 1)).
    headroom = max(1, 64 - 16 - max_live)
    suggested_max_unroll = max(1, int(headroom ** 0.5))
    # Tile suggestion is more conservative; for 2D tiles keep tile**2 ≤ headroom.
    suggested_max_tile = max(1, int(headroom ** 0.5))

    return RegisterPressureReport(
        kernel=name,
        max_live=max_live,
        avg_live=avg_live,
        by_block=by_block,
        suggested_max_unroll=suggested_max_unroll,
        suggested_max_tile=suggested_max_tile,
    )


def format_report(rp: RegisterPressureReport) -> str:
    """Render a RegisterPressureReport as a human-readable summary."""
    lines = [f"== Register Pressure: {rp.kernel} =="]
    lines.append(f"  max  live values : {rp.max_live}")
    lines.append(f"  avg  live values : {rp.avg_live:.1f}")
    lines.append(f"  RTX 2070 budget  : {rp.rtx2070_reg_budget} regs/block "
                 f"(sm_75; ≤ 64 regs/thread for 1024-thread full occupancy)")
    lines.append(f"  cost-model hint  : "
                 f"max_unroll={rp.suggested_max_unroll}, "
                 f"max_tile={rp.suggested_max_tile}")
    lines.append("  per-block peaks  :")
    for bid in sorted(rp.by_block):
        lines.append(f"    BB{bid}: {rp.by_block[bid]}")
    return "\n".join(lines)

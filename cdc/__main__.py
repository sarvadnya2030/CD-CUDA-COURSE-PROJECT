"""
cdc CLI — `python -m cdc <file.cu> [flags]`

Drives the compiler pipeline on a CUDA source file and prints the chosen
phase output.  Default is the Phase-1 frontend report.

Phase 1 (frontend)
------------------
  --tokens       Dump the token stream (lexer phase only).
  --ast          Include the full AST in the report.
  --no-symbols   Suppress the symbol-table dump.
  --diag-only    Print only diagnostics (errors/warnings).

Phase 2 (IR)
------------
  --tac          Emit three-address code for every kernel.
  --bb           Emit basic-block partitioning per kernel.
  --cfg          Emit control-flow graph (edges + dominator tree).
  --dot          Emit Graphviz DOT for the CFG (one per kernel).
  --dag          Emit per-basic-block DAG.
  --ir           Shorthand for --tac --bb --cfg --dag.

Filtering
---------
  --kernel NAME  Only run the chosen phase on the named kernel.

Running this module is the answer to a CD examiner asking "where is your
scanner / parser / IR generator / DAG?" — the output makes every textbook
phase visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .frontend import run_frontend, format_report
from .lexer import tokenize
from .preprocessor import preprocess
from .ir import (
    emit_tac, partition_blocks, build_cfg, build_dag,
)
from .ir.basic_block import format_blocks
from .ir.dag import format_dag
from .first_follow import (
    build_example_grammar, compute_nullable, compute_first,
    compute_follow, build_ll1_table, format_first_follow, format_ll1_table,
)


def _phase2_for_kernel(k, opts) -> str:
    """Render the requested Phase-2 / Phase-3 outputs for one FrontendKernel."""
    out = []
    prog = emit_tac(k.ast)

    if opts.tac:
        out.append(f"== TAC: {k.name} ==")
        out.append(prog.numbered())
        out.append("")

    blocks = partition_blocks(prog)

    if opts.bb:
        out.append(f"== Basic blocks: {k.name}  ({len(blocks)} blocks) ==")
        out.append(format_blocks(blocks))
        out.append("")

    cfg = build_cfg(blocks)

    if opts.cfg:
        out.append(f"== CFG: {k.name} ==")
        out.append(cfg.format_edges())
        out.append("")
        out.append(cfg.format_dominators())
        out.append("")

    if opts.dot:
        dot_path = Path("results") / f"cfg_{k.name}.dot"
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        dot_path.write_text(cfg.to_dot(), encoding="utf-8")
        out.append(f"[dot] CFG written to {dot_path}")

    if opts.dag:
        out.append(f"== DAG (per basic block): {k.name} ==")
        for bb in blocks:
            if not bb.quads:
                continue
            nodes = build_dag(bb)
            label = f"BB{bb.id}" + (f" ({bb.label})" if bb.label else "")
            out.append(f"-- DAG of {label} --")
            out.append(format_dag(nodes))
            out.append("")

    # ── Phase 3 ─────────────────────────────────────────────────────────
    if opts.dfa:
        from .opt.dfa import (
            LiveVariables, ReachingDefsSolver, AvailableExpressions,
            collect_universe, solve, format_dfa_result,
        )
        out.append(f"== DFA: {k.name} ==")

        lv = LiveVariables()
        in_l, out_l = solve(cfg, lv)
        out.append(format_dfa_result("Live Variables (backward, union)",
                                     in_l, out_l, cfg))
        out.append("")

        rd = ReachingDefsSolver(cfg)
        in_r, out_r = rd.solve()
        # Convert (block_id, idx) tuples to short labels for display.
        in_r_str  = {b: {f"BB{x[0]}.q{x[1]}" for x in v} for b, v in in_r.items()}
        out_r_str = {b: {f"BB{x[0]}.q{x[1]}" for x in v} for b, v in out_r.items()}
        out.append(format_dfa_result("Reaching Definitions (forward, union)",
                                     in_r_str, out_r_str, cfg))
        out.append("")

        ae = AvailableExpressions()
        ae.universe = collect_universe(cfg)
        in_a, out_a = solve(cfg, ae)
        out.append(format_dfa_result("Available Expressions (forward, intersect)",
                                     in_a, out_a, cfg))
        out.append("")

    if opts.opt:
        from .opt import (
            constant_propagation, common_subexpression_elimination,
            dead_code_elimination, loop_invariant_code_motion,
            strength_reduction,
        )
        cp_stats   = constant_propagation(prog, blocks)
        cse_stats  = common_subexpression_elimination(blocks)
        sr_stats   = strength_reduction(blocks)
        licm_stats = loop_invariant_code_motion(blocks, cfg)
        dce_stats  = dead_code_elimination(blocks, cfg)
        out.append(f"== Optimisation passes: {k.name} ==")
        out.append(f"  constant folding   : {cp_stats['folded']:>3} folded, "
                   f"{cp_stats['propagated']:>3} propagated")
        out.append(f"  CSE                : {cse_stats['eliminated']:>3} eliminated")
        out.append(f"  strength reduction : {sr_stats['rewritten']:>3} rewritten")
        out.append(f"  LICM               : {licm_stats['loops_found']:>3} loops, "
                   f"{licm_stats['invariants_identified']:>3} invariants, "
                   f"{licm_stats['hoisted']:>3} hoisted")
        out.append(f"  dead-code          : {dce_stats['removed']:>3} removed")
        out.append("")
        out.append("-- Optimised TAC --")
        # Re-flatten prog after passes that mutated blocks.
        prog.quads = [q for bb in blocks for q in bb.quads]
        out.append(prog.numbered())
        out.append("")

    if opts.regs:
        from .opt.register_pressure import (
            estimate_register_pressure, format_report,
        )
        rp = estimate_register_pressure(k.name, blocks, cfg)
        out.append(format_report(rp))
        out.append("")

    return "\n".join(out).rstrip()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cdc",
                                description="Compiler pipeline for the CUDA auto-tuner.")
    p.add_argument("file", type=Path, help="CUDA source file (.cu)")

    # Phase 1 flags
    p.add_argument("--tokens", action="store_true",
                   help="Dump token stream and exit (lexer phase only)")
    p.add_argument("--ast", action="store_true",
                   help="Include the full AST in the frontend report")
    p.add_argument("--no-symbols", action="store_true",
                   help="Suppress the symbol-table dump")
    p.add_argument("--diag-only", action="store_true",
                   help="Only print diagnostics (errors/warnings)")

    # Phase 2 flags
    p.add_argument("--tac", action="store_true", help="Emit three-address code")
    p.add_argument("--bb", action="store_true", help="Emit basic-block partitioning")
    p.add_argument("--cfg", action="store_true", help="Emit CFG edges + dominators")
    p.add_argument("--dot", action="store_true", help="Write CFG to results/cfg_<kernel>.dot")
    p.add_argument("--dag", action="store_true", help="Emit per-block DAG")
    p.add_argument("--ir", action="store_true",
                   help="Shorthand for --tac --bb --cfg --dag")

    # Phase 3 flags
    p.add_argument("--dfa", action="store_true",
                   help="Run live-vars / reaching-defs / available-exprs analyses")
    p.add_argument("--opt", action="store_true",
                   help="Run all classical optimisation passes (const-prop, "
                        "CSE, DCE, LICM, strength-reduction); print stats and "
                        "the optimised TAC")
    p.add_argument("--regs", action="store_true",
                   help="Estimate register pressure from live-variable analysis")

    # Unit III: FIRST/FOLLOW (Tutorial 5, Lab Practical 7)
    p.add_argument("--first-follow", action="store_true",
                   help="Compute and display FIRST/FOLLOW sets and LL(1) parse table "
                        "(Unit III: LL(1) parser construction)")

    p.add_argument("--kernel", type=str, default=None,
                   help="Only emit phase output for the named kernel")

    args = p.parse_args(argv)
    if args.ir:
        args.tac = args.bb = args.cfg = args.dag = True

    src = args.file.read_text(encoding="utf-8")

    # Phase 1: tokens-only
    if args.tokens:
        cleaned = preprocess(src)
        for tok in tokenize(cleaned):
            print(f"{tok.lineno:>4}: {tok.type:<14} {tok.value!r}")
        return 0

    # Unit III: FIRST/FOLLOW
    if args.first_follow:
        grammar = build_example_grammar()
        nullable = compute_nullable(grammar)
        first = compute_first(grammar, nullable)
        follow = compute_follow(grammar, first, nullable)
        print(format_first_follow(first, follow, grammar.non_terminals))
        print()
        table = build_ll1_table(grammar, first, follow, nullable)
        print(format_ll1_table(table, grammar))
        return 0

    # Always run the frontend (every other phase needs the AST).
    res = run_frontend(args.file)

    if args.diag_only:
        if not res.all_diagnostics():
            print("No diagnostics - frontend pipeline clean.")
            return 0
        for d in res.all_diagnostics():
            print(d)
        return 0 if res.ok() else 1

    phase2 = (args.tac or args.bb or args.cfg or args.dot or args.dag or
              args.dfa or args.opt or args.regs)
    if phase2:
        for k in res.kernels:
            if args.kernel and k.name != args.kernel:
                continue
            if not k.ok():
                print(f"== {k.name}: skipped (frontend errors) ==")
                continue
            print(_phase2_for_kernel(k, args))
            print()
        return 0 if res.ok() else 1

    # Default: frontend report.
    print(format_report(res, show_ast=args.ast, show_symbols=not args.no_symbols))
    return 0 if res.ok() else 1


if __name__ == "__main__":
    sys.exit(main())

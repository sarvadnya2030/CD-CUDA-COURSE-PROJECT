"""
cd_mapping_figure.py — render the syllabus-to-code mapping as a single
PNG slide for the next presentation.

Output: results/plots/cd_mapping.png

Usage:
    python -m src.cd_mapping_figure

The figure is a directed-graph-style layout where every CD syllabus
unit appears on the left, every project module on the right, and edges
connect them.  Colour-coded by unit so the professor can see at a glance
that every unit lights up.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# ── Mapping data ───────────────────────────────────────────────────────────

UNITS = [
    ("Unit I",   "Compiler phases\nCross-compiler\nLinker / Loader",
                 ["cdc/frontend.py", "autotune.py (orchestration)"]),
    ("Unit II",  "Lexical Analysis\nLEX/FLEX\nRegular expressions",
                 ["cdc/lexer.py (PLY)", "cdc/preprocessor.py"]),
    ("Unit III", "LR/LALR Parsing\nYACC/BISON\nSymbol Table\nType Checking",
                 ["cdc/parser.py (PLY YACC)", "cdc/symbol_table.py", "cdc/type_check.py"]),
    ("Unit IV",  "Syntax-Directed Translation\nIntermediate Code (Quadruples)\nError Recovery",
                 ["cdc/ast_nodes.py", "cdc/ir/tac.py", "Diagnostic stream"]),
    ("Unit V",   "Code Generation\nBasic Blocks / Flow Graphs\nDAG of basic blocks\nPeephole Optimization",
                 ["cdc/ir/basic_block.py", "cdc/ir/cfg.py", "cdc/ir/dag.py", "cdc/peephole/ptx_peephole.py"]),
    ("Unit VI",  "Code Optimization\nGlobal Data Flow Analysis\nLive Range Analysis\nLoop Optimization\nMachine-Dependent Opt.",
                 ["cdc/opt/dfa.py", "cdc/opt/const_prop.py", "cdc/opt/cse.py",
                  "cdc/opt/dce.py", "cdc/opt/licm.py",
                  "cdc/opt/strength_reduce.py", "cdc/opt/register_pressure.py"]),
]

UNIT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def render(output: Path) -> Path:
    n = len(UNITS)
    # Each row sized to comfortably fit the largest unit (Unit VI: 5 lines).
    row_h_in = 2.0
    fig_h = 3.0 + row_h_in * n
    fig, ax = plt.subplots(figsize=(15, fig_h), dpi=120)

    # Title
    fig.suptitle(
        "Compiler Design Syllabus  ->  CUDA Auto-Tuner Project",
        fontsize=20, fontweight="bold", y=0.97,
    )
    ax.text(0.5, 0.945, "CS3053 . VIT . AY 2024-25",
            transform=fig.transFigure, ha="center", fontsize=12, color="#555")

    # Vertical layout in axes coords.
    top = 0.92
    bottom = 0.08
    span = top - bottom
    row_h = span / n              # full slot per unit
    gap = row_h * 0.12            # gap between consecutive rows
    box_h = row_h - gap           # actual box height

    left_x      = 0.03
    right_x     = 0.53
    box_w_left  = 0.44
    box_w_right = 0.44

    for i, (unit, topics, files) in enumerate(UNITS):
        y = top - (i + 1) * row_h + gap / 2
        col = UNIT_COLORS[i]

        # Left box (unit header strip + topics)
        rect = patches.FancyBboxPatch(
            (left_x, y), box_w_left, box_h,
            boxstyle="round,pad=0.005",
            linewidth=1.5, edgecolor=col, facecolor=col + "1A",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Coloured header strip across the top of the left box.
        header_h = box_h * 0.25
        header = patches.Rectangle(
            (left_x, y + box_h - header_h),
            box_w_left, header_h,
            linewidth=0, facecolor=col, alpha=0.85,
            transform=ax.transAxes,
        )
        ax.add_patch(header)
        ax.text(
            left_x + 0.012, y + box_h - header_h / 2, unit,
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            color="white", va="center",
        )

        # Topics text below the strip.
        ax.text(
            left_x + 0.012, y + (box_h - header_h) * 0.55, topics,
            transform=ax.transAxes, fontsize=11, va="center",
            color="#202020", family="DejaVu Sans",
        )

        # Right box (file list)
        rect2 = patches.FancyBboxPatch(
            (right_x, y), box_w_right, box_h,
            boxstyle="round,pad=0.005",
            linewidth=1.5, edgecolor=col, facecolor="#ffffff",
            transform=ax.transAxes,
        )
        ax.add_patch(rect2)
        files_text = "\n".join("- " + f for f in files)
        ax.text(
            right_x + 0.015, y + box_h / 2, files_text,
            transform=ax.transAxes, fontsize=10.5, va="center",
            family="DejaVu Sans Mono", color="#202020",
        )

        # Connecting arrow at vertical centre.
        ax.annotate(
            "",
            xy=(right_x, y + box_h / 2),
            xytext=(left_x + box_w_left, y + box_h / 2),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.8, color=col),
        )

    # Footer
    ax.text(
        0.5, 0.04,
        "Detailed mapping: CD_MAPPING.md  .  All artefacts demonstrable in viva",
        transform=fig.transFigure, ha="center",
        fontsize=11, color="#444",
    )
    ax.text(
        0.5, 0.015,
        "Demo: python -m cdc src/kernels/baseline_kernels.cu --ir --opt --kernel matmul_naive",
        transform=fig.transFigure, ha="center",
        fontsize=10, family="DejaVu Sans Mono",
        color="#444",
    )

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    out = Path("results") / "plots" / "cd_mapping.png"
    rendered = render(out)
    print(f"Wrote {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

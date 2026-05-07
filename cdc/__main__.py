"""
cdc CLI — `python -m cdc <file.cu>`

Drives the Phase-1 frontend pipeline on a CUDA source file and prints a
human-readable report (kernels found, parameters, symbol-table scopes,
diagnostics).

Flags
-----
  --tokens       Dump the token stream (lexer phase only).
  --ast          Include the full AST in the report.
  --no-symbols   Suppress the symbol-table dump.
  --diag-only    Only print diagnostics (errors/warnings).

This module is the answer to a CD examiner asking "where is your scanner
and parser?" — running it produces visible output of every phase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .frontend import run_frontend, format_report
from .lexer import tokenize
from .preprocessor import preprocess


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cdc",
                                description="Compiler frontend for the CUDA auto-tuner.")
    p.add_argument("file", type=Path, help="CUDA source file (.cu)")
    p.add_argument("--tokens", action="store_true",
                   help="Dump token stream and exit (lexer phase only)")
    p.add_argument("--ast", action="store_true",
                   help="Include the full AST in the report")
    p.add_argument("--no-symbols", action="store_true",
                   help="Suppress the symbol-table dump")
    p.add_argument("--diag-only", action="store_true",
                   help="Only print diagnostics (errors/warnings)")
    args = p.parse_args(argv)

    src = args.file.read_text(encoding="utf-8")

    if args.tokens:
        cleaned = preprocess(src)
        for tok in tokenize(cleaned):
            print(f"{tok.lineno:>4}: {tok.type:<14} {tok.value!r}")
        return 0

    res = run_frontend(args.file)

    if args.diag_only:
        if not res.all_diagnostics():
            print("No diagnostics — frontend pipeline clean.")
            return 0
        for d in res.all_diagnostics():
            print(d)
        return 0 if res.ok() else 1

    print(format_report(res, show_ast=args.ast,
                        show_symbols=not args.no_symbols))
    return 0 if res.ok() else 1


if __name__ == "__main__":
    sys.exit(main())

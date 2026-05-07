"""
cdc — Compiler Design Components for the CUDA auto-tuner.

Phase 1 — frontend:
    lexer.py         PLY tokens for a CUDA C subset
    parser.py        PLY grammar producing an AST
    ast_nodes.py     AST class hierarchy
    symbol_table.py  Scoped symbol table (kernel / shared / register)
    type_check.py    Type and address-space rules
    preprocessor.py  Comment / #include / macro stripping
    frontend.py      Orchestrates preprocess → lex → parse → symtab → typecheck

Phase 2 — IR (planned):
    ir/tac.py, ir/basic_block.py, ir/cfg.py, ir/dag.py

Phase 3 — optimization (planned):
    opt/dfa.py, opt/const_prop.py, opt/cse.py, opt/dce.py, opt/licm.py,
    opt/strength_reduce.py

Phase 4 — backend (planned):
    peephole/ptx_peephole.py
"""

__all__ = [
    "ast_nodes",
    "lexer",
    "parser",
    "symbol_table",
    "type_check",
    "preprocessor",
    "frontend",
]

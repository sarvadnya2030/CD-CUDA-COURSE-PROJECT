"""
parser.py — PLY (Python Lex-Yacc) grammar for a CUDA C subset.

This is the syntax-analysis phase of the compiler frontend (CD Units II/III).
It corresponds to a YACC/BISON specification with action rules building
an abstract syntax tree (CD Unit IV — syntax-directed translation).

The grammar is LALR(1) and uses standard precedence/associativity rules
for C-style operators.  It accepts:

  translation_unit  →  function_definition+
  function_definition
                    →  qualifier* type_spec declarator '(' params ')' compound
  declaration       →  qualifier* type_spec init_declarator (',' init_declarator)* ';'
  init_declarator   →  declarator ('=' assignment_expression)?
  declarator        →  '*'? '__restrict__'? IDENT ('[' expr? ']')*

…plus the usual statement/expression hierarchy.

The grammar is intentionally permissive about pointer placement and
storage qualifiers; rigorous semantic checks happen in `type_check.py`.

Maps to syllabus
----------------
* Course Unit III  — Bottom-Up Parsing, LR/LALR, YACC/BISON
* Course Unit IV   — Syntax-Directed Translation, AST construction
* Lab Practicals   — 7 (LR/SLR/LALR), 8 (YACC), 10 (SDT)
"""

from __future__ import annotations

import ply.yacc as yacc

from . import ast_nodes as A
from .lexer import tokens, make_lexer  # noqa: F401  (PLY needs `tokens`)


# ── Operator precedence (lowest → highest) ──────────────────────────────────

precedence = (
    ("right", "ASSIGN", "PLUSEQ", "MINUSEQ", "STAREQ", "SLASHEQ", "PERCENTEQ",
              "ANDEQ", "OREQ", "XOREQ", "LSHIFTEQ", "RSHIFTEQ"),
    ("right", "QUESTION", "COLON"),
    ("left",  "LOR"),
    ("left",  "LAND"),
    ("left",  "PIPE"),
    ("left",  "CARET"),
    ("left",  "AMP"),
    ("left",  "EQ", "NEQ"),
    ("left",  "LT", "GT", "LE", "GE"),
    ("left",  "LSHIFT", "RSHIFT"),
    ("left",  "PLUS", "MINUS"),
    ("left",  "STAR", "SLASH", "PERCENT"),
    ("right", "LNOT", "TILDE", "INCREMENT", "DECREMENT", "UMINUS", "UPLUS",
              "DEREF", "ADDROF", "CAST"),
    ("left",  "LPAREN", "LBRACKET", "DOT"),
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _set_loc(node, p, idx=1):
    """Copy line/lexpos from a YACC production element into an AST node."""
    if hasattr(p, "lineno"):
        try:
            node.line = p.lineno(idx)
        except Exception:
            node.line = 0
    return node


# ── Translation unit ────────────────────────────────────────────────────────

def p_translation_unit(p):
    """translation_unit : external_decl_list"""
    tu = A.TranslationUnit(items=[d for d in p[1] if d is not None])
    tu.line = 1
    p[0] = tu


def p_external_decl_list_one(p):
    """external_decl_list : external_decl"""
    p[0] = [p[1]]


def p_external_decl_list_many(p):
    """external_decl_list : external_decl_list external_decl"""
    p[0] = p[1] + [p[2]]


def p_external_decl(p):
    """external_decl : function_definition"""
    p[0] = p[1]


# ── Function definition ────────────────────────────────────────────────────

def p_function_definition(p):
    """function_definition : qualifier_list_opt type_spec declarator LPAREN param_list_opt RPAREN compound_stmt"""
    quals       = p[1]
    ret_type    = p[2]
    decl_name, ptr_levels, _arr_sizes = p[3]
    if ptr_levels:
        ret_type = A.Type(base=ret_type.base, is_pointer=True,
                          is_const=ret_type.is_const)
    fn = A.FunctionDef(
        qualifiers=quals,
        return_type=ret_type,
        name=decl_name,
        params=p[5] or [],
        body=p[7],
    )
    fn.line = p.lineno(4)
    p[0] = fn


# ── Qualifier list ──────────────────────────────────────────────────────────

def p_qualifier_list_opt_empty(p):
    """qualifier_list_opt : """
    p[0] = []


def p_qualifier_list_opt_some(p):
    """qualifier_list_opt : qualifier_list"""
    p[0] = p[1]


def p_qualifier_list_one(p):
    """qualifier_list : qualifier"""
    p[0] = [p[1]]


def p_qualifier_list_many(p):
    """qualifier_list : qualifier_list qualifier"""
    p[0] = p[1] + [p[2]]


def p_qualifier(p):
    """qualifier : KW_GLOBAL
                 | KW_DEVICE
                 | KW_HOST
                 | KW_SHARED
                 | KW_CONSTANT
                 | KW_EXTERN
                 | KW_CONST"""
    p[0] = p[1]


# ── Type specifier ──────────────────────────────────────────────────────────

def p_type_spec_void(p):
    """type_spec : KW_VOID"""
    p[0] = A.Type(base="void")


def p_type_spec_int(p):
    """type_spec : KW_INT"""
    p[0] = A.Type(base="int")


def p_type_spec_float(p):
    """type_spec : KW_FLOAT"""
    p[0] = A.Type(base="float")


def p_type_spec_bool(p):
    """type_spec : KW_BOOL"""
    p[0] = A.Type(base="bool")


def p_type_spec_unsigned(p):
    """type_spec : KW_UNSIGNED KW_INT
                 | KW_UNSIGNED"""
    p[0] = A.Type(base="unsigned int")


def p_type_spec_const(p):
    """type_spec : KW_CONST type_spec"""
    t = p[2]
    p[0] = A.Type(base=t.base, is_pointer=t.is_pointer,
                  is_const=True, is_restrict=t.is_restrict,
                  addr_space=t.addr_space)


# ── Declarator ──────────────────────────────────────────────────────────────
# Returns a tuple (name, pointer_levels, array_sizes_list_of_Expr_or_None)

def p_declarator_ident(p):
    """declarator : IDENT"""
    p[0] = (p[1], 0, [])


def p_declarator_pointer(p):
    """declarator : STAR declarator
                  | STAR KW_RESTRICT declarator
                  | STAR KW_CONST declarator"""
    if len(p) == 3:
        name, ptr, arr = p[2]
        p[0] = (name, ptr + 1, arr)
    else:
        name, ptr, arr = p[3]
        p[0] = (name, ptr + 1, arr)


def p_declarator_array(p):
    """declarator : declarator LBRACKET expression RBRACKET
                  | declarator LBRACKET RBRACKET"""
    name, ptr, arr = p[1]
    size = p[3] if len(p) == 5 else None
    p[0] = (name, ptr, arr + [size])


# ── Parameter list ──────────────────────────────────────────────────────────

def p_param_list_opt_empty(p):
    """param_list_opt : """
    p[0] = []


def p_param_list_opt_some(p):
    """param_list_opt : param_list"""
    p[0] = p[1]


def p_param_list_one(p):
    """param_list : param"""
    p[0] = [p[1]]


def p_param_list_many(p):
    """param_list : param_list COMMA param"""
    p[0] = p[1] + [p[3]]


def p_param(p):
    """param : qualifier_list_opt type_spec declarator"""
    quals = p[1]
    base_t = p[2]
    name, ptr, _arr = p[3]
    is_const = base_t.is_const or "const" in quals
    is_restrict = False
    # __restrict__ has been absorbed by declarator; check by re-scanning is
    # avoided by tracking it in a pointer-form declarator (see p_declarator_pointer).
    addr_space = None
    if "__shared__" in quals:
        addr_space = "shared"
    elif "__constant__" in quals:
        addr_space = "constant"
    t = A.Type(
        base=base_t.base,
        is_pointer=ptr > 0,
        is_const=is_const,
        is_restrict=is_restrict,
        addr_space=addr_space or ("global" if ptr > 0 else "local"),
    )
    param = A.Param(type=t, name=name)
    param.line = p.lineno(2)
    p[0] = param


# ── Compound statement ─────────────────────────────────────────────────────

def p_compound_stmt(p):
    """compound_stmt : LBRACE block_items_opt RBRACE"""
    cs = A.Compound(items=p[2])
    cs.line = p.lineno(1)
    p[0] = cs


def p_block_items_opt_empty(p):
    """block_items_opt : """
    p[0] = []


def p_block_items_opt_some(p):
    """block_items_opt : block_items"""
    p[0] = p[1]


def p_block_items_one(p):
    """block_items : block_item"""
    p[0] = [p[1]]


def p_block_items_many(p):
    """block_items : block_items block_item"""
    p[0] = p[1] + [p[2]]


def p_block_item_decl(p):
    """block_item : declaration"""
    p[0] = p[1]


def p_block_item_stmt(p):
    """block_item : statement"""
    p[0] = p[1]


# ── Declarations (variable) ────────────────────────────────────────────────

def p_declaration(p):
    """declaration : qualifier_list_opt type_spec init_declarator_list SEMI"""
    quals = p[1]
    base_t = p[2]
    addr_space = None
    extern_shared = False
    if "__shared__" in quals:
        addr_space = "shared"
        if "extern" in quals:
            extern_shared = True
    elif "__constant__" in quals:
        addr_space = "constant"
    is_const = base_t.is_const or "const" in quals
    decls = []
    for (name, ptr, arr_sizes), init in p[3]:
        t = A.Type(
            base=base_t.base,
            is_pointer=ptr > 0,
            is_const=is_const,
            addr_space=addr_space or ("global" if ptr > 0 else "local"),
        )
        # If we have array dimensions, take the first one as the size
        # (multi-dim handled crudely as flat).
        size_expr = arr_sizes[0] if arr_sizes else None
        d = A.VarDecl(
            type=t, name=name, init=init,
            array_size=size_expr, extern_shared=extern_shared,
        )
        d.line = p.lineno(2)
        decls.append(d)
    # If multiple declarators, return them inside a Compound for simplicity.
    if len(decls) == 1:
        p[0] = decls[0]
    else:
        p[0] = A.Compound(items=decls, line=p.lineno(2))


def p_init_declarator_list_one(p):
    """init_declarator_list : init_declarator"""
    p[0] = [p[1]]


def p_init_declarator_list_many(p):
    """init_declarator_list : init_declarator_list COMMA init_declarator"""
    p[0] = p[1] + [p[3]]


def p_init_declarator_no_init(p):
    """init_declarator : declarator"""
    p[0] = (p[1], None)


def p_init_declarator_with_init(p):
    """init_declarator : declarator ASSIGN assignment_expr"""
    p[0] = (p[1], p[3])


# ── Statements ──────────────────────────────────────────────────────────────

def p_statement(p):
    """statement : compound_stmt
                 | expression_stmt
                 | if_stmt
                 | for_stmt
                 | while_stmt
                 | do_while_stmt
                 | jump_stmt"""
    p[0] = p[1]


def p_expression_stmt_empty(p):
    """expression_stmt : SEMI"""
    es = A.ExprStmt(expr=None); es.line = p.lineno(1); p[0] = es


def p_expression_stmt_some(p):
    """expression_stmt : expression SEMI"""
    es = A.ExprStmt(expr=p[1]); es.line = p.lineno(2); p[0] = es


def p_if_stmt_no_else(p):
    """if_stmt : KW_IF LPAREN expression RPAREN statement %prec UMINUS"""
    n = A.If(cond=p[3], then=p[5], else_=None); n.line = p.lineno(1); p[0] = n


def p_if_stmt_with_else(p):
    """if_stmt : KW_IF LPAREN expression RPAREN statement KW_ELSE statement"""
    n = A.If(cond=p[3], then=p[5], else_=p[7]); n.line = p.lineno(1); p[0] = n


def p_for_stmt(p):
    """for_stmt : KW_FOR LPAREN for_init expression_opt SEMI expression_opt RPAREN statement"""
    n = A.For(init=p[3], cond=p[4], step=p[6], body=p[8])
    n.line = p.lineno(1); p[0] = n


def p_for_init_decl(p):
    """for_init : declaration"""
    p[0] = p[1]


def p_for_init_expr(p):
    """for_init : expression_opt SEMI"""
    p[0] = (A.ExprStmt(expr=p[1]) if p[1] is not None else None)


def p_expression_opt_empty(p):
    """expression_opt : """
    p[0] = None


def p_expression_opt_some(p):
    """expression_opt : expression"""
    p[0] = p[1]


def p_while_stmt(p):
    """while_stmt : KW_WHILE LPAREN expression RPAREN statement"""
    n = A.While(cond=p[3], body=p[5]); n.line = p.lineno(1); p[0] = n


def p_do_while_stmt(p):
    """do_while_stmt : KW_DO statement KW_WHILE LPAREN expression RPAREN SEMI"""
    n = A.DoWhile(body=p[2], cond=p[5]); n.line = p.lineno(1); p[0] = n


def p_jump_stmt_return_void(p):
    """jump_stmt : KW_RETURN SEMI"""
    n = A.Return(value=None); n.line = p.lineno(1); p[0] = n


def p_jump_stmt_return_value(p):
    """jump_stmt : KW_RETURN expression SEMI"""
    n = A.Return(value=p[2]); n.line = p.lineno(1); p[0] = n


def p_jump_stmt_break(p):
    """jump_stmt : KW_BREAK SEMI"""
    n = A.Break(); n.line = p.lineno(1); p[0] = n


def p_jump_stmt_continue(p):
    """jump_stmt : KW_CONTINUE SEMI"""
    n = A.Continue(); n.line = p.lineno(1); p[0] = n


# ── Expressions ─────────────────────────────────────────────────────────────

def p_expression(p):
    """expression : assignment_expr"""
    p[0] = p[1]


def p_assignment_expr_simple(p):
    """assignment_expr : conditional_expr"""
    p[0] = p[1]


def p_assignment_expr_assign(p):
    """assignment_expr : unary_expr ASSIGN assignment_expr
                       | unary_expr PLUSEQ assignment_expr
                       | unary_expr MINUSEQ assignment_expr
                       | unary_expr STAREQ assignment_expr
                       | unary_expr SLASHEQ assignment_expr
                       | unary_expr PERCENTEQ assignment_expr
                       | unary_expr ANDEQ assignment_expr
                       | unary_expr OREQ assignment_expr
                       | unary_expr XOREQ assignment_expr
                       | unary_expr LSHIFTEQ assignment_expr
                       | unary_expr RSHIFTEQ assignment_expr"""
    n = A.Assign(op=p[2], target=p[1], value=p[3])
    n.line = p.lineno(2); p[0] = n


def p_conditional_expr_simple(p):
    """conditional_expr : binary_expr"""
    p[0] = p[1]


def p_conditional_expr_ternary(p):
    """conditional_expr : binary_expr QUESTION expression COLON conditional_expr"""
    n = A.Cond(cond=p[1], then=p[3], else_=p[5])
    n.line = p.lineno(2); p[0] = n


def p_binary_expr(p):
    """binary_expr : binary_expr PLUS    binary_expr
                   | binary_expr MINUS   binary_expr
                   | binary_expr STAR    binary_expr
                   | binary_expr SLASH   binary_expr
                   | binary_expr PERCENT binary_expr
                   | binary_expr LSHIFT  binary_expr
                   | binary_expr RSHIFT  binary_expr
                   | binary_expr LT      binary_expr
                   | binary_expr GT      binary_expr
                   | binary_expr LE      binary_expr
                   | binary_expr GE      binary_expr
                   | binary_expr EQ      binary_expr
                   | binary_expr NEQ     binary_expr
                   | binary_expr AMP     binary_expr
                   | binary_expr PIPE    binary_expr
                   | binary_expr CARET   binary_expr
                   | binary_expr LAND    binary_expr
                   | binary_expr LOR     binary_expr
                   | unary_expr"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        n = A.BinOp(op=p[2], left=p[1], right=p[3])
        n.line = p.lineno(2); p[0] = n


def p_unary_expr_postfix(p):
    """unary_expr : postfix_expr"""
    p[0] = p[1]


def p_unary_expr_prefix(p):
    """unary_expr : INCREMENT unary_expr
                  | DECREMENT unary_expr
                  | LNOT      unary_expr
                  | TILDE     unary_expr
                  | MINUS     unary_expr   %prec UMINUS
                  | PLUS      unary_expr   %prec UPLUS
                  | STAR      unary_expr   %prec DEREF
                  | AMP       unary_expr   %prec ADDROF"""
    n = A.UnaryOp(op=p[1], operand=p[2], postfix=False)
    n.line = p.lineno(1); p[0] = n


def p_unary_expr_sizeof_expr(p):
    """unary_expr : KW_SIZEOF LPAREN type_spec RPAREN"""
    # Treat sizeof(T) as a constant integer literal of unknown value.
    n = A.IntLit(value=4)   # placeholder; real value set during type-check.
    n.line = p.lineno(1); p[0] = n


def p_postfix_expr_primary(p):
    """postfix_expr : primary_expr"""
    p[0] = p[1]


def p_postfix_expr_subscript(p):
    """postfix_expr : postfix_expr LBRACKET expression RBRACKET"""
    n = A.Subscript(array=p[1], index=p[3]); n.line = p.lineno(2); p[0] = n


def p_postfix_expr_call_empty(p):
    """postfix_expr : postfix_expr LPAREN RPAREN"""
    n = A.Call(fn=p[1], args=[]); n.line = p.lineno(2); p[0] = n


def p_postfix_expr_call_args(p):
    """postfix_expr : postfix_expr LPAREN argument_list RPAREN"""
    n = A.Call(fn=p[1], args=p[3]); n.line = p.lineno(2); p[0] = n


def p_postfix_expr_member(p):
    """postfix_expr : postfix_expr DOT IDENT"""
    n = A.Member(obj=p[1], field=p[3]); n.line = p.lineno(2); p[0] = n


def p_postfix_expr_inc(p):
    """postfix_expr : postfix_expr INCREMENT
                    | postfix_expr DECREMENT"""
    n = A.UnaryOp(op=p[2], operand=p[1], postfix=True)
    n.line = p.lineno(2); p[0] = n


def p_argument_list_one(p):
    """argument_list : assignment_expr"""
    p[0] = [p[1]]


def p_argument_list_many(p):
    """argument_list : argument_list COMMA assignment_expr"""
    p[0] = p[1] + [p[3]]


def p_primary_expr_ident(p):
    """primary_expr : IDENT"""
    n = A.Ident(name=p[1]); n.line = p.lineno(1); p[0] = n


def p_primary_expr_int(p):
    """primary_expr : INT"""
    n = A.IntLit(value=p[1]); n.line = p.lineno(1); p[0] = n


def p_primary_expr_float(p):
    """primary_expr : FLOAT_LIT"""
    n = A.FloatLit(value=p[1]); n.line = p.lineno(1); p[0] = n


def p_primary_expr_bool_true(p):
    """primary_expr : TRUE"""
    n = A.BoolLit(value=True); n.line = p.lineno(1); p[0] = n


def p_primary_expr_bool_false(p):
    """primary_expr : FALSE"""
    n = A.BoolLit(value=False); n.line = p.lineno(1); p[0] = n


def p_primary_expr_paren(p):
    """primary_expr : LPAREN expression RPAREN"""
    p[0] = p[2]


def p_primary_expr_cast(p):
    """primary_expr : LPAREN type_spec RPAREN unary_expr %prec CAST"""
    n = A.Cast(type=p[2], operand=p[4]); n.line = p.lineno(1); p[0] = n


# ── Error recovery ─────────────────────────────────────────────────────────

class ParseError(Exception):
    """Raised when the parser hits an unrecoverable syntax error."""


def p_error(p):
    if p is None:
        raise ParseError("Unexpected end of input")
    raise ParseError(
        f"Syntax error at line {p.lineno}: unexpected {p.type} {p.value!r}"
    )


# ── Public API ─────────────────────────────────────────────────────────────

_PARSER = None


def make_parser(**kwargs):
    """
    Build (or reuse) a YACC parser instance.  PLY caches the parser tables
    on disk by default; we suppress that to keep the project clean.
    """
    global _PARSER
    if _PARSER is None:
        _PARSER = yacc.yacc(
            module=__import__(__name__, fromlist=["_"]),
            debug=False,
            write_tables=False,
            **kwargs,
        )
    return _PARSER


def parse(source: str) -> A.TranslationUnit:
    """Parse a CUDA source string into a `TranslationUnit` AST."""
    lexer = make_lexer()
    parser = make_parser()
    return parser.parse(source, lexer=lexer, tracking=True)


if __name__ == "__main__":
    import sys
    from .preprocessor import preprocess, extract_device_functions

    if len(sys.argv) < 2:
        print("usage: python -m cdc.parser <file.cu>", file=sys.stderr)
        sys.exit(1)

    src = open(sys.argv[1], encoding="utf-8").read()
    cleaned = preprocess(src)

    # Pull each __global__/__device__ kernel out and parse separately —
    # avoids host-only C++ constructs (dim3, cudaEvent_t, <<<...>>>).
    funcs = extract_device_functions(src)
    print(f"Found {len(funcs)} device functions")
    print()
    for snippet, lineno in funcs:
        print(f"== Parsing kernel at line {lineno} ==")
        tu = parse(snippet)
        print(A.pretty(tu))
        print()

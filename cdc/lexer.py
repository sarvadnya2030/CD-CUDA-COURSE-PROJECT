"""
lexer.py — PLY (Python Lex-Yacc) tokeniser for a CUDA C subset.

This is the lexical-analysis phase of the compiler frontend (CD Unit II).
It corresponds to a LEX/FLEX specification: token regular expressions,
reserved-word table, operator priorities, and line-number tracking.

Tokens recognised
-----------------
* Keywords:      if else for while do return break continue
                 void int float bool unsigned const sizeof
                 true false extern
* CUDA quals:    __global__ __device__ __host__ __shared__
                 __constant__ __restrict__
* Builtin names: threadIdx blockIdx blockDim gridDim warpSize
                 __syncthreads (lex-level recognition; treated as IDENT
                 by the parser but with a flag for type checking)
* Identifiers:   [A-Za-z_][A-Za-z0-9_]*
* Literals:      INT  (decimal, hex 0x.., octal 0..)
                 FLOAT (1.0, 1.0e-5, 1.0f)
                 BOOL  (true / false)
* Operators:     +  -  *  /  %  ++  --  =  +=  -=  *=  /=  %=
                 ==  !=  <  >  <=  >=  &&  ||  !  ~
                 &  |  ^  <<  >>  &=  |=  ^=  <<=  >>=
                 ?  :
* Punctuation:   (  )  {  }  [  ]  ,  ;  .

Output
------
Each token has the standard PLY attributes (`type`, `value`, `lineno`,
`lexpos`).  A helper `column(token, source)` reports the 1-based column
within the original source for diagnostics.

Maps to syllabus
----------------
* Course Unit II  — Lexical Analysis (LEX/FLEX, regular expressions)
* Lab Practicals  — 1, 2, 3, 4, 5 (lex regex programs)
"""

from __future__ import annotations

import ply.lex as lex


# ── Reserved words (keyword → token type) ───────────────────────────────────

reserved = {
    # Storage / qualifier
    "__global__":     "KW_GLOBAL",
    "__device__":     "KW_DEVICE",
    "__host__":       "KW_HOST",
    "__shared__":     "KW_SHARED",
    "__constant__":   "KW_CONSTANT",
    "__restrict__":   "KW_RESTRICT",
    "extern":         "KW_EXTERN",
    "const":          "KW_CONST",

    # Types
    "void":           "KW_VOID",
    "int":            "KW_INT",
    "float":          "KW_FLOAT",
    "bool":           "KW_BOOL",
    "unsigned":       "KW_UNSIGNED",

    # Statements
    "if":             "KW_IF",
    "else":           "KW_ELSE",
    "for":            "KW_FOR",
    "while":          "KW_WHILE",
    "do":             "KW_DO",
    "return":         "KW_RETURN",
    "break":          "KW_BREAK",
    "continue":       "KW_CONTINUE",
    "sizeof":         "KW_SIZEOF",

    # Boolean literals
    "true":           "TRUE",
    "false":          "FALSE",
}


# ── Token list (required by PLY) ────────────────────────────────────────────

tokens = [
    # literals
    "INT", "FLOAT_LIT", "IDENT",

    # arithmetic
    "PLUS", "MINUS", "STAR", "SLASH", "PERCENT",
    "INCREMENT", "DECREMENT",

    # assignment
    "ASSIGN",
    "PLUSEQ", "MINUSEQ", "STAREQ", "SLASHEQ", "PERCENTEQ",
    "ANDEQ", "OREQ", "XOREQ", "LSHIFTEQ", "RSHIFTEQ",

    # comparison
    "EQ", "NEQ", "LT", "GT", "LE", "GE",

    # logical
    "LAND", "LOR", "LNOT",

    # bitwise
    "AMP", "PIPE", "CARET", "TILDE", "LSHIFT", "RSHIFT",

    # ternary
    "QUESTION", "COLON",

    # punctuation
    "LPAREN", "RPAREN", "LBRACE", "RBRACE", "LBRACKET", "RBRACKET",
    "COMMA", "SEMI", "DOT",
] + list(set(reserved.values()))


# ── Simple operator rules (regex strings) ───────────────────────────────────

t_PLUSEQ      = r"\+="
t_MINUSEQ     = r"-="
t_STAREQ      = r"\*="
t_SLASHEQ     = r"/="
t_PERCENTEQ   = r"%="
t_LSHIFTEQ    = r"<<="
t_RSHIFTEQ    = r">>="
t_ANDEQ       = r"&="
t_OREQ        = r"\|="
t_XOREQ       = r"\^="

t_INCREMENT   = r"\+\+"
t_DECREMENT   = r"--"

t_EQ          = r"=="
t_NEQ         = r"!="
t_LE          = r"<="
t_GE          = r">="
t_LAND        = r"&&"
t_LOR         = r"\|\|"
t_LSHIFT      = r"<<"
t_RSHIFT      = r">>"

t_PLUS        = r"\+"
t_MINUS       = r"-"
t_STAR        = r"\*"
t_SLASH       = r"/"
t_PERCENT     = r"%"
t_ASSIGN      = r"="
t_LT          = r"<"
t_GT          = r">"
t_LNOT        = r"!"
t_TILDE       = r"~"
t_AMP         = r"&"
t_PIPE        = r"\|"
t_CARET       = r"\^"
t_QUESTION    = r"\?"
t_COLON       = r":"
t_LPAREN      = r"\("
t_RPAREN      = r"\)"
t_LBRACE      = r"\{"
t_RBRACE      = r"\}"
t_LBRACKET    = r"\["
t_RBRACKET    = r"\]"
t_COMMA       = r","
t_SEMI        = r";"
t_DOT         = r"\."


# Whitespace ignored; newlines counted.
t_ignore = " \t\r"


def t_NEWLINE(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


# ── Literals ────────────────────────────────────────────────────────────────

# Float must be tried before INT so e.g. "0.0f" doesn't lex as "0" then ".0f".
def t_FLOAT_LIT(t):
    r"((\d+\.\d*|\.\d+)([eE][+-]?\d+)?[fFlL]?|\d+[eE][+-]?\d+[fFlL]?|\d+[fF])"
    s = t.value.rstrip("fFlL")
    t.value = float(s)
    return t


def t_INT(t):
    r"(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9]\d*)[uUlL]*"
    s = t.value.rstrip("uUlL")
    if s.startswith(("0x", "0X")):
        t.value = int(s, 16)
    elif s.startswith("0") and len(s) > 1:
        t.value = int(s, 8)
    else:
        t.value = int(s)
    return t


def t_IDENT(t):
    r"[A-Za-z_][A-Za-z_0-9]*"
    t.type = reserved.get(t.value, "IDENT")
    return t


# ── Error reporting ─────────────────────────────────────────────────────────

class LexError(Exception):
    pass


def t_error(t):
    raise LexError(f"Illegal character {t.value[0]!r} at line {t.lineno}")


# ── Public helpers ──────────────────────────────────────────────────────────

def make_lexer(**kwargs):
    """Build a fresh PLY lexer instance."""
    return lex.lex(module=__import__(__name__, fromlist=["_"]), **kwargs)


def column(token, source: str) -> int:
    """Compute the 1-based column position of `token` within `source`."""
    last_nl = source.rfind("\n", 0, token.lexpos)
    return token.lexpos - last_nl


def tokenize(source: str) -> list:
    """
    Tokenise an entire string and return a list of `LexToken` objects.

    Useful for unit tests and the `python -m cdc <file> --tokens` CLI mode.
    """
    lx = make_lexer()
    lx.input(source)
    out = []
    for tok in lx:
        out.append(tok)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m cdc.lexer <file.cu>", file=sys.stderr)
        sys.exit(1)
    src = open(sys.argv[1], encoding="utf-8").read()
    from cdc.preprocessor import preprocess
    cleaned = preprocess(src)
    for t in tokenize(cleaned):
        print(f"{t.lineno:>4}: {t.type:<14} {t.value!r}")

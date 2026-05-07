"""
preprocessor.py — minimal preprocessor for CUDA C source.

The PLY lexer + parser only need to handle a CUDA subset (kernels, device
functions, host launchers).  Real preprocessing is delegated to a quick
pre-pass that:

  1. Strips line comments  // ...
  2. Strips block comments /* ... */
  3. Drops `#include` and `#pragma` lines outright
  4. Handles multi-line `#define` macros by collapsing line continuations,
     then dropping them.  We do NOT expand macros — we just remove their
     definitions.  Macro *uses* in source are parsed as ordinary identifier
     calls, which is fine for our purposes (the autotuner does not rely on
     macro expansion semantics; it only needs the AST shape of __global__
     kernels and their parameters).

This keeps the grammar small and the lexer table compact.

Line numbers are preserved across the preprocessor (we replace stripped
text with blank lines) so that diagnostics still point to the right place.
"""

from __future__ import annotations

import re

# Patterns
_LINE_COMMENT  = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_PP_LINE       = re.compile(r"^[ \t]*#[^\n]*$", re.MULTILINE)

# Multi-line continuation: a backslash at end-of-line continues the next line.
_BACKSLASH_NL  = re.compile(r"\\\s*\n")


def preprocess(source: str) -> str:
    """
    Strip comments and preprocessor directives from CUDA source.

    Whitespace and newlines are preserved so that downstream line numbers
    match the original file.

    Parameters
    ----------
    source : str
        Raw CUDA C source.

    Returns
    -------
    str
        Source with comments and `#...` lines blanked out (newlines kept).
    """
    # Step 1 — collapse line continuations inside #define etc.
    src = _BACKSLASH_NL.sub(" ", source)

    # Step 2 — strip block comments, replacing with blanks (preserve newlines).
    def _blank_block(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")
    src = _BLOCK_COMMENT.sub(_blank_block, src)

    # Step 3 — strip line comments (does not span newlines).
    src = _LINE_COMMENT.sub("", src)

    # Step 4 — drop preprocessor directives (whole line).
    src = _PP_LINE.sub("", src)

    return src


def extract_device_functions(source: str) -> list[tuple[str, int]]:
    """
    Locate `__global__` and `__device__` function definitions in the source.

    Uses brace matching on the cleaned source.  Returns a list of
    `(snippet, start_line)` pairs where `snippet` is the full text of one
    device function (qualifiers + signature + body) and `start_line` is the
    1-based line number where it begins.

    This is a convenience for the frontend driver — the parser itself can
    handle a whole translation unit, but kernels are often embedded in
    files that also contain host launchers and macros that the simplified
    grammar doesn't recognise.  Extracting just the kernels gives a clean
    input.
    """
    cleaned = preprocess(source)
    n = len(cleaned)
    results: list[tuple[str, int]] = []

    # Match qualifier at start of a function: __global__ or __device__
    qualifier_re = re.compile(r"\b(__global__|__device__)\b")

    pos = 0
    while pos < n:
        m = qualifier_re.search(cleaned, pos)
        if not m:
            break
        start = m.start()
        # Walk forward to find the opening brace of the body.
        i = m.end()
        depth_paren = 0
        while i < n:
            c = cleaned[i]
            if c == "(":
                depth_paren += 1
            elif c == ")":
                depth_paren -= 1
            elif c == "{" and depth_paren == 0:
                break
            i += 1
        if i >= n:
            break
        # Brace-match to find body end.
        depth = 0
        body_start = i
        while i < n:
            c = cleaned[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        snippet = cleaned[start:i]
        start_line = cleaned.count("\n", 0, start) + 1
        results.append((snippet, start_line))
        pos = i

    return results

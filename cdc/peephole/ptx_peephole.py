"""
ptx_peephole.py - peephole optimiser that runs on NVIDIA PTX text.

PTX is NVIDIA's intermediate representation between CUDA C++ and SASS;
it is the canonical IR a CD examiner can see directly (`nvcc -ptx`).
This module implements the textbook *peephole optimiser* (CD Unit V) on
that real production IR.

What we do
----------
We tokenise PTX line by line, walk a small sliding window, and apply
pattern-matching rewrites.  Every match is logged so we can show the
examiner exactly which patterns fired and how often.

Implemented patterns
--------------------
1. **mul + add => fma**

       mul.f32   t1, a, b;
       add.f32   r,  t1, c;          (or)   add.f32   r, c, t1;
       =>
       fma.f32   r,  a, b, c;

   Eliminates one instruction and one temporary.  Equivalent to LLVM's
   ``InstCombine`` mul-add fusion but operating on raw PTX.

2. **redundant mov.f32 (self-copy)**

       mov.f32   r, r;
       =>
       (deleted)

   Defensive pattern - sometimes shows up after dead-code elimination
   on the front-end side.

3. **double-mov elimination**

       mov.f32   t,  a;
       mov.f32   r,  t;       (and `t` not used afterwards in window)
       =>
       mov.f32   r,  a;

   One-instruction copy propagation in the small window.

4. **add zero / multiply by one**

       add.f32   r, a, 0f00000000;     -> mov.f32 r, a;
       mul.f32   r, a, 0f3F800000;     -> mov.f32 r, a;

   Standard algebraic identities; `nvcc` usually emits these but not
   always (e.g. when the constant comes from a macro the strength-reducer
   missed).

5. **ld.global; ld.global with immediate reuse => single load**

       ld.global.f32  t1, [p];
       ld.global.f32  t2, [p];        (no store of *p between them)
       =>
       ld.global.f32  t1, [p];
       mov.f32        t2, t1;

   Coalesces a redundant load when the pointer is unchanged in the window.

These five families are enough to cover the textbook peephole-optimiser
topic without overstepping into nvcc's domain.  Each pattern is an
explicit `PeepholePattern` instance so adding more is trivial.

Usage
-----

    from cdc.peephole import optimise_ptx_file
    new_ptx, stats = optimise_ptx_file(Path("kernel.ptx"))
    Path("kernel_peep.ptx").write_text(new_ptx)
    print(stats)

CLI: see ``python -m cdc.peephole.ptx_peephole <file.ptx>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ── A single instruction (parsed line) ─────────────────────────────────────

_INSN_RE = re.compile(
    r"^\s*(?P<op>[a-zA-Z_][a-zA-Z0-9_.]*)"
    r"\s+(?P<args>[^;]+);"
    r"(?P<rest>.*)$"
)
_LABEL_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_$]*\s*:\s*$")
_DIRECTIVE_RE = re.compile(r"^\s*\.")


@dataclass
class PtxLine:
    raw: str
    op: Optional[str] = None
    args: Optional[List[str]] = None
    is_directive: bool = False
    is_label: bool = False
    deleted: bool = False              # marked True by a successful rewrite

    def render(self) -> str:
        return self.raw


def _parse_line(raw: str) -> PtxLine:
    if _LABEL_RE.match(raw):
        return PtxLine(raw=raw, is_label=True)
    if _DIRECTIVE_RE.match(raw) or raw.strip().startswith("//"):
        return PtxLine(raw=raw, is_directive=True)
    m = _INSN_RE.match(raw)
    if not m:
        return PtxLine(raw=raw, is_directive=True)
    args = [a.strip() for a in m.group("args").split(",")]
    return PtxLine(raw=raw, op=m.group("op"), args=args)


# ── Patterns ───────────────────────────────────────────────────────────────

PatternFn = Callable[["PtxPeepholeOptimizer", List[PtxLine], int], int]


@dataclass
class PeepholePattern:
    """One rewrite rule.

    `match` consumes a window of `window_size` instructions starting at
    index `i`.  If the pattern fires, it must mutate the lines list
    (e.g. set `deleted=True`, edit `raw`) and return the *number of
    instructions consumed* (so the caller can advance).  If nothing
    happens it returns 0.
    """
    name: str
    description: str
    window_size: int
    match: PatternFn


# ── Helpers ────────────────────────────────────────────────────────────────

_NEXT_INSN_RE = re.compile(r"^\s+")


def _is_zero_f32(s: str) -> bool:
    """True if PTX immediate `s` is +0.0 fp32."""
    s = s.strip()
    return s in ("0f00000000", "0F00000000", "0f80000000", "0F80000000")


def _is_one_f32(s: str) -> bool:
    """True if PTX immediate `s` is 1.0 fp32 (0x3F800000)."""
    return s.strip() in ("0f3F800000", "0F3F800000", "0f3f800000")


def _strip(s: str) -> str:
    return s.strip()


# ── Pattern: mul + add  =>  fma ────────────────────────────────────────────

def _pat_mul_add_fma(opt: "PtxPeepholeOptimizer", lines: List[PtxLine], i: int) -> int:
    if i + 1 >= len(lines):
        return 0
    a, b = lines[i], lines[i + 1]
    if a.deleted or b.deleted or a.op is None or b.op is None:
        return 0
    if not a.op.startswith("mul.f"):
        return 0
    if not b.op.startswith("add.f"):
        return 0
    # mul: dst, x, y          add: r, dst, c   (or)  add: r, c, dst
    if len(a.args) != 3 or len(b.args) != 3:
        return 0
    dst_mul = _strip(a.args[0])
    x, y = _strip(a.args[1]), _strip(a.args[2])
    add_dst, add_a, add_b = (_strip(b.args[0]), _strip(b.args[1]), _strip(b.args[2]))
    if dst_mul not in (add_a, add_b):
        return 0
    c = add_b if add_a == dst_mul else add_a
    fma_op = a.op.replace("mul", "fma")        # mul.f32 -> fma.f32
    new = f"\tfma{fma_op[3:]:>0} \t{add_dst}, {x}, {y}, {c};"
    # Replace the add line with FMA, mark mul deleted.
    b.raw = new
    b.op = fma_op
    b.args = [add_dst, x, y, c]
    a.deleted = True
    opt.fired["mul_add_fma"] = opt.fired.get("mul_add_fma", 0) + 1
    return 2


# ── Pattern: self-copy mov ─────────────────────────────────────────────────

def _pat_self_mov(opt, lines, i):
    a = lines[i]
    if a.deleted or a.op is None:
        return 0
    if not a.op.startswith("mov."):
        return 0
    if len(a.args) != 2:
        return 0
    if _strip(a.args[0]) == _strip(a.args[1]):
        a.deleted = True
        opt.fired["self_mov"] = opt.fired.get("self_mov", 0) + 1
        return 1
    return 0


# ── Pattern: double-mov => single mov ──────────────────────────────────────

def _pat_double_mov(opt, lines, i):
    if i + 1 >= len(lines):
        return 0
    a, b = lines[i], lines[i + 1]
    if a.deleted or b.deleted or a.op is None or b.op is None:
        return 0
    if not a.op.startswith("mov.") or not b.op.startswith("mov."):
        return 0
    if len(a.args) != 2 or len(b.args) != 2:
        return 0
    t1 = _strip(a.args[0])
    src = _strip(a.args[1])
    if _strip(b.args[1]) != t1:
        return 0
    # Double-mov: a writes t1 from src; b writes b.dst from t1.
    # Safe only if t1 is not used elsewhere in the window beyond `b`.
    later_use = False
    for j in range(i + 2, min(i + 6, len(lines))):
        if lines[j].deleted:
            continue
        if lines[j].args is not None and t1 in (_strip(x) for x in lines[j].args):
            later_use = True
            break
    if later_use:
        return 0
    new_dst = _strip(b.args[0])
    b.raw = f"\t{b.op} \t{new_dst}, {src};"
    b.args = [new_dst, src]
    a.deleted = True
    opt.fired["double_mov"] = opt.fired.get("double_mov", 0) + 1
    return 2


# ── Pattern: add zero / mul one => mov ─────────────────────────────────────

def _pat_algebraic_identity(opt, lines, i):
    a = lines[i]
    if a.deleted or a.op is None:
        return 0
    if not (a.op.startswith("add.f") or a.op.startswith("mul.f")):
        return 0
    if len(a.args) != 3:
        return 0
    dst, x, y = _strip(a.args[0]), _strip(a.args[1]), _strip(a.args[2])
    is_add = a.op.startswith("add.f")
    is_mul = a.op.startswith("mul.f")
    src = None
    if is_add:
        if _is_zero_f32(x):
            src = y
        elif _is_zero_f32(y):
            src = x
    elif is_mul:
        if _is_one_f32(x):
            src = y
        elif _is_one_f32(y):
            src = x
    if src is None:
        return 0
    a.raw = f"\tmov{a.op[3:]} \t{dst}, {src};"
    a.op = "mov" + a.op[3:]
    a.args = [dst, src]
    opt.fired["alg_identity"] = opt.fired.get("alg_identity", 0) + 1
    return 1


# ── Pattern: redundant ld.global => mov ────────────────────────────────────

def _pat_redundant_load(opt, lines, i):
    if i + 1 >= len(lines):
        return 0
    a, b = lines[i], lines[i + 1]
    if a.deleted or b.deleted or a.op is None or b.op is None:
        return 0
    if not a.op.startswith("ld.") or not b.op.startswith("ld."):
        return 0
    if a.op != b.op:
        return 0
    if len(a.args) != 2 or len(b.args) != 2:
        return 0
    if _strip(a.args[1]) != _strip(b.args[1]):
        return 0
    # Make sure no store to that address happens between them.  Simple
    # version: no st.* or call between.  In our 2-window we have no
    # intermediate line, so it's automatically safe.
    new_dst = _strip(b.args[0])
    src = _strip(a.args[0])
    # Drop the state-space modifier from the load opcode (mov has no `.global`).
    # ld.global.f32 -> mov.f32, ld.shared.u32 -> mov.u32, ld.f32 -> mov.f32, ...
    parts = a.op.split(".")
    type_part = parts[-1]                              # f32 / u32 / b64 / ...
    b.raw = f"\tmov.{type_part} \t{new_dst}, {src};"
    b.op = f"mov.{type_part}"
    b.args = [new_dst, src]
    opt.fired["redundant_ld"] = opt.fired.get("redundant_ld", 0) + 1
    return 2


# ── Optimiser ──────────────────────────────────────────────────────────────

class PtxPeepholeOptimizer:
    """Apply a list of `PeepholePattern`s repeatedly until quiescent."""

    DEFAULT_PATTERNS: List[PeepholePattern] = [
        PeepholePattern("mul_add_fma",   "mul + add => fma",                 2, _pat_mul_add_fma),
        PeepholePattern("self_mov",      "mov.f32 r, r => deleted",          1, _pat_self_mov),
        PeepholePattern("double_mov",    "mov t, a; mov r, t => mov r, a",   2, _pat_double_mov),
        PeepholePattern("alg_identity",  "add 0 / mul 1 => mov",             1, _pat_algebraic_identity),
        PeepholePattern("redundant_ld",  "ld dup => mov",                    2, _pat_redundant_load),
    ]

    def __init__(self, patterns: Optional[List[PeepholePattern]] = None,
                 max_passes: int = 8):
        self.patterns = patterns or list(self.DEFAULT_PATTERNS)
        self.max_passes = max_passes
        self.fired: Dict[str, int] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def optimise_text(self, ptx_text: str) -> Tuple[str, Dict[str, int]]:
        lines = [_parse_line(l) for l in ptx_text.splitlines()]
        for _ in range(self.max_passes):
            before_fired = sum(self.fired.values())
            i = 0
            while i < len(lines):
                if lines[i].deleted or lines[i].is_directive or lines[i].is_label \
                   or lines[i].op is None:
                    i += 1
                    continue
                advanced = False
                for pat in self.patterns:
                    n = pat.match(self, lines, i)
                    if n > 0:
                        i += n
                        advanced = True
                        break
                if not advanced:
                    i += 1
            if sum(self.fired.values()) == before_fired:
                break
        out = "\n".join(l.render() for l in lines if not l.deleted)
        return out, dict(self.fired)


# ── Façade & CLI ───────────────────────────────────────────────────────────

def optimise_ptx_file(src: Path, dst: Optional[Path] = None) -> Tuple[str, Dict[str, int]]:
    """Optimise a PTX file; return ``(new_ptx, stats)``.  If `dst` is set,
    also write the result there."""
    opt = PtxPeepholeOptimizer()
    text = Path(src).read_text(encoding="utf-8")
    new, stats = opt.optimise_text(text)
    if dst is not None:
        Path(dst).write_text(new, encoding="utf-8")
    return new, stats


def summarise_passes(stats: Dict[str, int]) -> str:
    lines = ["PTX peephole pass summary:"]
    if not stats:
        lines.append("  (no patterns fired)")
        return "\n".join(lines)
    width = max(len(k) for k in stats) + 1
    for name in sorted(stats):
        lines.append(f"  {name:<{width}}  {stats[name]:>3} fired")
    lines.append(f"  total fired         : {sum(stats.values())}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="cdc.peephole.ptx_peephole",
                                description="Run peephole optimisations on a PTX file")
    p.add_argument("file", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Write the optimised PTX here (default: <file>.peep.ptx)")
    args = p.parse_args()

    out_path = args.output or args.file.with_suffix(".peep.ptx")
    new_text, stats = optimise_ptx_file(args.file, out_path)
    print(summarise_passes(stats))
    print(f"wrote {out_path}")

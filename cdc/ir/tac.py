"""
tac.py — Three-Address Code generator (quadruples).

Lowers the AST produced by `cdc.parser` into a linear list of quadruples,
the canonical textbook IR (Aho/Ullman §6.2).

Quadruple form
--------------
Every instruction has up to three named operands plus an opcode:

        result := arg1  op  arg2

We represent a quadruple as a dataclass with explicit `(op, arg1, arg2,
result)` fields, plus a source-line back-pointer for diagnostics and an
optional label name for jump/label instructions.

Opcode set
----------
Arithmetic       :  +  -  *  /  %                       (binary)
Bitwise          :  &  |  ^  <<  >>                     (binary)
Comparison       :  ==  !=  <  >  <=  >=                (binary, result is int 0/1)
Unary            :  uminus  uplus  unot  bnot           (unary)
Memory           :  []=  =[]  =*  *=                    (load/store)
Member           :  member                              (obj.field)
Cast             :  cast                                (cast x → T)
Assign / copy    :  =                                   (result := arg1)
Control          :  label  goto  iffalse                (branch on falsy)
                    ifgoto                              (branch on truthy)
Functions        :  param  call  return                 (caller-side ABI)
Phi (placeholder):  phi                                 (currently unused;
                                                         reserved for SSA)

Names
-----
Every temporary is a fresh string `t0, t1, t2, …` (allocated by `Namer`).
Every label is `Lk` for k = 0, 1, ….  Source-language identifiers are
preserved verbatim so the listing reads naturally.

Booleans / short-circuit evaluation
-----------------------------------
`&&` and `||` are lowered with control-flow short-circuits, which matches
the textbook scheme and keeps the DAG-of-basic-block step rich (the DAG
sees ordinary comparison/branch operations rather than a synthetic
boolean operator).

Maps to syllabus
----------------
* Course Unit IV — Intermediate Code Generation, IR formats (quadruples)
* Lab Practical  — 11 (TAC and quadruples)
* Tutorial       — 12 (quadruples examples)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .. import ast_nodes as A


# ── Quadruple ──────────────────────────────────────────────────────────────

ArgT = Union[str, int, float, bool, None]


@dataclass
class Quad:
    """One three-address instruction.

    Attributes
    ----------
    op     : opcode string (see module docstring).
    arg1   : first source operand (name string or constant).
    arg2   : second source operand.
    result : destination name (or jump target for control ops).
    line   : original source line for diagnostics.
    """
    op: str
    arg1: ArgT = None
    arg2: ArgT = None
    result: ArgT = None
    line: int = 0

    # Display / pretty-printing -------------------------------------------------

    def is_label(self) -> bool:
        return self.op == "label"

    def is_branch(self) -> bool:
        return self.op in ("goto", "ifgoto", "iffalse")

    def is_terminator(self) -> bool:
        return self.op in ("goto", "ifgoto", "iffalse", "return")

    def __str__(self) -> str:
        op, a, b, r = self.op, self.arg1, self.arg2, self.result

        # Labels render flush-left.
        if op == "label":
            return f"{r}:"
        if op == "goto":
            return f"    goto {r}"
        if op == "ifgoto":
            return f"    if {a} goto {r}"
        if op == "iffalse":
            return f"    if not {a} goto {r}"
        if op == "return":
            if a is None:
                return "    return"
            return f"    return {a}"
        if op == "param":
            return f"    param {a}"
        if op == "call":
            return f"    {r} = call {a}, {b}"
        if op == "=":
            return f"    {r} = {a}"
        if op == "uminus":
            return f"    {r} = -{a}"
        if op == "uplus":
            return f"    {r} = +{a}"
        if op == "unot":
            return f"    {r} = !{a}"
        if op == "bnot":
            return f"    {r} = ~{a}"
        if op == "[]=":
            return f"    {r} = {a}[{b}]"
        if op == "=[]":
            return f"    {a}[{b}] = {r}"
        if op == "=*":
            return f"    {r} = *{a}"
        if op == "*=":
            return f"    *{a} = {r}"
        if op == "member":
            return f"    {r} = {a}.{b}"
        if op == "cast":
            return f"    {r} = ({b}) {a}"
        if op == "phi":
            return f"    {r} = phi({a}, {b})"
        # Default: binary op.
        return f"    {r} = {a} {op} {b}"


@dataclass
class TacProgram:
    """A flat list of quadruples plus metadata about the source kernel."""
    name: str = ""
    params: List[str] = field(default_factory=list)
    quads: List[Quad] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"function {self.name}({', '.join(self.params)}):"
        return head + "\n" + "\n".join(str(q) for q in self.quads)

    def numbered(self) -> str:
        """Return quads with sequential indices (helpful in basic-block dumps)."""
        head = f"function {self.name}({', '.join(self.params)}):"
        body = []
        for i, q in enumerate(self.quads):
            body.append(f"  {i:>3}: {q}")
        return head + "\n" + "\n".join(body)


# ── Helpers: temporary and label allocators ────────────────────────────────

class Namer:
    """Mints fresh temporary names and labels."""

    def __init__(self, temp_prefix: str = "t", label_prefix: str = "L"):
        self._t = 0
        self._l = 0
        self.tprefix = temp_prefix
        self.lprefix = label_prefix

    def temp(self) -> str:
        n = f"{self.tprefix}{self._t}"
        self._t += 1
        return n

    def label(self) -> str:
        n = f"{self.lprefix}{self._l}"
        self._l += 1
        return n


# ── TAC emitter ────────────────────────────────────────────────────────────

class TacEmitter:
    """
    Walks a `FunctionDef` AST and emits quadruples into `prog.quads`.

    The emitter follows the standard Aho/Ullman scheme:

      * `gen(stmt)`  appends quads for a statement.
      * `gen_expr(expr)` returns the *name* (string) holding the value;
        if the expression is non-trivial it also appends quads.
      * Boolean/branching expressions use `gen_bool(expr, true_lbl, false_lbl)`
        to emit short-circuit jumps directly.

    Loop control statements (`break`, `continue`) consult a small stack of
    `(break_label, continue_label)` records so they can target the
    correct enclosing loop.
    """

    def __init__(self, namer: Optional[Namer] = None):
        self.namer = namer or Namer()
        self.prog = TacProgram()
        self._loops: List[tuple[str, str]] = []   # (break, continue)

    # ── Public driver ────────────────────────────────────────────────────

    def emit_function(self, fn: A.FunctionDef) -> TacProgram:
        self.prog = TacProgram(
            name=fn.name,
            params=[p.name for p in fn.params],
        )
        for p in fn.params:
            # Record params as defs at line 0 — useful for DFA "live in".
            self._emit("param", arg1=p.name, line=p.line)
        if fn.body is not None:
            self.gen_stmt(fn.body)
        return self.prog

    # ── Internal helpers ─────────────────────────────────────────────────

    def _emit(self, op: str, arg1=None, arg2=None, result=None, line: int = 0) -> Quad:
        q = Quad(op=op, arg1=arg1, arg2=arg2, result=result, line=line)
        self.prog.quads.append(q)
        return q

    def _label(self, name: str, line: int = 0) -> None:
        self._emit("label", result=name, line=line)

    def _goto(self, name: str, line: int = 0) -> None:
        self._emit("goto", result=name, line=line)

    def _ifgoto(self, cond_name: str, target: str, line: int = 0) -> None:
        self._emit("ifgoto", arg1=cond_name, result=target, line=line)

    def _iffalse(self, cond_name: str, target: str, line: int = 0) -> None:
        self._emit("iffalse", arg1=cond_name, result=target, line=line)

    # ── Statement lowering ───────────────────────────────────────────────

    def gen_stmt(self, s) -> None:
        if isinstance(s, A.Compound):
            for it in s.items:
                self.gen_stmt(it)
            return

        if isinstance(s, A.VarDecl):
            self._gen_vardecl(s)
            return

        if isinstance(s, A.ExprStmt):
            if s.expr is not None:
                self.gen_expr(s.expr)
            return

        if isinstance(s, A.If):
            self._gen_if(s)
            return

        if isinstance(s, A.For):
            self._gen_for(s)
            return

        if isinstance(s, A.While):
            self._gen_while(s)
            return

        if isinstance(s, A.DoWhile):
            self._gen_dowhile(s)
            return

        if isinstance(s, A.Return):
            if s.value is None:
                self._emit("return", line=s.line)
            else:
                v = self.gen_expr(s.value)
                self._emit("return", arg1=v, line=s.line)
            return

        if isinstance(s, A.Break):
            if not self._loops:
                return
            br, _ = self._loops[-1]
            self._goto(br, s.line)
            return

        if isinstance(s, A.Continue):
            if not self._loops:
                return
            _, cn = self._loops[-1]
            self._goto(cn, s.line)
            return

    def _gen_vardecl(self, d: A.VarDecl) -> None:
        # Declare-only is a no-op in TAC; emit an assignment if there's an init.
        if d.init is not None:
            v = self.gen_expr(d.init)
            self._emit("=", arg1=v, result=d.name, line=d.line)
        # Arrays need no init; their storage is handled by the backend.

    def _gen_if(self, s: A.If) -> None:
        Lthen = self.namer.label()
        Lelse = self.namer.label() if s.else_ is not None else None
        Lend  = self.namer.label()

        cond_name = self.gen_expr(s.cond)
        if s.else_ is not None:
            self._iffalse(cond_name, Lelse, s.line)
        else:
            self._iffalse(cond_name, Lend, s.line)
        # then-branch
        self._label(Lthen, s.line)
        self.gen_stmt(s.then)
        if s.else_ is not None:
            self._goto(Lend, s.line)
            self._label(Lelse, s.line)
            self.gen_stmt(s.else_)
        self._label(Lend, s.line)

    def _gen_for(self, s: A.For) -> None:
        Ltop  = self.namer.label()
        Lstep = self.namer.label()
        Lend  = self.namer.label()
        Lbody = self.namer.label()

        # init
        if s.init is not None:
            if isinstance(s.init, A.VarDecl):
                self._gen_vardecl(s.init)
            elif isinstance(s.init, A.ExprStmt):
                if s.init.expr is not None:
                    self.gen_expr(s.init.expr)
            elif isinstance(s.init, A.Compound):
                for d in s.init.items:
                    self.gen_stmt(d)
            else:  # bare expression
                self.gen_expr(s.init)

        self._label(Ltop, s.line)
        if s.cond is not None:
            cond = self.gen_expr(s.cond)
            self._iffalse(cond, Lend, s.line)
        # body
        self._label(Lbody, s.line)
        self._loops.append((Lend, Lstep))
        self.gen_stmt(s.body)
        self._loops.pop()
        # step
        self._label(Lstep, s.line)
        if s.step is not None:
            self.gen_expr(s.step)
        self._goto(Ltop, s.line)
        self._label(Lend, s.line)

    def _gen_while(self, s: A.While) -> None:
        Ltop = self.namer.label()
        Lend = self.namer.label()
        self._label(Ltop, s.line)
        cond = self.gen_expr(s.cond)
        self._iffalse(cond, Lend, s.line)
        self._loops.append((Lend, Ltop))
        self.gen_stmt(s.body)
        self._loops.pop()
        self._goto(Ltop, s.line)
        self._label(Lend, s.line)

    def _gen_dowhile(self, s: A.DoWhile) -> None:
        Ltop = self.namer.label()
        Lend = self.namer.label()
        self._label(Ltop, s.line)
        self._loops.append((Lend, Ltop))
        self.gen_stmt(s.body)
        self._loops.pop()
        cond = self.gen_expr(s.cond)
        self._ifgoto(cond, Ltop, s.line)
        self._label(Lend, s.line)

    # ── Expression lowering ──────────────────────────────────────────────

    def gen_expr(self, e) -> str:
        """Return the *name* holding the result of `e`, emitting quads as needed."""
        if isinstance(e, A.IntLit):
            return str(e.value)
        if isinstance(e, A.FloatLit):
            return repr(e.value)
        if isinstance(e, A.BoolLit):
            return "1" if e.value else "0"
        if isinstance(e, A.Ident):
            return e.name

        if isinstance(e, A.Subscript):
            arr = self.gen_expr(e.array)
            idx = self.gen_expr(e.index)
            t = self.namer.temp()
            self._emit("[]=", arg1=arr, arg2=idx, result=t, line=e.line)
            return t

        if isinstance(e, A.Member):
            base = self.gen_expr(e.obj)
            t = self.namer.temp()
            self._emit("member", arg1=base, arg2=e.field, result=t, line=e.line)
            return t

        if isinstance(e, A.UnaryOp):
            return self._gen_unary(e)

        if isinstance(e, A.BinOp):
            return self._gen_binop(e)

        if isinstance(e, A.Cond):
            return self._gen_cond(e)

        if isinstance(e, A.Assign):
            return self._gen_assign(e)

        if isinstance(e, A.Call):
            return self._gen_call(e)

        if isinstance(e, A.Cast):
            v = self.gen_expr(e.operand)
            t = self.namer.temp()
            self._emit("cast", arg1=v, arg2=str(e.type), result=t, line=e.line)
            return t

        return "<?>"

    def _gen_unary(self, e: A.UnaryOp) -> str:
        if e.op in ("++", "--"):
            return self._gen_inc_dec(e)
        if e.op == "*":   # pointer dereference
            v = self.gen_expr(e.operand)
            t = self.namer.temp()
            self._emit("=*", arg1=v, result=t, line=e.line)
            return t
        if e.op == "&":   # address-of
            v = self.gen_expr(e.operand)
            t = self.namer.temp()
            self._emit("=", arg1=f"&{v}", result=t, line=e.line)
            return t

        v = self.gen_expr(e.operand)
        t = self.namer.temp()
        op = {"-": "uminus", "+": "uplus", "!": "unot", "~": "bnot"}.get(e.op, e.op)
        self._emit(op, arg1=v, result=t, line=e.line)
        return t

    def _gen_inc_dec(self, e: A.UnaryOp) -> str:
        # x++  →  t = x; x = x + 1; produce t (postfix)  / x (prefix)
        target = self.gen_expr(e.operand)
        delta = "1"
        op = "+" if e.op == "++" else "-"
        if e.postfix:
            t = self.namer.temp()
            self._emit("=", arg1=target, result=t, line=e.line)
            new = self.namer.temp()
            self._emit(op, arg1=target, arg2=delta, result=new, line=e.line)
            self._emit("=", arg1=new, result=target, line=e.line)
            return t
        # prefix
        new = self.namer.temp()
        self._emit(op, arg1=target, arg2=delta, result=new, line=e.line)
        self._emit("=", arg1=new, result=target, line=e.line)
        return target

    def _gen_binop(self, e: A.BinOp) -> str:
        if e.op == "&&":
            return self._gen_logical(e, short_op="and")
        if e.op == "||":
            return self._gen_logical(e, short_op="or")

        lhs = self.gen_expr(e.left)
        rhs = self.gen_expr(e.right)
        t = self.namer.temp()
        self._emit(e.op, arg1=lhs, arg2=rhs, result=t, line=e.line)
        return t

    def _gen_logical(self, e: A.BinOp, short_op: str) -> str:
        """Short-circuit && / ||."""
        Ldone = self.namer.label()
        result = self.namer.temp()

        lhs = self.gen_expr(e.left)
        if short_op == "and":
            # if not lhs: result = 0; goto Ldone
            self._emit("=", arg1="0", result=result, line=e.line)
            self._iffalse(lhs, Ldone, e.line)
        else:                                  # 'or'
            self._emit("=", arg1="1", result=result, line=e.line)
            self._ifgoto(lhs, Ldone, e.line)

        rhs = self.gen_expr(e.right)
        # result = (rhs != 0)
        nez = self.namer.temp()
        self._emit("!=", arg1=rhs, arg2="0", result=nez, line=e.line)
        self._emit("=", arg1=nez, result=result, line=e.line)
        self._label(Ldone, e.line)
        return result

    def _gen_cond(self, e: A.Cond) -> str:
        # x = c ? a : b
        Lelse = self.namer.label()
        Lend  = self.namer.label()
        result = self.namer.temp()

        c = self.gen_expr(e.cond)
        self._iffalse(c, Lelse, e.line)
        a = self.gen_expr(e.then)
        self._emit("=", arg1=a, result=result, line=e.line)
        self._goto(Lend, e.line)
        self._label(Lelse, e.line)
        b = self.gen_expr(e.else_)
        self._emit("=", arg1=b, result=result, line=e.line)
        self._label(Lend, e.line)
        return result

    def _gen_assign(self, e: A.Assign) -> str:
        rhs = self.gen_expr(e.value)
        if e.op != "=":
            # compound assignment: lhs op= rhs   →   lhs = lhs op rhs
            base_op = e.op.rstrip("=")
            cur = self.gen_expr(e.target)
            new = self.namer.temp()
            self._emit(base_op, arg1=cur, arg2=rhs, result=new, line=e.line)
            rhs = new

        # Pick the right write opcode based on target form.
        tgt = e.target
        if isinstance(tgt, A.Ident):
            self._emit("=", arg1=rhs, result=tgt.name, line=e.line)
            return tgt.name
        if isinstance(tgt, A.Subscript):
            arr = self.gen_expr(tgt.array)
            idx = self.gen_expr(tgt.index)
            self._emit("=[]", arg1=arr, arg2=idx, result=rhs, line=e.line)
            return rhs
        if isinstance(tgt, A.UnaryOp) and tgt.op == "*":
            ptr = self.gen_expr(tgt.operand)
            self._emit("*=", arg1=ptr, result=rhs, line=e.line)
            return rhs
        if isinstance(tgt, A.Member):
            base = self.gen_expr(tgt.obj)
            self._emit("=", arg1=rhs, result=f"{base}.{tgt.field}", line=e.line)
            return rhs
        # Fallback: treat target as a scalar name.
        nm = self.gen_expr(tgt)
        self._emit("=", arg1=rhs, result=nm, line=e.line)
        return nm

    def _gen_call(self, e: A.Call) -> str:
        for a in e.args:
            v = self.gen_expr(a)
            self._emit("param", arg1=v, line=e.line)
        if isinstance(e.fn, A.Ident):
            fname = e.fn.name
        else:
            fname = self.gen_expr(e.fn)
        t = self.namer.temp()
        self._emit("call", arg1=fname, arg2=len(e.args), result=t, line=e.line)
        return t


# ── Public façade ──────────────────────────────────────────────────────────

def emit_tac(fn: A.FunctionDef) -> TacProgram:
    """Lower a parsed function into TAC; return a `TacProgram`."""
    return TacEmitter().emit_function(fn)

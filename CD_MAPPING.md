# Compiler Design Syllabus → Code Mapping

> Vishwakarma Institute of Technology · CS3053 Compiler Design (AY 2024-25)
> Course Project · CUDA Kernel Auto-Tuner with full compiler frontend, IR, optimisation, and peephole layer

This document is the answer to the professor's question
*"Where in this project is the Compiler Design content?"*
Every Unit of the syllabus, every listed lab practical, and every
tutorial maps to a specific file, function, or CLI command in this
repository.  Each row can be demonstrated live in viva.

The CD layer lives in the **`cdc/`** package (Compiler Design Components).
The auto-tuner that drives it lives in the **`src/`** package.

---

## High-level pipeline

```
   baseline_kernels.cu                           src/kernels/
        │
        ▼  cdc/preprocessor.py
   cleaned source (#includes/comments stripped, line-numbers preserved)
        │
        ▼  cdc/lexer.py        (PLY/LEX  ── Unit II)
   token stream
        │
        ▼  cdc/parser.py       (PLY/YACC ── Unit III)
        │  cdc/ast_nodes.py    (AST       ── Unit IV)
   AST  ┐
        │  cdc/symbol_table.py + cdc/type_check.py  (Unit III)
   typed AST
        │
        ▼  cdc/ir/tac.py       (Unit IV — Intermediate Code Generation)
   three-address code (quadruples)
        │
        ▼  cdc/ir/basic_block.py   (Unit V — Basic Blocks)
        │  cdc/ir/cfg.py           (Unit V — Flow Graphs + dominators)
        │  cdc/ir/dag.py           (Unit V — DAG of basic blocks)
   CFG + per-block DAG
        │
        ▼  cdc/opt/dfa.py          (Unit VI — Global Data-Flow Analysis)
        │     LiveVariables, ReachingDefinitions, AvailableExpressions
        │  cdc/opt/const_prop.py   (Unit VI — Constant propagation/folding)
        │  cdc/opt/cse.py          (Unit VI — Common Subexpression Elim.)
        │  cdc/opt/dce.py          (Unit VI — Dead Code Elimination)
        │  cdc/opt/licm.py         (Unit VI — Loop-Invariant Code Motion)
        │  cdc/opt/strength_reduce.py (Unit V/VI — Strength Reduction)
        │  cdc/opt/register_pressure.py (Unit VI — live-range analysis)
   optimised IR + cost-model output
        │
        ▼  nvcc -ptx                                (compile)
   PTX (NVIDIA's intermediate representation)
        │
        ▼  cdc/peephole/ptx_peephole.py     (Unit V — Peephole Optimisation)
   optimised PTX
        │
        ▼  nvcc                                       (compile)
   SASS (machine code, sm_75 / RTX 2070)
```

---

## Per-Unit mapping

### Unit I — Introduction to Compilers, Interpreters, Assembler, Linker, Loader

| Topic | Where in code | Demo |
|---|---|---|
| Compiler phases | `cdc/frontend.py`, `cdc/__main__.py` | `python -m cdc <file.cu> --ir` shows every phase output sequentially |
| Cross-compiler | `nvcc` orchestration in `autotune.py:run_baseline_benchmark` | host x86 → device sm_75 PTX → SASS |
| Pipeline orchestration | `cdc.frontend.run_frontend()` | one entry point that runs preprocess → lex → parse → typecheck |

### Unit II — Lexical Analysis

| Topic | Where in code | Demo |
|---|---|---|
| Lexical Analyzer | `cdc/lexer.py` | `python -m cdc <file.cu> --tokens` |
| Specification & recognition of tokens | `reserved` table + 50 token regexes | First lines of `cdc/lexer.py` |
| LEX/FLEX | `ply.lex` (Python LEX-YACC) | `make_lexer()` builds the lexer |
| Regular expressions | `t_FLOAT_LIT`, `t_INT`, `t_IDENT` | Each is a Python regex string |
| Lab Practicals 1–5 (regex programs) | All 50 `t_*` rules | Token output is identical to a lex run |

### Unit III — Syntax Analysis & Symbol Table

| Topic | Where in code | Demo |
|---|---|---|
| Bottom-up parsing, LR/LALR | `ply.yacc` LALR(1) parser | `cdc/parser.py:make_parser()` |
| YACC/BISON specification | `p_*` grammar productions | ~80 production rules |
| Operator precedence | `precedence` table | Standard C precedence + UMINUS, DEREF, ADDROF, CAST |
| Type Checking | `cdc/type_check.py:TypeChecker._compute` | Numeric promotion, address-space rules |
| Type Conversion | `_is_assignable`, `_binop_result` | `int → float`, narrowing warnings |
| Symbol Table | `cdc/symbol_table.py` | Scoped table with `push`/`pop` and CUDA builtins |
| Lab Practicals 6, 7, 8, 9, 10 | Whole frontend | `python -m cdc <file.cu>` |

### Unit IV — Syntax-Directed Translation & Intermediate Code Generation

| Topic | Where in code | Demo |
|---|---|---|
| Syntax-Directed Definitions | `p_*` actions building AST nodes | `cdc/parser.py:p_function_definition` |
| AST as IR | `cdc/ast_nodes.py` | 22 node types, `pretty()` printer |
| Quadruples form | `cdc/ir/tac.py:Quad` | `python -m cdc <file.cu> --tac` |
| Triples / indirect triples | Quadruple form chosen (more uniform) | — |
| Error Detection & Recovery | `Diagnostic` stream + `ParseError` | `python -m cdc <file.cu> --diag-only` |
| Lexical-phase errors | `cdc/lexer.py:t_error` | Illegal-character handler raises `LexError` |
| Syntactic-phase errors | `cdc/parser.py:p_error` | `ParseError` with file:line:column |
| Semantic errors | `cdc/type_check.py` | Use-before-decl, ternary mismatch, narrowing |
| Array refs in expressions | `_compute(Subscript)` + `[]=` / `=[]` quads | matmul: `A[row*N + k]` is one TAC instruction |
| Tutorial 11/12 (Quadruples) | `cdc/ir/tac.py` | `python -m cdc <file.cu> --tac --kernel matmul_naive` |
| Lab Practical 11 (TAC) | `cdc.ir.emit_tac` | Same demo |

### Unit V — Code Generation

| Topic | Where in code | Demo |
|---|---|---|
| Issues in Code Generation | Comments throughout `cdc/ir/tac.py` | — |
| Basic Blocks | `cdc/ir/basic_block.py:partition_blocks` | `python -m cdc <file.cu> --bb --kernel matmul_naive` |
| Flow Graphs | `cdc/ir/cfg.py:build_cfg` | `python -m cdc <file.cu> --cfg` |
| Next-use information | Computed in `cdc/opt/dce.py` (live ranges) | — |
| Simple Code Generator | TAC → kernel C++ via `src/generator.py` | — |
| DAG representation | `cdc/ir/dag.py:DagBuilder` | `python -m cdc <file.cu> --dag --kernel matmul_naive` |
| Generating code from DAGs | `cdc/ir/dag.py:to_dot` + DOT renderer | `--dot` writes Graphviz to `results/cfg_*.dot` |
| Peephole Optimization | `cdc/peephole/ptx_peephole.py` | `python -m cdc.peephole.ptx_peephole results/<k>.ptx` |
| Tutorial 13 (DAG examples) | `cdc/ir/dag.py` | `python -m cdc <file> --dag` |

### Unit VI — Code Optimization, Run-Time Environments, Data Flow Analysis

| Topic | Where in code | Demo |
|---|---|---|
| Principle Sources of Optimization | `cdc/opt/*.py` | Each pass file documents its source category |
| Optimization of Basic Blocks | `cdc/opt/cse.py`, `cdc/opt/const_prop.py` | `python -m cdc <file> --opt` |
| Global Data Flow Analysis | `cdc/opt/dfa.py` | `python -m cdc <file> --dfa` |
| Iterative worklist solver | `cdc/opt/dfa.py:solve` | All three analyses share it |
| Live Variables (backward, ∪) | `cdc/opt/dfa.py:LiveVariables` | `--dfa` output |
| Reaching Definitions (forward, ∪) | `cdc/opt/dfa.py:ReachingDefsSolver` | `--dfa` output |
| Available Expressions (forward, ∩) | `cdc/opt/dfa.py:AvailableExpressions` | `--dfa` output |
| Constant Propagation + Folding | `cdc/opt/const_prop.py` | `python -m cdc <file> --opt` |
| Live Range Analysis | `cdc/opt/register_pressure.py` | `python autotune.py --cdc-regs` |
| Loop-Invariant Code Motion | `cdc/opt/licm.py` | matmul: `t = row*N` hoisted out of `k` loop |
| Strength Reduction | `cdc/opt/strength_reduce.py` | reduction: `2*stride` → `stride<<1` |
| Dead Code Elimination | `cdc/opt/dce.py` | uses live-vars output |
| Storage Organization | CUDA address spaces in `cdc/symbol_table.py` | `__shared__`, `__constant__`, register, global |
| Storage Allocation strategies | tile/unroll variants in `src/generator.py` | register-blocked matmul |
| Machine-Dependent Optimization | RTX 2070 cost model in `cdc/opt/register_pressure.py` | `--cdc-regs` |
| Run-Time Environments | `dim3 grid/block`, `__shared__` extern memory | sample baseline kernels |
| **Case study — LLVM** | acknowledged in README; PTX = NVIDIA IR analogy | `cdc/peephole/ptx_peephole.py` runs on real production IR |
| **Case study — Deep learning compilation** | `src/attention.py` (Flash-Attention variant) + `src/pytorch_integration.py` | matmul + softmax + layernorm + attention kernels are the building blocks of a transformer |
| **Case study — Compiling in multicore environment** | autotuner targets 1024 threads × 36 SMs | RTX 2070 = "multicore" GPU |
| **Case study — Parallel Compilers** | `--workers N` parallel compile of variants | `python autotune.py --kernel=matmul --workers=4` |
| Lab Practical 12 (Code optimiser) | All of `cdc/opt/` | `python -m cdc <file> --opt` |

---

## End-to-end demo for the viva

```bash
# 1. Lexer (Unit II) — show the tokens
python -m cdc src/kernels/baseline_kernels.cu --tokens | head -40

# 2. Frontend (Units I, II, III, IV) — symbol table + type check
python -m cdc src/kernels/baseline_kernels.cu

# 3. IR (Units IV, V) — TAC + basic blocks + CFG + DAG
python -m cdc src/kernels/baseline_kernels.cu --ir --kernel matmul_naive

# 4. Data-flow analysis (Unit VI)
python -m cdc src/kernels/baseline_kernels.cu --dfa --kernel matmul_naive

# 5. Optimisation passes (Unit VI)
python -m cdc src/kernels/baseline_kernels.cu --opt --kernel matmul_naive

# 6. Register-pressure cost model (Unit VI live-range analysis)
python -m cdc src/kernels/baseline_kernels.cu --regs

# 7. Peephole optimisation on real PTX (Unit V)
python -m cdc.peephole.ptx_peephole cdc/peephole/sample.ptx

# 8. Auto-tuner with full CD pipeline reporting
python autotune.py --baseline-only --cdc-frontend --cdc-ir --cdc-opt --cdc-regs
```

---

## Visual artifacts

| Artifact | How to produce | Where it ends up |
|---|---|---|
| Token table | `--tokens`                         | stdout |
| AST dump   | `--ast`                            | stdout |
| Symbol-table report | (default mode)            | stdout |
| Quadruple listing | `--tac`                      | stdout |
| Basic-block listing | `--bb`                     | stdout |
| CFG (textual)   | `--cfg`                        | stdout |
| CFG (Graphviz)  | `--dot`                        | `results/cfg_<kernel>.dot` |
| Dominator tree  | `--cfg`                        | stdout |
| DAG (textual)   | `--dag`                        | stdout |
| DFA results     | `--dfa`                        | stdout |
| Optimised TAC   | `--opt`                        | stdout |
| Reg-pressure report | `--regs`                   | stdout |
| Peephole stats  | `python -m cdc.peephole.ptx_peephole` | stdout + `*.peep.ptx` |
| Roofline plot   | `autotune.py --plots`          | `results/plots/roofline.png` |
| Speedup bars    | `autotune.py --plots`          | `results/plots/speedup.png` |
| Live dashboard  | `streamlit run streamlit_dashboard.py` | browser |

---

## Bibliography (matched to the syllabus text books)

1. Aho, Lam, Sethi, Ullman — *Compilers: Principles, Techniques and Tools* (Dragon Book, 2nd ed., 2006).
   Used for: lexer/parser design (§3, §4), syntax-directed translation (§5),
   type checking (§6), three-address code (§6.2), basic blocks and DAGs
   (§8.4–8.5), data-flow analysis (§9), loop optimisation (§9.6).
2. Cooper, Torczon — *Engineering a Compiler* (2nd ed., 2011).
   Used for: dominator computation (§9.2), iterative DFA framework (§9.3),
   peephole optimisation (§11.5).
3. Williams, Waterman, Patterson — *Roofline: An Insightful Visual Performance
   Model* (CACM 2009).  Used for the cost model in `src/roofline.py`.
4. NVIDIA — *Parallel Thread Execution ISA Application Guide* (PTX manual).
   Used for `cdc/peephole/ptx_peephole.py` opcode patterns.

---

## Future work (mapped to the Course Outcomes / future-course mapping)

| CO | Status |
|---|---|
| CO1: Design basic components (scanner, parser, code generator) | Done — `cdc/lexer.py`, `cdc/parser.py`, `cdc/ir/tac.py` |
| CO2: Perform semantic analysis with attributed definitions | Done — `cdc/type_check.py` |
| CO3: Apply local + global code optimisation | Done — `cdc/opt/*.py` |
| CO4: Synthesise machine code for runtime environment | Partially — code emitted via `nvcc` from optimised templates |
| CO5: Develop software solutions for compiler problems | Done — auto-tuner is the application |
| CO6: Adapt to emerging trends in language processing | Done — Deep learning kernels (attention) + GPU compilation case study |

The "Future Course Mapping: Parallel Compiler" line in the syllabus is
already partially honoured by `autotune.py --workers N` (parallel compile
of variants).

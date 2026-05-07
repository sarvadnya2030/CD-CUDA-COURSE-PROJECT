# Graph Report - Cuda-Optimization-v2  (2026-05-07)

## Corpus Check
- 44 files · ~55,570 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 952 nodes · 1704 edges · 39 communities detected
- Extraction: 57% EXTRACTED · 43% INFERRED · 0% AMBIGUOUS · INFERRED: 728 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `BasicBlock` - 61 edges
2. `Quad` - 59 edges
3. `RooflineAnalyzer` - 43 edges
4. `ControlFlowGraph` - 38 edges
5. `TypeChecker` - 26 edges
6. `Node` - 26 edges
7. `BenchmarkResult` - 26 edges
8. `TacEmitter` - 24 edges
9. `SymbolTable` - 23 edges
10. `MemoryAccessPattern` - 21 edges

## Surprising Connections (you probably didn't know these)
- `autotune.py — main entry point for the CUDA kernel auto-tuner.  Usage:     pytho` --uses--> `RooflineAnalyzer`  [INFERRED]
  autotune.py → src/roofline.py
- `autotune.py — main entry point for the CUDA kernel auto-tuner.  Usage:     pytho` --uses--> `CorrectnessVerifier`  [INFERRED]
  autotune.py → src/verifier.py
- `Kernel-name-based fallback used when source parsing fails.` --uses--> `RooflineAnalyzer`  [INFERRED]
  autotune.py → src/roofline.py
- `Kernel-name-based fallback used when source parsing fails.` --uses--> `CorrectnessVerifier`  [INFERRED]
  autotune.py → src/verifier.py
- `Return a KernelProfile for the given kernel name.      When use_libclang=True (P` --uses--> `RooflineAnalyzer`  [INFERRED]
  autotune.py → src/roofline.py

## Communities

### Community 0 - "PLY Parser Frontend"
Cohesion: 0.01
Nodes (163): analyze_file(), build_search_space(), _count_loop_depth(), _count_reductions_from_tokens(), _extract_block_dim(), _extract_block_dim_from_tu(), _extract_memory(), _extract_params() (+155 more)

### Community 1 - "Phase 2 Compiler"
Cohesion: 0.03
Nodes (94): BasicBlock, _find_leaders(), format_blocks(), partition_blocks(), basic_block.py — partition a TAC program into basic blocks.  A *basic block* is, Pretty-print a list of basic blocks., A straight-line sequence of quadruples., Return sorted list of indices in `prog.quads` that begin a basic block. (+86 more)

### Community 2 - "Kernel Autotuner"
Cohesion: 0.04
Nodes (79): autotune_attention(), autotune_kernel(), build_kernel_profile(), compile_variant(), _hardcoded_profile(), main(), process_variant(), autotune.py — main entry point for the CUDA kernel auto-tuner.  Usage:     pytho (+71 more)

### Community 3 - "Analytics Dashboard"
Cohesion: 0.05
Nodes (59): _fmt_float(), _md_table(), print_final_summary(), print_terminal_summary(), reporter.py — Markdown report generation and terminal summary tables.  Generates, Write the Markdown report to results/{kernel}_report.md.          Returns:, Compact one-line param string for table cells., Print a clean terminal summary after tuning completes.      Uses only stdlib str (+51 more)

### Community 4 - "Type System"
Cohesion: 0.07
Nodes (40): format_report(), FrontendKernel, FrontendResult, frontend.py — orchestrates the compiler frontend pipeline.      raw .cu file, Walk an AST and offset every node's `line` attribute by `delta`., Build a human-readable summary of a FrontendResult., Result of running the full frontend on one kernel., Full-file frontend result — one entry per __global__/__device__ kernel. (+32 more)

### Community 5 - "Search Strategies"
Cohesion: 0.05
Nodes (26): ABC, BayesianOptStrategy, _build_skopt_space(), _expand(), _key(), make_strategy(), search.py — Search strategy implementations for CUDA kernel auto-tuning.  Provid, Bayesian optimisation with a Gaussian Process surrogate (scikit-optimize). (+18 more)

### Community 6 - "GPU Integration"
Cohesion: 0.09
Nodes (26): apply_tuned_layers(), benchmark_gpt2_attention(), benchmark_sdpa(), cuda_time_samples(), _flash_attention(), GPT2SelfAttention, load_best_params(), _manual_attention() (+18 more)

### Community 7 - "Verification Engine"
Cohesion: 0.08
Nodes (27): _compare(), CorrectnessVerifier, _extract_kernel_code(), _gamma_input(), _make_input(), _parse_verify_output(), verifier.py — Correctness verification for generated CUDA kernel variants.  For, Match: (i % 97) * 0.01 - 0.5 (+19 more)

### Community 8 - "TAC IR Generation"
Cohesion: 0.19
Nodes (9): emit_tac(), Namer, tac.py — Three-Address Code generator (quadruples).  Lowers the AST produced by, Mints fresh temporary names and labels., Walks a `FunctionDef` AST and emits quadruples into `prog.quads`.      The emitt, Return the *name* holding the result of `e`, emitting quads as needed., Short-circuit && / ||., Lower a parsed function into TAC; return a `TacProgram`. (+1 more)

### Community 9 - "AST Nodes"
Cohesion: 0.11
Nodes (31): Assign, BinOp, BoolLit, Break, Call, Cast, Compound, Cond (+23 more)

### Community 10 - "Phase 3 Analysis"
Cohesion: 0.13
Nodes (19): _is_one_f32(), _is_zero_f32(), optimise_ptx_file(), _parse_line(), _pat_algebraic_identity(), _pat_double_mov(), _pat_mul_add_fma(), _pat_redundant_load() (+11 more)

### Community 11 - "Data Flow Analysis"
Cohesion: 0.12
Nodes (23): collect_ncu_metrics(), compile_kernel(), compute_significance(), _compute_stats(), parse_check_output(), _parse_ncu_csv(), parse_timing_output(), parse_timing_output_simple() (+15 more)

### Community 12 - "Optimization Passes"
Cohesion: 0.18
Nodes (19): _ascii_bar(), _convergence_chart(), _dark_layout(), generate_html_dashboard(), _load_convergence(), _load_tuning(), main(), _occupancy_heatmap() (+11 more)

### Community 13 - "Peephole Optimizer"
Cohesion: 0.12
Nodes (16): Exception, column(), LexError, make_lexer(), lexer.py — PLY (Python Lex-Yacc) tokeniser for a CUDA C subset.  This is the lex, r"((\d+\.\d*|\.\d+)([eE][+-]?\d+)?[fFlL]?|\d+[eE][+-]?\d+[fFlL]?|\d+[fF]), r"(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9]\d*)[uUlL]*, r"[A-Za-z_][A-Za-z_0-9]* (+8 more)

### Community 14 - "Results Storage"
Cohesion: 0.14
Nodes (17): attention_bytes_flash(), attention_bytes_naive(), attention_flops(), build_attention_search_space(), enumerate_attention_variants(), generate_attention(), _make_tag(), attention.py — Multi-Head Attention kernel templates and search space.  Implemen (+9 more)

### Community 15 - "Benchmarking"
Cohesion: 0.16
Nodes (11): compute_occupancy(), _make_info(), OccupancyInfo, parse_ptxas_stderr(), occupancy.py — CUDA occupancy and register pressure analysis for sm_75.  Parses, Compute theoretical occupancy for sm_75 given resource usage.      Args:, Compile *src_path* and parse ptxas occupancy info from stderr.          If *out_, Derive OccupancyInfo from already-captured ptxas stderr.          Used when the (+3 more)

### Community 16 - "Reporting"
Cohesion: 0.27
Nodes (4): build_dag(), DagBuilder, DagNode, to_dot()

### Community 17 - "Configuration"
Cohesion: 0.16
Nodes (11): _dump_sass(), _parse_ptx_file(), PTXComparison, PTXMetrics, ptx_analysis.py — PTX and SASS static analysis for CUDA kernel variants.  Workfl, Run cuobjdump --dump-sass and return the output text.      Returns None if cuobj, Compile *src_path* to PTX and return per-function metrics.          Optionally a, Build a PTXComparison between baseline and optimised metrics.          Matches t (+3 more)

### Community 18 - "Testing Utils"
Cohesion: 0.33
Nodes (8): _load(), main(), plot_ms_bars(), plot_roofline(), plot_speedup(), plots.py — generate figures from the auto-tuner's JSON artifacts.  Reads:   resu, Log-log roofline: x = arithmetic intensity (flop/byte),     y = throughput (GFLO, Return {kernel: {baseline_ms, best_ms, roofline}} — missing kernels skipped.

### Community 19 - "Symbol Table"
Cohesion: 0.4
Nodes (5): extract_device_functions(), preprocess(), preprocessor.py — minimal preprocessor for CUDA C source.  The PLY lexer + parse, Strip comments and preprocessor directives from CUDA source.      Whitespace and, Locate `__global__` and `__device__` function definitions in the source.      Us

### Community 20 - "Code Generation"
Cohesion: 0.67
Nodes (3): main(), cd_mapping_figure.py — render the syllabus-to-code mapping as a single PNG slide, render()

### Community 21 - "Error Handling"
Cohesion: 1.0
Nodes (1): Build an OccupancyInfo from a parsed ptxas entry.

### Community 22 - "Memory Management"
Cohesion: 1.0
Nodes (1): Print a formatted occupancy table.          Accepts either BenchmarkResult objec

### Community 23 - "Performance Metrics"
Cohesion: 1.0
Nodes (1): Extract parameter names and type strings from a function cursor.

### Community 24 - "Compilation Pipeline"
Cohesion: 1.0
Nodes (1): Derive MemoryAccessPattern from kernel body text.

### Community 25 - "Analysis Framework"
Cohesion: 1.0
Nodes (1): Return everything in src_path before the benchmark driver comment.          The

### Community 26 - "Utilities"
Cohesion: 1.0
Nodes (1): Parse VERIFY_OUTPUT lines into a list of floats.

### Community 27 - "Logging"
Cohesion: 1.0
Nodes (1): Return (max_abs_diff, passes_tolerance).

### Community 28 - "Dependencies"
Cohesion: 1.0
Nodes (1): Return up to *n* parameter configurations to evaluate next.          Each config

### Community 29 - "Build System"
Cohesion: 1.0
Nodes (1): Record the observed latency *result_ms* for *params*.          Called after each

### Community 30 - "Project Config"
Cohesion: 1.0
Nodes (1): Return the params dict that achieved the lowest latency so far.

### Community 31 - "Documentation"
Cohesion: 1.0
Nodes (1): Expand the full Cartesian product, applying validity pruning.

### Community 32 - "Examples"
Cohesion: 1.0
Nodes (1): Total number of configs in the grid.

### Community 33 - "Integration Tests"
Cohesion: 1.0
Nodes (1): Number of configs not yet suggested.

### Community 34 - "Unit Tests"
Cohesion: 1.0
Nodes (1): Convert our space dict into scikit-optimize dimensions.

### Community 35 - "Mock Data"
Cohesion: 1.0
Nodes (1): Stable string key for a param dict.

### Community 36 - "Fixtures"
Cohesion: 1.0
Nodes (1): Number of configs in the current round.

### Community 37 - "Cache Management"
Cohesion: 1.0
Nodes (1): Current SHA round index (0-based).

### Community 38 - "Runtime State"
Cohesion: 1.0
Nodes (1): True if CUDA runtime was successfully loaded.

## Knowledge Gaps
- **252 isolated node(s):** `parser.py — CUDA kernel analyzer.  Extracts optimization-relevant parameters fro`, `external_decl_list : external_decl`, `external_decl_list : external_decl_list external_decl`, `external_decl : function_definition`, `function_definition : qualifier_list_opt type_spec declarator LPAREN param_list_` (+247 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Error Handling`** (1 nodes): `Build an OccupancyInfo from a parsed ptxas entry.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Memory Management`** (1 nodes): `Print a formatted occupancy table.          Accepts either BenchmarkResult objec`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Performance Metrics`** (1 nodes): `Extract parameter names and type strings from a function cursor.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Compilation Pipeline`** (1 nodes): `Derive MemoryAccessPattern from kernel body text.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Analysis Framework`** (1 nodes): `Return everything in src_path before the benchmark driver comment.          The`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Utilities`** (1 nodes): `Parse VERIFY_OUTPUT lines into a list of floats.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Logging`** (1 nodes): `Return (max_abs_diff, passes_tolerance).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dependencies`** (1 nodes): `Return up to *n* parameter configurations to evaluate next.          Each config`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Build System`** (1 nodes): `Record the observed latency *result_ms* for *params*.          Called after each`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Project Config`** (1 nodes): `Return the params dict that achieved the lowest latency so far.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Documentation`** (1 nodes): `Expand the full Cartesian product, applying validity pruning.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Examples`** (1 nodes): `Total number of configs in the grid.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Integration Tests`** (1 nodes): `Number of configs not yet suggested.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Unit Tests`** (1 nodes): `Convert our space dict into scikit-optimize dimensions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mock Data`** (1 nodes): `Stable string key for a param dict.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Fixtures`** (1 nodes): `Number of configs in the current round.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cache Management`** (1 nodes): `Current SHA round index (0-based).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Runtime State`** (1 nodes): `True if CUDA runtime was successfully loaded.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataFlowAnalysis` connect `Phase 2 Compiler` to `Search Strategies`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `Quad` connect `Phase 2 Compiler` to `TAC IR Generation`, `Reporting`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Are the 56 inferred relationships involving `BasicBlock` (e.g. with `partition_blocks()` and `Quad`) actually correct?**
  _`BasicBlock` has 56 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `Quad` (e.g. with `._emit()` and `BasicBlock`) actually correct?**
  _`Quad` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `RooflineAnalyzer` (e.g. with `autotune.py — main entry point for the CUDA kernel auto-tuner.  Usage:     pytho` and `Kernel-name-based fallback used when source parsing fails.`) actually correct?**
  _`RooflineAnalyzer` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `ControlFlowGraph` (e.g. with `build_cfg()` and `BasicBlock`) actually correct?**
  _`ControlFlowGraph` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `TypeChecker` (e.g. with `check_function()` and `Symbol`) actually correct?**
  _`TypeChecker` has 13 INFERRED edges - model-reasoned connections that need verification._
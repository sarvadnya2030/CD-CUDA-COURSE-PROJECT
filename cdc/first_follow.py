"""
FIRST/FOLLOW Sets and LL(1) Parse Table Construction.

Maps to Course Unit III: LL(1) Parsers, FIRST/FOLLOW set computation,
LL(1) parse table construction (Tutorial 7, Lab Practical 7).

This module implements the textbook algorithms for:
  1. Computing nullable non-terminals (which can derive ε)
  2. Computing FIRST(α) for strings α
  3. Computing FOLLOW(A) for non-terminals A
  4. Constructing LL(1) predict tables

Grammar used: simplified CUDA C subset (cf. cdc/parser.py).
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from enum import Enum


class Symbol(Enum):
    """Terminal vs non-terminal."""
    TERMINAL = "terminal"
    NON_TERMINAL = "non_terminal"


@dataclass
class Production:
    """One grammar production: lhs -> rhs."""
    lhs: str
    rhs: List[str]  # sequence of symbols; empty list = epsilon

    def __str__(self):
        if not self.rhs:
            return f"{self.lhs} → ε"
        return f"{self.lhs} → {' '.join(self.rhs)}"


class Grammar:
    """Context-free grammar for CUDA C subset (education)."""

    def __init__(self, productions: List[Production], start_symbol: str = "translation_unit"):
        """
        Args:
            productions: List of Production objects
            start_symbol: axiom / goal symbol
        """
        self.productions = productions
        self.start_symbol = start_symbol

        # Pre-compute non-terminals
        self.non_terminals = {start_symbol}
        for prod in productions:
            self.non_terminals.add(prod.lhs)
            for sym in prod.rhs:
                if not self._is_terminal(sym):
                    self.non_terminals.add(sym)

    def _is_terminal(self, sym: str) -> bool:
        """Check if symbol is a terminal (not a non-terminal)."""
        return sym in {
            # Keywords and punctuation
            "int", "float", "void", "double", "bool",
            "const", "__global__", "__device__", "__shared__",
            "if", "else", "for", "while", "do", "return", "break", "continue",
            "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=",
            "&&", "||", "!", "&", "|", "^", "~", "<<", ">>",
            "(", ")", "{", "}", "[", "]", ";", ",", ".", "->",
            "threadIdx", "blockIdx", "blockDim", "gridDim",
            "ID", "INT_LIT", "FLOAT_LIT", "EOF"
        }


def compute_nullable(grammar: Grammar) -> Set[str]:
    """
    Compute which non-terminals can derive ε (epsilon).

    Algorithm: Iterate over productions, marking non-terminals whose RHS
    is all nullable as themselves nullable, until fixpoint.

    Returns:
        Set of nullable non-terminal names.
    """
    nullable = set()
    changed = True

    while changed:
        changed = False
        for prod in grammar.productions:
            if prod.lhs in nullable:
                continue

            # If RHS is empty, it's nullable
            if not prod.rhs:
                nullable.add(prod.lhs)
                changed = True
            # If all RHS symbols are nullable, LHS is nullable
            elif all(sym in nullable or grammar._is_terminal(sym) for sym in prod.rhs):
                if prod.lhs not in nullable:
                    nullable.add(prod.lhs)
                    changed = True

    return nullable


def compute_first(grammar: Grammar, nullable: Set[str]) -> Dict[str, Set[str]]:
    """
    Compute FIRST sets for all non-terminals.

    FIRST(A) = set of terminals that can appear as the first symbol
    of any string derivable from A.

    Algorithm: Iterate over non-terminals, collecting FIRST symbols from
    productions, until fixpoint.

    Returns:
        Dict[non_terminal] → Set[terminals]
    """
    first = {nt: set() for nt in grammar.non_terminals}

    changed = True
    while changed:
        changed = False

        for prod in grammar.productions:
            lhs = prod.lhs

            # For each symbol in RHS:
            for sym in prod.rhs:
                if grammar._is_terminal(sym):
                    # Terminal: add it directly
                    if sym not in first[lhs]:
                        first[lhs].add(sym)
                        changed = True
                    # Stop; non-nullable blocks further symbols
                    break
                else:
                    # Non-terminal: add its FIRST (minus epsilon)
                    for t in first[sym]:
                        if t != "ε" and t not in first[lhs]:
                            first[lhs].add(t)
                            changed = True
                    # If this symbol is not nullable, stop
                    if sym not in nullable:
                        break
            else:
                # All RHS symbols were nullable (or RHS is empty)
                # So ε is in FIRST(LHS)
                if "ε" not in first[lhs]:
                    first[lhs].add("ε")
                    changed = True

    return first


def compute_follow(
    grammar: Grammar,
    first: Dict[str, Set[str]],
    nullable: Set[str]
) -> Dict[str, Set[str]]:
    """
    Compute FOLLOW sets for all non-terminals.

    FOLLOW(A) = set of terminals that can appear immediately after A
    in some sentential form derivable from the start symbol.

    Algorithm: Iterate over productions, for each RHS position, collect
    FIRST of the suffix and transitively FOLLOW of LHS if suffix is nullable,
    until fixpoint.

    Returns:
        Dict[non_terminal] → Set[terminals]
    """
    follow = {nt: set() for nt in grammar.non_terminals}

    # Bootstrap: $ (end of input) is in FOLLOW of start symbol
    follow[grammar.start_symbol].add("$")

    changed = True
    while changed:
        changed = False

        for prod in grammar.productions:
            lhs = prod.lhs

            # For each position i in RHS
            for i, sym in enumerate(prod.rhs):
                if grammar._is_terminal(sym):
                    continue

                # sym is a non-terminal at position i
                # Add FIRST(sym[i+1:]) - {ε} to FOLLOW(sym)
                suffix = prod.rhs[i + 1:]
                suffix_first = first_of_string(suffix, first, nullable, grammar)

                for t in suffix_first:
                    if t != "ε" and t not in follow[sym]:
                        follow[sym].add(t)
                        changed = True

                # If suffix is nullable, add FOLLOW(lhs) to FOLLOW(sym)
                if is_string_nullable(suffix, nullable):
                    for t in follow[lhs]:
                        if t not in follow[sym]:
                            follow[sym].add(t)
                            changed = True

    return follow


def first_of_string(
    symbols: List[str],
    first: Dict[str, Set[str]],
    nullable: Set[str],
    grammar: Grammar
) -> Set[str]:
    """
    Compute FIRST(α) for a string α of symbols.
    """
    result = set()

    for sym in symbols:
        if grammar._is_terminal(sym):
            result.add(sym)
            return result
        else:
            for t in first[sym]:
                if t != "ε":
                    result.add(t)
            if sym not in nullable:
                return result

    # All symbols are nullable
    result.add("ε")
    return result


def is_string_nullable(symbols: List[str], nullable: Set[str]) -> bool:
    """Check if a string is nullable (all symbols nullable)."""
    return all(sym in nullable for sym in symbols)


def build_ll1_table(
    grammar: Grammar,
    first: Dict[str, Set[str]],
    follow: Dict[str, Set[str]],
    nullable: Set[str]
) -> Dict[Tuple[str, str], Optional[Production]]:
    """
    Construct LL(1) predictive parse table.

    Table[non_terminal, lookahead] = production to apply.

    Algorithm: For each production A → α:
      For each terminal a in FIRST(α) - {ε}:
        add production to TABLE[A, a]
      If α is nullable:
        For each terminal a in FOLLOW(A):
          add production to TABLE[A, a]

    Returns:
        Dict[(non_terminal, lookahead_terminal)] → Production
        (None if no entry, indicating a parse error).
    """
    table = {}
    conflicts = []

    for prod in grammar.productions:
        lhs = prod.lhs
        rhs = prod.rhs

        # Compute FIRST(rhs)
        first_rhs = first_of_string(rhs, first, nullable, grammar)

        # Add entries for FIRST(rhs) - {ε}
        for a in first_rhs:
            if a != "ε":
                key = (lhs, a)
                if key in table and table[key] != prod:
                    conflicts.append((key, table[key], prod))
                table[key] = prod

        # If rhs is nullable, add entries for FOLLOW(lhs)
        if is_string_nullable(rhs, nullable):
            for a in follow[lhs]:
                key = (lhs, a)
                if key in table and table[key] != prod:
                    conflicts.append((key, table[key], prod))
                table[key] = prod

    if conflicts:
        # Report conflicts (indicates grammar is not LL(1))
        print("⚠ LL(1) conflicts detected (grammar may not be LL(1)):")
        for (nt, term), prod1, prod2 in conflicts[:3]:
            print(f"  [{nt}, {term}]: {prod1} vs {prod2}")

    return table


def format_first_follow(
    first: Dict[str, Set[str]],
    follow: Dict[str, Set[str]],
    non_terminals: Set[str]
) -> str:
    """Pretty-print FIRST and FOLLOW tables."""
    lines = []
    lines.append("=" * 70)
    lines.append("FIRST and FOLLOW Sets")
    lines.append("=" * 70)

    # Sort non-terminals for consistent output
    nts = sorted(non_terminals)

    # FIRST table
    lines.append("\nFIRST Sets:")
    lines.append("-" * 70)
    for nt in nts:
        first_set = sorted(first.get(nt, set()))
        first_str = ", ".join(first_set) if first_set else "{}"
        lines.append(f"  FIRST({nt:20s}) = {{ {first_str:40s} }}")

    # FOLLOW table
    lines.append("\nFOLLOW Sets:")
    lines.append("-" * 70)
    for nt in nts:
        follow_set = sorted(follow.get(nt, set()))
        follow_str = ", ".join(follow_set) if follow_set else "{}"
        lines.append(f"  FOLLOW({nt:19s}) = {{ {follow_str:40s} }}")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_ll1_table(table: Dict[Tuple[str, str], Production], grammar: Grammar) -> str:
    """Pretty-print LL(1) predictive parse table."""
    lines = []
    lines.append("=" * 90)
    lines.append("LL(1) Predictive Parse Table")
    lines.append("=" * 90)

    # Collect all lookaheads
    lookaheads = sorted(set(term for (_, term) in table.keys()))
    non_terminals = sorted(grammar.non_terminals)

    # Header
    header = "Non-Terminal".ljust(20) + " | " + " | ".join(f"{la:12s}" for la in lookaheads)
    lines.append(header)
    lines.append("-" * len(header))

    # Table rows
    for nt in non_terminals:
        row = nt.ljust(20) + " | "
        cells = []
        for la in lookaheads:
            key = (nt, la)
            if key in table:
                prod = table[key]
                prod_str = " ".join(prod.rhs) if prod.rhs else "ε"
                cells.append(f"{prod_str:12s}")
            else:
                cells.append("error".ljust(12))
        row += " | ".join(cells)
        lines.append(row)

    lines.append("=" * 90)
    return "\n".join(lines)


# ==============================================================================
# Example Grammar: Simplified CUDA C Subset
# ==============================================================================

def build_example_grammar() -> Grammar:
    """
    Build a simplified CUDA C grammar for educational purposes.

    This is not the full grammar used by the PLY parser (which uses LALR(1)),
    but a simplified LL(1)-friendly subset to demonstrate FIRST/FOLLOW
    and parse table construction.
    """
    productions = [
        # translation_unit
        Production("translation_unit", ["function_def"]),
        Production("translation_unit", ["translation_unit", "function_def"]),

        # function_def
        Production("function_def", ["type", "ID", "(", "param_list", ")", "compound"]),
        Production("function_def", ["__global__", "void", "ID", "(", "param_list", ")", "compound"]),

        # param_list
        Production("param_list", []),
        Production("param_list", ["param"]),
        Production("param_list", ["param_list", ",", "param"]),

        # param
        Production("param", ["type", "ID"]),
        Production("param", ["type", "*", "ID"]),

        # type
        Production("type", ["int"]),
        Production("type", ["float"]),
        Production("type", ["void"]),
        Production("type", ["double"]),
        Production("type", ["bool"]),

        # compound
        Production("compound", ["{", "stmt_list", "}"]),

        # stmt_list
        Production("stmt_list", []),
        Production("stmt_list", ["stmt_list", "stmt"]),

        # stmt
        Production("stmt", ["expr", ";"]),
        Production("stmt", ["ID", "=", "expr", ";"]),
        Production("stmt", ["if", "(", "expr", ")", "stmt"]),
        Production("stmt", ["if", "(", "expr", ")", "stmt", "else", "stmt"]),
        Production("stmt", ["for", "(", "expr", ";", "expr", ";", "expr", ")", "stmt"]),
        Production("stmt", ["while", "(", "expr", ")", "stmt"]),
        Production("stmt", ["return", "expr", ";"]),
        Production("stmt", ["return", ";"]),
        Production("stmt", ["compound"]),

        # expr
        Production("expr", ["term"]),
        Production("expr", ["expr", "+", "term"]),
        Production("expr", ["expr", "-", "term"]),
        Production("expr", ["expr", "*", "term"]),
        Production("expr", ["expr", "/", "term"]),
        Production("expr", ["expr", "==", "term"]),
        Production("expr", ["expr", "!=", "term"]),
        Production("expr", ["expr", "<", "term"]),
        Production("expr", ["expr", ">", "term"]),
        Production("expr", ["expr", "&&", "term"]),
        Production("expr", ["expr", "||", "term"]),

        # term
        Production("term", ["ID"]),
        Production("term", ["INT_LIT"]),
        Production("term", ["FLOAT_LIT"]),
        Production("term", ["(", "expr", ")"]),
    ]

    return Grammar(productions, start_symbol="translation_unit")


if __name__ == "__main__":
    # Demo
    grammar = build_example_grammar()
    nullable = compute_nullable(grammar)
    first = compute_first(grammar, nullable)
    follow = compute_follow(grammar, first, nullable)

    print(format_first_follow(first, follow, grammar.non_terminals))

    table = build_ll1_table(grammar, first, follow, nullable)
    print("\n" + format_ll1_table(table, grammar))

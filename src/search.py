"""
search.py — Search strategy implementations for CUDA kernel auto-tuning.

Provides three strategies with a unified interface:

  GridSearchStrategy       — exhaustive parameter grid (original behaviour)
  BayesianOptStrategy      — Gaussian Process surrogate via scikit-optimize
  SuccessiveHalvingStrategy — SHA bracket (Jamieson & Talwalkar, 2016)

All strategies share:
    suggest(n: int) → List[dict]      return up to n parameter configs to try
    update(params, ms)                feed back a measured latency
    best() → dict                     return best params found so far

Reference:
  Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian
  Optimization of Machine Learning Algorithms. NeurIPS.

  Jamieson, K., & Talwalkar, A. (2016). Non-stochastic Best Arm
  Identification and Hyperparameter Optimization. AISTATS.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

try:
    import skopt
    from skopt import gp_minimize
    from skopt.space import Integer, Categorical
    _SKOPT_OK = True
except ImportError:
    skopt = None  # type: ignore
    _SKOPT_OK = False
    warnings.warn(
        "scikit-optimize not installed — Bayesian strategy will fall back to grid. "
        "Install with: pip install scikit-optimize",
        RuntimeWarning,
        stacklevel=2,
    )


# ── Base interface ─────────────────────────────────────────────────────────

class SearchStrategy(ABC):
    """Abstract base for all search strategies."""

    @abstractmethod
    def suggest(self, n: int = 1) -> List[dict]:
        """
        Return up to *n* parameter configurations to evaluate next.

        Each config is a plain dict mapping param names to values.
        """

    @abstractmethod
    def update(self, params: dict, result_ms: float) -> None:
        """
        Record the observed latency *result_ms* for *params*.

        Called after each variant is benchmarked so the strategy can
        exploit the result.
        """

    @abstractmethod
    def best(self) -> Optional[dict]:
        """Return the params dict that achieved the lowest latency so far."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── Grid Search ────────────────────────────────────────────────────────────

class GridSearchStrategy(SearchStrategy):
    """
    Exhaustive grid search over all parameter combinations.

    This is the original auto-tuner behaviour, preserved for backward
    compatibility and as a baseline comparison for other strategies.
    """

    def __init__(self, space: dict, kernel: str = "") -> None:
        """
        Args:
            space:  Search-space dict from parser.build_search_space().
            kernel: Kernel name (used for matmul tile pruning).
        """
        self._kernel   = kernel
        self._space    = space
        self._queue    = self._expand(space, kernel)
        self._results: List[tuple[dict, float]] = []
        self._ptr      = 0

    @staticmethod
    def _expand(space: dict, kernel: str) -> List[dict]:
        """Expand the full Cartesian product, applying validity pruning."""
        keys = [k for k in space if not k.startswith("_")]
        vals = [space[k] for k in keys]
        configs: List[dict] = []
        for combo in itertools.product(*vals):
            params = dict(zip(keys, combo))
            if kernel == "matmul":
                if params["tile_x"] != params["tile_y"]:
                    continue
                if params["tile_x"] > params["block_size"]:
                    continue
            configs.append(params)
        return configs

    def suggest(self, n: int = 1) -> List[dict]:
        """Return next *n* configs from the grid in order."""
        batch = self._queue[self._ptr: self._ptr + n]
        self._ptr += n
        return batch

    def update(self, params: dict, result_ms: float) -> None:
        """Record result (grid search ignores it during traversal)."""
        self._results.append((params, result_ms))

    def best(self) -> Optional[dict]:
        """Return params with lowest recorded latency."""
        if not self._results:
            return None
        return min(self._results, key=lambda x: x[1])[0]

    @property
    def total(self) -> int:
        """Total number of configs in the grid."""
        return len(self._queue)

    @property
    def remaining(self) -> int:
        """Number of configs not yet suggested."""
        return max(0, len(self._queue) - self._ptr)


# ── Bayesian Optimisation ──────────────────────────────────────────────────

class BayesianOptStrategy(SearchStrategy):
    """
    Bayesian optimisation with a Gaussian Process surrogate (scikit-optimize).

    Falls back to GridSearchStrategy if scikit-optimize is not installed.

    n_calls (default 40) limits total objective evaluations.  The GP is
    updated after each observed result and subsequent suggestions use the
    Expected Improvement acquisition function.
    """

    def __init__(self, space: dict, kernel: str = "", n_calls: int = 40) -> None:
        """
        Args:
            space:   Search-space dict from parser.build_search_space().
            kernel:  Kernel name.
            n_calls: Maximum number of evaluations for the GP.
        """
        self._kernel   = kernel
        self._n_calls  = n_calls
        self._results: List[tuple[dict, float]] = []

        if not _SKOPT_OK:
            warnings.warn("Falling back to GridSearch (scikit-optimize missing).")
            self._fallback: Optional[GridSearchStrategy] = GridSearchStrategy(space, kernel)
        else:
            self._fallback = None

        self._space_def = space
        self._skopt_dims, self._keys = self._build_skopt_space(space, kernel)
        self._x_obs: List[list] = []
        self._y_obs: List[float] = []
        self._next_batch: List[dict] = []
        self._n_initial_points = min(10, n_calls // 2)

    @staticmethod
    def _build_skopt_space(space: dict, kernel: str):
        """Convert our space dict into scikit-optimize dimensions."""
        keys = [k for k in space if not k.startswith("_")]
        dims = []
        for k in keys:
            vals = space[k]
            if all(isinstance(v, bool) for v in vals):
                dims.append(Categorical([int(v) for v in vals], name=k))
            elif all(isinstance(v, int) for v in vals):
                dims.append(Categorical(vals, name=k))
            else:
                dims.append(Categorical(vals, name=k))
        return dims, keys

    def _vec_to_params(self, x: list) -> dict:
        """Convert a skopt point vector back to a param dict."""
        p = {}
        for k, v in zip(self._keys, x):
            orig_vals = self._space_def.get(k, [])
            # Restore bool if original space was bool
            if orig_vals and all(isinstance(ov, bool) for ov in orig_vals):
                p[k] = bool(v)
            else:
                p[k] = v
        return p

    def _params_to_vec(self, params: dict) -> list:
        """Convert params dict to a skopt point vector."""
        return [int(params[k]) if isinstance(params[k], bool) else params[k]
                for k in self._keys]

    def _is_valid(self, params: dict) -> bool:
        """Apply kernel-specific validity constraints."""
        if self._kernel == "matmul":
            return (params.get("tile_x") == params.get("tile_y") and
                    params.get("tile_x", 0) <= params.get("block_size", 0))
        return True

    def suggest(self, n: int = 1) -> List[dict]:
        """Return up to *n* next configs suggested by the GP."""
        if self._fallback is not None:
            return self._fallback.suggest(n)

        configs: List[dict] = []
        attempts = 0

        while len(configs) < n and attempts < 200:
            attempts += 1
            if len(self._x_obs) < self._n_initial_points:
                # Initial random exploration
                x = [random.choice(self._space_def[k])
                     for k in self._keys if not k.startswith("_")]
                p = self._vec_to_params(x)
            else:
                # GP suggestion via dummy minimisation call
                try:
                    result = gp_minimize(
                        func=lambda _: 0.0,          # dummy; we supply observations
                        dimensions=self._skopt_dims,
                        n_calls=1,
                        n_initial_points=0,
                        x0=self._x_obs[-self._n_initial_points:],
                        y0=self._y_obs[-self._n_initial_points:],
                        random_state=random.randint(0, 2**31),
                        acq_func="EI",
                    )
                    p = self._vec_to_params(result.x_iters[-1])
                except Exception:
                    x = [random.choice(self._space_def[k])
                         for k in self._keys if not k.startswith("_")]
                    p = self._vec_to_params(x)

            if self._is_valid(p):
                configs.append(p)
        return configs

    def update(self, params: dict, result_ms: float) -> None:
        """Record observed result and update GP observations."""
        if self._fallback is not None:
            self._fallback.update(params, result_ms)
            return
        self._results.append((params, result_ms))
        try:
            vec = self._params_to_vec(params)
            self._x_obs.append(vec)
            self._y_obs.append(result_ms)
        except Exception:
            pass

    def best(self) -> Optional[dict]:
        """Return params with lowest observed latency."""
        if self._fallback is not None:
            return self._fallback.best()
        if not self._results:
            return None
        return min(self._results, key=lambda x: x[1])[0]


# ── Successive Halving ─────────────────────────────────────────────────────

class SuccessiveHalvingStrategy(SearchStrategy):
    """
    Successive Halving (SHA) bracket strategy.

    Protocol (Jamieson & Talwalkar, 2016):
      - Start with n_configs=80 random configurations
      - Run each for r_min=5 evaluations
      - Keep the top half; double the budget; repeat
      - Rounds: 80 → 40 → 20 → 10 → 5 → 1

    The "evaluation budget" is simulated here by treating each suggest()/
    update() cycle as one evaluation.  In practice each config is run in
    the benchmark harness and its latency is returned.
    """

    def __init__(
        self,
        space: dict,
        kernel: str = "",
        n_configs: int = 80,
        eta: int = 2,
    ) -> None:
        """
        Args:
            space:     Search-space dict.
            kernel:    Kernel name (for validity pruning).
            n_configs: Initial population size.
            eta:       Halving factor (default 2 → halve each round).
        """
        self._kernel  = kernel
        self._eta     = eta
        self._results: dict[str, tuple[dict, float]] = {}  # key → (params, best_ms)
        self._all_results: List[tuple[dict, float]] = []

        # Build candidate pool
        grid = GridSearchStrategy(space, kernel)
        all_configs = grid._queue  # full expanded list
        n = min(n_configs, len(all_configs))
        random.shuffle(all_configs)
        self._candidates: List[dict] = all_configs[:n]
        self._bracket: List[dict] = list(self._candidates)
        self._ptr = 0
        self._round = 0
        self._round_results: dict[str, float] = {}  # param_key → ms

    @staticmethod
    def _key(params: dict) -> str:
        """Stable string key for a param dict."""
        return "_".join(f"{k}{v}" for k, v in sorted(params.items()))

    def suggest(self, n: int = 1) -> List[dict]:
        """Return next *n* configs from the current SHA bracket."""
        if not self._bracket:
            return []
        batch = []
        while len(batch) < n and self._ptr < len(self._bracket):
            batch.append(self._bracket[self._ptr])
            self._ptr += 1
        return batch

    def update(self, params: dict, result_ms: float) -> None:
        """
        Record result and, when the current round is exhausted, halve the bracket.
        """
        key = self._key(params)
        self._round_results[key] = result_ms
        self._all_results.append((params, result_ms))

        # Track best per config (in case of multiple evaluations)
        if key not in self._results or result_ms < self._results[key][1]:
            self._results[key] = (params, result_ms)

        # When current bracket is exhausted, run SHA halving
        if self._ptr >= len(self._bracket) and self._round_results:
            self._advance_round()

    def _advance_round(self) -> None:
        """Halve the bracket, keeping top 1/eta survivors."""
        ranked = sorted(
            [(p, ms) for p, ms in [
                (params, self._round_results[self._key(params)])
                for params in self._bracket
                if self._key(params) in self._round_results
            ]],
            key=lambda x: x[1],
        )
        n_survivors = max(1, len(ranked) // self._eta)
        self._bracket = [p for p, _ in ranked[:n_survivors]]
        self._ptr = 0
        self._round += 1
        self._round_results = {}

    def best(self) -> Optional[dict]:
        """Return the params with the lowest latency observed across all rounds."""
        if not self._results:
            return None
        return min(self._results.values(), key=lambda x: x[1])[0]

    @property
    def bracket_size(self) -> int:
        """Number of configs in the current round."""
        return len(self._bracket)

    @property
    def current_round(self) -> int:
        """Current SHA round index (0-based)."""
        return self._round


# ── Factory ────────────────────────────────────────────────────────────────

def make_strategy(
    name: str,
    space: dict,
    kernel: str = "",
    n_calls: int = 40,
    n_configs: int = 80,
) -> SearchStrategy:
    """
    Instantiate a search strategy by name.

    Args:
        name:      "grid" | "bayesian" | "sha"
        space:     Search-space dict from parser.build_search_space().
        kernel:    Kernel name.
        n_calls:   For Bayesian: maximum GP evaluations.
        n_configs: For SHA: initial population size.

    Returns:
        A SearchStrategy instance ready to use.
    """
    name = name.lower()
    if name in ("grid", "gridsearch"):
        return GridSearchStrategy(space, kernel)
    if name in ("bayesian", "bayes", "bo"):
        return BayesianOptStrategy(space, kernel, n_calls=n_calls)
    if name in ("sha", "successive_halving", "halving"):
        return SuccessiveHalvingStrategy(space, kernel, n_configs=n_configs)
    raise ValueError(f"Unknown strategy {name!r}. Choose: grid | bayesian | sha")


# ── Convergence logging ────────────────────────────────────────────────────

class ConvergenceLogger:
    """
    Records eval-number → best-so-far timing for post-run convergence analysis.

    Saves to results/{kernel}_convergence.json.
    """

    def __init__(self, kernel: str, results_dir: Path) -> None:
        self._kernel   = kernel
        self._path     = results_dir / f"{kernel}_convergence.json"
        self._curve: List[dict] = []
        self._best_ms  = float("inf")
        self._eval     = 0

    def record(self, params: dict, ms: float) -> None:
        """Record one evaluation result."""
        self._eval += 1
        if ms < self._best_ms:
            self._best_ms = ms
        self._curve.append({
            "eval":      self._eval,
            "ms":        ms,
            "best_ms":   self._best_ms,
            "params":    params,
        })

    def save(self) -> None:
        """Flush convergence curve to disk."""
        with open(self._path, "w") as f:
            json.dump({
                "kernel":   self._kernel,
                "n_evals":  self._eval,
                "best_ms":  self._best_ms,
                "curve":    self._curve,
            }, f, indent=2)

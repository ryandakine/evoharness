"""Exact Shapley attribution for the explicitly zero-valued nonviable DAG game.

AAAI HiveMind, Algorithms 3-4. Enumeration remains exponential; only viable
coalitions invoke the utility callback. Node-output reuse lives in run_genome.
"""
from collections.abc import Callable
from dataclasses import dataclass
from math import comb, isfinite

from .genome import Genome


@dataclass
class Attribution:
    scores: list[float]
    values: dict[frozenset[int], float]
    evaluated: int
    pruned: int


def viable(g: Genome, coalition: set[int] | frozenset[int]) -> bool:
    """Require the original sink and an original source-to-sink path."""
    reachable = set()
    for i in range(g.n):
        if i in coalition and (not g.parents(i) or any(p in reachable for p in g.parents(i))):
            reachable.add(i)
    return g.n - 1 in reachable


def shapley(n: int, value: Callable[[frozenset[int]], float], *,
            allowed: Callable[[frozenset[int]], bool] | None = None,
            max_nodes: int = 16) -> Attribution:
    """Exact enumeration. Disallowed coalitions have value zero by definition.

    This does not assert that an arbitrary task's missing-source utility is zero.
    The node limit prevents accidental exponential API workloads.
    """
    if not 1 <= n <= max_nodes:
        raise ValueError(f"exact attribution requires 1 <= n <= {max_nodes}")
    values, evaluated = {}, 0
    for mask in range(1 << n):
        s = frozenset(i for i in range(n) if mask & (1 << i))
        if allowed is None or allowed(s):
            v = float(value(s))
            if not isfinite(v):
                raise ValueError("coalition utility must be finite")
            evaluated += 1
        else:
            v = 0.0
        values[s] = v
    scores = [0.0] * n
    for s, v in values.items():
        for i in range(n):
            if i not in s:
                scores[i] += (values[s | {i}] - v) / (n * comb(n - 1, len(s)))
    return Attribution(scores, values, evaluated, (1 << n) - evaluated)


def dag_shapley(g: Genome, value: Callable[[frozenset[int]], float], *,
                max_nodes: int = 16) -> Attribution:
    return shapley(g.n, value, allowed=lambda s: viable(g, s), max_nodes=max_nodes)

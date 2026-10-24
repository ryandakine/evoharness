"""Genome: a small DAG of agents. Node i may feed node j only if i < j; node n-1 is the output.

Nodes that cannot reach the output are pruned at run time, so structural mutation
directly changes cost.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

DEFAULT_ROLES = [
    "You are a careful solver. Work step by step, then finish with 'Answer: <final answer>'.",
    "You are a skeptical verifier. Check the inputs for errors, then finish with 'Answer: <final answer>'.",
    "You are a concise expert. Give the answer directly, finishing with 'Answer: <final answer>'.",
]


@dataclass
class Genome:
    roles: list[int]
    models: list[int]
    edges: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self):
        if not self.roles or len(self.roles) != len(self.models):
            raise ValueError("roles and models must have the same nonzero length")
        if any(i < 0 for i in self.roles + self.models):
            raise ValueError("role and model indices must be nonnegative")
        if any(not (0 <= i < j < self.n) for i, j in self.edges):
            raise ValueError("edges must satisfy 0 <= source < target < n")

    @property
    def n(self) -> int:
        return len(self.roles)

    def parents(self, j: int) -> list[int]:
        return sorted(i for (i, k) in self.edges if k == j)

    def active_nodes(self) -> list[int]:
        """Ancestors of the output node (inclusive), in topological order."""
        need, stack = {self.n - 1}, [self.n - 1]
        while stack:
            for p in self.parents(stack.pop()):
                if p not in need:
                    need.add(p)
                    stack.append(p)
        return sorted(need)

    def copy(self) -> "Genome":
        return Genome(list(self.roles), list(self.models), set(self.edges))

    def key(self) -> tuple:
        return (tuple(self.roles), tuple(self.models), tuple(sorted(self.edges)))


def random_genome(rng: random.Random, n: int, n_roles: int, n_models: int, p_edge: float = 0.35) -> Genome:
    g = Genome([rng.randrange(n_roles) for _ in range(n)], [rng.randrange(n_models) for _ in range(n)])
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                g.edges.add((i, j))
    return g


def mutate(g: Genome, rng: random.Random, n_roles: int, n_models: int, rate: float = 0.15) -> Genome:
    c = g.copy()
    for i in range(c.n):
        if rng.random() < rate:
            c.roles[i] = rng.randrange(n_roles)
        if rng.random() < rate:
            c.models[i] = rng.randrange(n_models)
    for i in range(c.n):
        for j in range(i + 1, c.n):
            if rng.random() < rate / 2:
                c.edges ^= {(i, j)}  # toggle
    return c


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    assert a.n == b.n
    roles, models, edges = [], [], set()
    for i in range(a.n):
        src = a if rng.random() < 0.5 else b
        roles.append(src.roles[i])
        models.append(src.models[i])
    for i in range(a.n):
        for j in range(i + 1, a.n):
            src = a if rng.random() < 0.5 else b
            if (i, j) in src.edges:
                edges.add((i, j))
    return Genome(roles, models, edges)

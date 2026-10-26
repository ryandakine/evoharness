"""Genome: a small DAG of agents. Node i may feed node j only if i < j; node n-1 is the output.

Nodes that cannot reach the output are pruned at run time, so structural mutation
directly changes cost.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

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
    tools: list[list[str]] | None = None

    def __post_init__(self):
        if not self.roles or len(self.roles) != len(self.models):
            raise ValueError("roles and models must have the same nonzero length")
        if self.tools is not None and len(self.tools) != len(self.roles):
            raise ValueError("tools must have the same length as roles/models if specified")
        if any(not isinstance(i, int) or i < 0 for i in self.roles + self.models):
            raise ValueError("role and model indices must be nonnegative integers")
        for e in self.edges:
            if (not isinstance(e, tuple) or len(e) != 2 or not all(isinstance(x, int) for x in e)
                    or not 0 <= e[0] < e[1] < self.n):
                raise ValueError("edges must be (source, target) pairs with 0 <= source < target < n")

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

    def copy(self) -> Genome:
        return Genome(
            list(self.roles),
            list(self.models),
            set(self.edges),
            tools=[list(t) for t in self.tools] if self.tools is not None else None,
        )

    def key(self) -> tuple:
        tools_key = tuple(tuple(t) for t in self.tools) if self.tools is not None else None
        return (tuple(self.roles), tuple(self.models), tuple(sorted(self.edges)), tools_key)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form; see from_dict."""
        d: dict[str, Any] = {
            "roles": list(self.roles),
            "models": list(self.models),
            "edges": [list(e) for e in sorted(self.edges)],
        }
        if self.tools is not None:
            d["tools"] = [list(t) for t in self.tools]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Genome:
        try:
            roles, models = list(data["roles"]), list(data["models"])
            edges = {(int(a), int(b)) for a, b in data.get("edges", [])}
            tools = [list(t) for t in data["tools"]] if "tools" in data and data["tools"] is not None else None
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("genome dict needs 'roles', 'models' and optional 'edges' "
                             "as [source, target] pairs") from exc
        return cls(roles, models, edges, tools=tools)

    def validate_against(self, n_roles: int, n_models: int) -> None:
        """Raise if any node refers to a role or model index outside the given tables."""
        if any(r >= n_roles for r in self.roles):
            raise ValueError(f"genome uses role index >= {n_roles}; the role table is too short")
        if any(m >= n_models for m in self.models):
            raise ValueError(f"genome uses model index >= {n_models}; the model table is too short")


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
    if a.n != b.n:
        raise ValueError(f"cannot cross genomes with {a.n} and {b.n} nodes")
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

"""Multi-objective GA (NSGA-II) over agent-graph genomes. All objectives are minimized."""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable
from .genome import Genome, random_genome, mutate, crossover

Objs = tuple[float, ...]


def dominates(a: Objs, b: Objs) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated_sort(objs: list[Objs]) -> list[list[int]]:
    n = len(objs)
    dom_by: list[list[int]] = [[] for _ in range(n)]
    count = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objs[i], objs[j]):
                dom_by[i].append(j)
            elif dominates(objs[j], objs[i]):
                count[i] += 1
    fronts, cur = [], [i for i in range(n) if count[i] == 0]
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in dom_by[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def crowding(objs: list[Objs], front: list[int]) -> dict[int, float]:
    d = {i: 0.0 for i in front}
    for m in range(len(objs[0])):
        s = sorted(front, key=lambda i: objs[i][m])
        d[s[0]] = d[s[-1]] = float("inf")
        span = (objs[s[-1]][m] - objs[s[0]][m]) or 1.0
        for k in range(1, len(s) - 1):
            d[s[k]] += (objs[s[k + 1]][m] - objs[s[k - 1]][m]) / span
    return d


@dataclass
class GAConfig:
    n_nodes: int = 4
    n_roles: int = 3
    n_models: int = 2
    pop_size: int = 16
    generations: int = 8
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    seed: int = 0


@dataclass
class GAResult:
    front: list[tuple[Genome, Objs]]
    history: list[dict] = field(default_factory=list)


def evolve(cfg: GAConfig, evaluate: Callable[[Genome], Objs], seeds: list[Genome] | None = None) -> GAResult:
    rng = random.Random(cfg.seed)
    memo: dict[tuple, Objs] = {}

    def ev(g: Genome) -> Objs:
        k = g.key()
        if k not in memo:
            memo[k] = evaluate(g)
        return memo[k]

    pop = list(seeds or [])
    while len(pop) < cfg.pop_size:
        pop.append(random_genome(rng, cfg.n_nodes, cfg.n_roles, cfg.n_models))
    pop = pop[: cfg.pop_size]
    history = []

    def rank_info(P: list[Genome]):
        objs = [ev(g) for g in P]
        fronts = nondominated_sort(objs)
        rank, crowd = {}, {}
        for r, f in enumerate(fronts):
            for i in f:
                rank[i] = r
            crowd.update(crowding(objs, f))
        return objs, fronts, rank, crowd

    for gen in range(cfg.generations):
        _, _, rank, crowd = rank_info(pop)

        def tourney() -> Genome:
            a, b = rng.randrange(len(pop)), rng.randrange(len(pop))
            return pop[a if (rank[a], -crowd[a]) <= (rank[b], -crowd[b]) else b]

        kids = []
        while len(kids) < cfg.pop_size:
            p1, p2 = tourney(), tourney()
            c = crossover(p1, p2, rng) if rng.random() < cfg.crossover_rate else p1.copy()
            kids.append(mutate(c, rng, cfg.n_roles, cfg.n_models, cfg.mutation_rate))

        union = pop + kids
        uobjs, ufronts, _, _ = rank_info(union)
        new: list[Genome] = []
        for f in ufronts:
            if len(new) + len(f) <= cfg.pop_size:
                new += [union[i] for i in f]
            else:
                cd = crowding(uobjs, f)
                f = sorted(f, key=lambda i: -cd[i])
                new += [union[i] for i in f[: cfg.pop_size - len(new)]]
                break
        pop = new
        history.append({"gen": gen, "best_error": min(ev(g)[0] for g in pop),
                        "min_cost": min(ev(g)[1] for g in pop), "evals": len(memo)})

    objs, fronts, _, _ = rank_info(pop)
    seen, front = set(), []
    for i in fronts[0]:
        if objs[i] not in seen:
            seen.add(objs[i])
            front.append((pop[i], objs[i]))
    return GAResult(sorted(front, key=lambda t: t[1][1]), history)

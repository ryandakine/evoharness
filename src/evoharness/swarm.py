"""Heterogeneous Swarms role search and JFK attribution (arXiv:2502.04510v2).

PSO includes normalized inertia/cognitive/social/worst-repulsion terms. This
module does NOT update LLM checkpoints: model selection and weight training
are distinct. Arbitrary continuous vectors can be searched with ParticleSwarm.
"""
from collections import Counter
from dataclasses import dataclass
from math import exp, isfinite
import random
from typing import Callable
from .genome import Genome


@dataclass
class PSOConfig:
    particles: int = 10
    iterations: int = 20
    step_length: float = 1.0
    inertia: float = 0.5
    cognitive: float = 1.0
    social: float = 1.0
    repel: float = 0.1
    seed: int = 0


class ParticleSwarm:
    """Stateful maximization PSO; no clipping of the continuous search space."""
    def __init__(self, positions: list[list[float]], cfg: PSOConfig):
        if not positions or not positions[0] or any(len(x) != len(positions[0]) for x in positions):
            raise ValueError("positions must be a nonempty rectangular matrix")
        if any(not isfinite(v) for row in positions for v in row):
            raise ValueError("positions must be finite")
        coefficients = (cfg.inertia, cfg.cognitive, cfg.social, cfg.repel)
        if any(not isfinite(c) or c < 0 for c in coefficients) or sum(coefficients) == 0:
            raise ValueError("PSO coefficients must be nonnegative with positive sum")
        if not isfinite(cfg.step_length) or cfg.step_length <= 0:
            raise ValueError("step_length must be finite and positive")
        self.cfg, self.rng = cfg, random.Random(cfg.seed)
        self.positions = [list(x) for x in positions]
        self.velocities = [[0.0] * len(x) for x in positions]
        self.personal = [list(x) for x in positions]
        self.personal_scores = [float('-inf')] * len(positions)
        self.best, self.worst = list(positions[0]), list(positions[0])
        self.best_score, self.worst_score = float('-inf'), float('inf')

    def observe(self, scores: list[float]):
        if len(scores) != len(self.positions) or not all(isfinite(s) for s in scores):
            raise ValueError("one finite score is required per particle")
        for i, (x, score) in enumerate(zip(self.positions, scores)):
            if score > self.personal_scores[i]:
                self.personal[i], self.personal_scores[i] = list(x), score
            if score > self.best_score:
                self.best, self.best_score = list(x), score
            if score < self.worst_score:
                self.worst, self.worst_score = list(x), score

    def move(self):
        if self.best_score == float('-inf'):
            raise ValueError("observe utility before moving")
        c = self.cfg
        for i, x in enumerate(self.positions):
            a, b, d, e = [self.rng.random() * v for v in (c.inertia, c.cognitive, c.social, c.repel)]
            norm = a + b + d + e
            if norm == 0:
                continue
            velocity = [(a * v + b * (p - y) + d * (g - y) - e * (w - y)) / norm
                        for y, v, p, g, w in zip(x, self.velocities[i], self.personal[i], self.best, self.worst)]
            self.velocities[i] = velocity
            self.positions[i] = [y + c.step_length * v for y, v in zip(x, velocity)]


def _top_p(items: list[int], weights: list[float], p: float, rng: random.Random) -> int:
    order = sorted(range(len(items)), key=lambda k: weights[k], reverse=True)
    total, cumulative, kept = sum(weights), 0.0, []
    for k in order:
        kept.append(k)
        cumulative += weights[k]
        if cumulative >= p * total:
            break
    return rng.choices([items[k] for k in kept], weights=[weights[k] for k in kept], k=1)[0]


def g_decode(matrix: list[list[float]], template: Genome, rng: random.Random,
             top_p: float = 0.8) -> Genome:
    """Decode by choosing a sink then attaching new nodes to existing nodes.

    Original model/role identities are preserved when relabeling to the harness's
    forward-index convention. Nonpositive row sums are floored at 1e-12 for
    degree sampling; raw entries are used for edge softmax probabilities.
    """
    n = template.n
    if not 0 < top_p <= 1 or len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("expected an n x n matrix and 0 < top_p <= 1")
    if not all(isfinite(x) for row in matrix for x in row):
        raise ValueError("adjacency entries must be finite")
    degrees = [max(1e-12, sum(row)) for row in matrix]
    remaining = list(range(n))
    sink = _top_p(remaining, [1 / d for d in degrees], top_p, rng)
    existing, edges = [sink], set()
    remaining.remove(sink)
    while remaining:
        u = _top_p(remaining, [degrees[i] for i in remaining], top_p, rng)
        peak = max(matrix[u][v] for v in existing)
        weights = [exp(matrix[u][v] - peak) for v in existing]
        denominator = sum(weights)
        for v, weight in zip(existing, weights):
            if rng.random() < weight / denominator:
                edges.add((u, v))
        existing.append(u)
        remaining.remove(u)
    order = list(reversed(existing))
    index = {old: new for new, old in enumerate(order)}
    return Genome([template.roles[i] for i in order], [template.models[i] for i in order],
                  {(index[i], index[j]) for i, j in edges})


@dataclass
class RoleSearchResult:
    genome: Genome
    utility: float
    adjacency: list[list[float]]
    history: list[float]


def search_roles(template: Genome, utility: Callable[[Genome], float],
                 cfg: PSOConfig | None = None, *, top_p: float = 0.8,
                 decode_samples: int = 1) -> RoleSearchResult:
    """Optimize continuous adjacency matrices, retaining the actual best decode.

    Utility is averaged across decode_samples. A fresh decode of the best matrix
    need not reproduce its measured utility; return the observed graph as well.
    """
    cfg = cfg or PSOConfig()
    if cfg.particles < 1 or cfg.iterations < 1 or decode_samples < 1:
        raise ValueError("particles, iterations and decode_samples must be positive")
    rng, n = random.Random(cfg.seed), template.n
    swarm = ParticleSwarm([[rng.random() for _ in range(n * n)] for _ in range(cfg.particles)], cfg)
    best_g, best_u, best_matrix, history = template.copy(), float('-inf'), [], []
    for step in range(cfg.iterations):
        scores = []
        for x in swarm.positions:
            matrix = [x[i*n:(i+1)*n] for i in range(n)]
            observations = []
            for _ in range(decode_samples):
                graph = g_decode(matrix, template, rng, top_p)
                score = float(utility(graph))
                if not isfinite(score):
                    raise ValueError("utility must be finite")
                observations.append(score)
                if score > best_u:
                    best_g, best_u, best_matrix = graph.copy(), score, [list(row) for row in matrix]
            scores.append(sum(observations) / decode_samples)
        swarm.observe(scores)
        history.append(swarm.best_score)
        if step + 1 < cfg.iterations:
            swarm.move()
    return RoleSearchResult(best_g, best_u, best_matrix, history)


def jfk_scores(g: Genome, n_models: int, utility: Callable[[Genome], float], *,
               assignments: int = 10, seed: int = 0) -> list[float | None]:
    """Frequency-weighted utility over random model assignments (Algorithm 2).

    An unsampled expert is None, not a fabricated zero score. This is attribution,
    not checkpoint adaptation or an instruction to discard unsampled experts.
    """
    if n_models < 1 or assignments < 1:
        raise ValueError("n_models and assignments must be positive")
    rng = random.Random(seed)
    totals, counts = [0.0] * n_models, [0] * n_models
    for _ in range(assignments):
        candidate = g.copy()
        candidate.models = [rng.randrange(n_models) for _ in range(g.n)]
        score = float(utility(candidate))
        if not isfinite(score):
            raise ValueError("utility must be finite")
        for i, count in Counter(candidate.models).items():
            totals[i] += count * score
            counts[i] += count
    return [total / count if count else None for total, count in zip(totals, counts)]

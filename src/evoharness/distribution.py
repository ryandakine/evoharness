"""SOHM-style Bernoulli graph distribution learning, fixed topological order.

Implements REINFORCE with a learned sigmoid baseline (Sec. 3.1.1).
This is the static Darwinian branch, not the task-conditioned GAT branch.
"""
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from math import exp, isfinite

from .genome import Genome


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-max(-40.0, min(40.0, x))))


@dataclass
class DistributionConfig:
    iterations: int = 50
    batch_size: int = 16
    learning_rate: float = 0.2
    baseline_rate: float = 0.2
    seed: int = 0


@dataclass
class GraphDistribution:
    roles: list[int]
    models: list[int]
    logits: list[float]

    def __post_init__(self):
        Genome(self.roles, self.models)
        if len(self.logits) != len(self.roles) * (len(self.roles) - 1) // 2:
            raise ValueError("one logit is required per forward edge")
        if not all(isfinite(x) for x in self.logits):
            raise ValueError("logits must be finite")

    @property
    def edges(self):
        return [(i, j) for i in range(len(self.roles)) for j in range(i + 1, len(self.roles))]

    @property
    def probabilities(self):
        return [sigmoid(x) for x in self.logits]

    def sample(self, rng: random.Random) -> Genome:
        return Genome(list(self.roles), list(self.models),
                      {e for e, p in zip(self.edges, self.probabilities) if rng.random() < p})

    def threshold(self) -> Genome:
        return Genome(list(self.roles), list(self.models),
                      {e for e, p in zip(self.edges, self.probabilities) if p > 0.5})


@dataclass
class DistributionResult:
    distribution: GraphDistribution
    best_genome: Genome
    best_utility: float
    history: list[dict] = field(default_factory=list)


def reinforce_graph(template: Genome, utility: Callable[[Genome], float],
                    cfg: DistributionConfig | None = None) -> DistributionResult:
    cfg = cfg or DistributionConfig()
    if (cfg.iterations < 1 or cfg.batch_size < 1 or
            not isfinite(cfg.learning_rate) or not isfinite(cfg.baseline_rate) or
            cfg.learning_rate <= 0 or cfg.baseline_rate <= 0):
        raise ValueError("iterations, batch size, and learning rates must be positive")
    rng = random.Random(cfg.seed)
    d = GraphDistribution(list(template.roles), list(template.models),
                          [rng.uniform(-0.1, 0.1) for _ in range(template.n * (template.n - 1) // 2)])
    w = rng.uniform(-0.1, 0.1)
    best, best_u, history = template.copy(), float('-inf'), []
    for step in range(cfg.iterations):
        probabilities, baseline = d.probabilities, sigmoid(w)
        gradients, rewards = [0.0] * len(d.logits), []
        for _ in range(cfg.batch_size):
            graph = d.sample(rng)
            reward = float(utility(graph))
            if not isfinite(reward):
                raise ValueError("utility must be finite")
            rewards.append(reward)
            if reward > best_u:
                best, best_u = graph.copy(), reward
            # Bernoulli score derivative with respect to logit is action - p.
            for k, (edge, p) in enumerate(zip(d.edges, probabilities)):
                gradients[k] += (reward - baseline) * (float(edge in graph.edges) - p)
        mean = sum(rewards) / cfg.batch_size
        d.logits = [x + cfg.learning_rate * grad / cfg.batch_size for x, grad in zip(d.logits, gradients)]
        # Use the OLD baseline for every sample above; fitting this batch first
        # would correlate baseline and sampled actions, biasing the estimate.
        w -= cfg.baseline_rate * (baseline - mean) * baseline * (1 - baseline)
        history.append(dict(step=step, mean_utility=mean, baseline=baseline, best_utility=best_u))
    return DistributionResult(d, best, best_u, history)

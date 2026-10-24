"""Fitness and compute-matched baselines."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Callable
from .executor import Task, run_genome, exact_match, extract_answer
from .genome import Genome, DEFAULT_ROLES
from .llm import Backend, ModelSpec
from .attribution import dag_shapley, Attribution


@dataclass
class Evaluator:
    tasks: list[Task]
    backend: Backend
    models: list[ModelSpec]
    roles: list[str] | None = None
    scorer: Callable[[str, str], float] = exact_match

    def __post_init__(self):
        if not self.tasks or not self.models:
            raise ValueError("tasks and models must be nonempty")
        self.roles = self.roles or DEFAULT_ROLES

    def score(self, g: Genome, sample: int = 0) -> tuple[float, float]:
        """Return (accuracy, mean cost per task)."""
        acc = cost = 0.0
        for t in self.tasks:
            r = run_genome(g, t, self.backend, self.models, self.roles, sample)
            acc += self.scorer(r.answer, t.reference)
            cost += r.cost
        n = len(self.tasks)
        return acc / n, cost / n

    def objectives(self, g: Genome) -> tuple[float, float]:
        acc, cost = self.score(g)
        return (1.0 - acc, cost)  # minimize both

    def utility(self, g: Genome, cost_weight: float = 0.0) -> float:
        """Scalar maximization objective for PSO/REINFORCE; USD coefficient is explicit."""
        if cost_weight < 0:
            raise ValueError("cost_weight must be nonnegative")
        acc, cost = self.score(g)
        return acc - cost_weight * cost

    def attribute(self, g: Genome, *, deterministic: bool = False,
                  memoize: bool = True, sample: int = 0, max_nodes: int = 16,
                  external_inputs: Callable[[Task], dict[int, str]] | None = None) -> Attribution:
        """Exact DAG-game attribution using mean task score, not trading Sharpe.

        Requires an explicit deterministic execution contract. A stochastic
        response cache freezes a sample but does not prove population exactness.
        Cache scope is this call only, so prompt/checkpoint changes cannot reuse
        stale intermediate results. Nonviable coalitions are assigned zero.
        """
        if not deterministic:
            raise ValueError("exact DAG attribution requires deterministic=True")
        cache = {} if memoize else None
        def value(coalition):
            total = 0.0
            for task in self.tasks:
                result = run_genome(g, task, self.backend, self.models, self.roles, sample,
                                    coalition=set(coalition), node_cache=cache,
                                    external_inputs=external_inputs(task) if external_inputs else None)
                total += self.scorer(result.answer, task.reference)
            return total / len(self.tasks)
        return dag_shapley(g, value, max_nodes=max_nodes)


def single_agent(model_idx: int, role_idx: int = 0) -> Genome:
    return Genome([role_idx], [model_idx], set())


def majority_vote(ev: Evaluator, model_idx: int, k: int) -> tuple[float, float]:
    """k independent samples of one model, majority answer. Returns (accuracy, mean cost)."""
    if k < 1:
        raise ValueError("k must be positive")
    acc = cost = 0.0
    for t in ev.tasks:
        answers = []
        for s in range(k):
            prompt = f"{ev.roles[0]}\n\nTask: {t.prompt}\n"
            r = ev.backend.complete(prompt, ev.models[model_idx], sample=s)
            answers.append(extract_answer(r.text))
            cost += r.cost
        acc += ev.scorer(Counter(answers).most_common(1)[0][0], t.reference)
    n = len(ev.tasks)
    return acc / n, cost / n

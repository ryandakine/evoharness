"""Fitness and compute-matched baselines."""
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import dataclass

from .attribution import Attribution, dag_shapley
from .executor import Task, exact_match, extract_answer, run_genome
from .genome import DEFAULT_ROLES, Genome
from .llm import Backend, ModelSpec


@dataclass
class TaskScore:
    """One task's result: what it scored and what it cost to get there."""

    score: float
    cost: float
    calls: int
    input_tokens: int
    output_tokens: int
    seconds: float
    answer: str


@dataclass
class Evaluator:
    """Scores genomes on a task set. ``max_workers`` > 1 runs independent agents concurrently."""

    tasks: list[Task]
    backend: Backend
    models: list[ModelSpec]
    roles: list[str] | None = None
    scorer: Callable[[str, str], float] = exact_match
    max_workers: int = 1

    def __post_init__(self):
        if not self.tasks or not self.models:
            raise ValueError("tasks and models must be nonempty")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.roles = self.roles or DEFAULT_ROLES

    def score_detail(self, g: Genome, sample: int = 0) -> list[TaskScore]:
        """One TaskScore per task, in task order. Seconds are wall-clock for the run, scoring excluded."""
        rows = []
        for t in self.tasks:
            t0 = time.perf_counter()
            r = run_genome(g, t, self.backend, self.models, self.roles, sample, max_workers=self.max_workers)
            seconds = time.perf_counter() - t0  # the run only: a judge scorer's own time is not the swarm's
            rows.append(TaskScore(self.scorer(r.answer, t.reference), r.cost, r.calls,
                                  r.input_tokens, r.output_tokens, seconds, r.answer))
        return rows

    def score(self, g: Genome, sample: int = 0) -> tuple[float, float]:
        """Return (accuracy, mean cost per task)."""
        acc = cost = 0.0
        for r in self.score_detail(g, sample):  # running totals in task order: sum() rounds differently
            acc += r.score
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
                                    external_inputs=external_inputs(task) if external_inputs else None,
                                    max_workers=self.max_workers)
                total += self.scorer(result.answer, task.reference)
            return total / len(self.tasks)
        return dag_shapley(g, value, max_nodes=max_nodes)


def single_agent(model_idx: int, role_idx: int = 0) -> Genome:
    return Genome([role_idx], [model_idx], set())


def _vote(ev: Evaluator, model_idx: int, k: int, sample_base: int,
          vote_key: Callable[[str], Hashable] | None = None) -> tuple[list[TaskScore], float]:
    """Rows per task, plus the cost totalled call by call (the order majority_vote has always used).

    vote_key=None groups raw answer strings. With a key, answers sharing a key vote together
    and the first raw answer of the largest group wins (ties: first-seen group). Blank
    replies are not dropped: they group under their own key and can win, as on the raw path.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if sample_base < 0:
        raise ValueError("sample_base must be nonnegative")
    rows, total = [], 0.0
    for t in ev.tasks:
        t0 = time.perf_counter()
        answers, cost, tin, tout = [], 0.0, 0, 0
        for s in range(sample_base, sample_base + k):
            prompt = f"{ev.roles[0]}\n\nTask: {t.prompt}\n"
            r = ev.backend.complete(prompt, ev.models[model_idx], sample=s)
            answers.append(extract_answer(r.text))
            cost, tin, tout = cost + r.cost, tin + r.input_tokens, tout + r.output_tokens
            total += r.cost
        if vote_key is None:
            winner = Counter(answers).most_common(1)[0][0]
        else:
            groups: dict[Hashable, list[str]] = {}
            for a in answers:
                groups.setdefault(vote_key(a), []).append(a)
            winner = max(groups.values(), key=len)[0]
        seconds = time.perf_counter() - t0
        rows.append(TaskScore(ev.scorer(winner, t.reference), cost, k, tin, tout, seconds, winner))
    return rows, total


def majority_vote_detail(ev: Evaluator, model_idx: int, k: int, *, sample_base: int = 0,
                         vote_key: Callable[[str], Hashable] | None = None) -> list[TaskScore]:
    """k independent samples of one model per task, majority answer; one TaskScore per task.

    Samples sample_base .. sample_base+k-1 are drawn. Give separate runs different
    bases, or a response cache will replay the same votes.
    """
    return _vote(ev, model_idx, k, sample_base, vote_key)[0]


def majority_vote(ev: Evaluator, model_idx: int, k: int, *, sample_base: int = 0,
                  vote_key: Callable[[str], Hashable] | None = None) -> tuple[float, float]:
    """k independent samples of one model, majority answer. Returns (accuracy, mean cost)."""
    rows, total = _vote(ev, model_idx, k, sample_base, vote_key)
    acc = 0.0
    for r in rows:
        acc += r.score
    return acc / len(rows), total / len(rows)

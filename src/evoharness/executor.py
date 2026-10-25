"""Run a genome on a task and report the answer plus nominal cost."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .genome import Genome
from .llm import Backend, BudgetExceeded, ModelSpec, Response


@dataclass
class Task:
    prompt: str
    reference: str


@dataclass
class RunResult:
    answer: str
    cost: float
    calls: int
    outputs: dict[int, str] = field(default_factory=dict)


class AgentError(RuntimeError):
    """A node's model call failed. ``node`` and ``model_name`` say which one."""

    def __init__(self, node: int, model_name: str, cause: BaseException):
        super().__init__(f"agent {node} ({model_name}) failed: {cause}")
        self.node, self.model_name = node, model_name


_ANSWER = re.compile(r"(?im)\b(?:final\s+)?answer\s*[*_`]*\s*[:：]\s*(.+?)\s*$")
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_WRAPPERS = " \t*_`\"'"


def extract_answer(text: str) -> str:
    """Pull the final answer out of a model reply.

    Looks for the last ``Answer:`` / ``Final Answer:`` line (any case, markdown
    emphasis tolerated), then the last LaTeX ``\\boxed{}``, and otherwise falls
    back to the last non-empty line.
    """
    found = _ANSWER.findall(text)
    if found:
        return found[-1].strip(_WRAPPERS).strip()
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip(_WRAPPERS).strip() if lines else ""


def exact_match(answer: str, reference: str) -> float:
    def norm(s: str) -> str:
        return re.sub(r"\W+", " ", s.lower()).strip()
    return float(norm(answer) == norm(reference))


def _layers(g: Genome, need: set[int]) -> list[list[int]]:
    """Group the needed nodes so that every node's included parents sit in an earlier layer."""
    level: dict[int, int] = {}
    for i in sorted(need):
        level[i] = 1 + max((level[p] for p in g.parents(i) if p in need), default=0)
    layers: dict[int, list[int]] = {}
    for i, lv in level.items():
        layers.setdefault(lv, []).append(i)
    return [sorted(layers[lv]) for lv in sorted(layers)]


def run_genome(g: Genome, task: Task, backend: Backend, models: list[ModelSpec],
               roles: list[str], sample: int = 0, *, coalition: set[int] | None = None,
               node_cache: dict | None = None,
               external_inputs: dict[int, str] | None = None,
               max_workers: int = 1) -> RunResult:
    """Execute an induced coalition; optional memoization assumes deterministic calls.

    external_inputs restricts raw task data to explicitly listed nodes. Without it,
    all nodes receive the task (the original general-purpose behavior).
    Cache scope must be one fixed backend/checkpoint configuration.

    max_workers > 1 runs independent nodes (those with no path between them) in
    parallel threads, one layer of the DAG at a time. Results, costs and call
    counts are identical to the sequential run; the backend must be thread-safe.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    g.validate_against(len(roles), len(models))
    included = set(range(g.n)) if coalition is None else set(coalition)
    if not included <= set(range(g.n)):
        raise ValueError("coalition contains unknown nodes")
    if g.n - 1 not in included:
        return RunResult("", 0.0, 0)
    need, stack = {g.n - 1}, [g.n - 1]
    while stack:
        for p in g.parents(stack.pop()):
            if p in included and p not in need:
                need.add(p)
                stack.append(p)
    outputs: dict[int, str] = {}
    cost, calls = 0.0, 0

    def call(job: tuple[int, str, ModelSpec]) -> Response:
        i, prompt, model = job
        try:
            return backend.complete(prompt, model, sample)
        except BudgetExceeded:
            raise
        except Exception as exc:
            raise AgentError(i, model.name, exc) from exc

    for layer in _layers(g, need):
        jobs: list[tuple[int, str, ModelSpec]] = []
        for i in layer:
            ctx = "".join(f"\n[input from agent {p}]: {outputs[p]}" for p in g.parents(i) if p in outputs)
            raw = task.prompt if external_inputs is None else external_inputs.get(i, "")
            prompt = f"{roles[g.roles[i]]}\n\nTask: {raw}\n{ctx}"
            model = models[g.models[i]]
            key = (i, model, sample, prompt)
            if node_cache is not None and key in node_cache:
                r = node_cache[key]
                outputs[i], cost = r.text, cost + r.cost
            else:
                jobs.append((i, prompt, model))
        if len(jobs) > 1 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
                results = list(pool.map(call, jobs))
        else:
            results = [call(job) for job in jobs]
        for (i, prompt, model), r in zip(jobs, results):
            calls += 1
            if node_cache is not None:
                node_cache[(i, model, sample, prompt)] = r
            outputs[i], cost = r.text, cost + r.cost
    outputs = dict(sorted(outputs.items()))
    return RunResult(extract_answer(outputs[g.n - 1]), cost, calls, outputs)

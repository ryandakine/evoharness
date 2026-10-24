"""Run a genome on a task and report the answer plus nominal cost."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .genome import Genome
from .llm import Backend, ModelSpec


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


def extract_answer(text: str) -> str:
    m = re.findall(r"Answer:\s*(.+)", text)
    if m:
        return m[-1].strip()
    lines = text.strip().splitlines()
    return lines[-1].strip() if lines else ""


def exact_match(answer: str, reference: str) -> float:
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    return float(norm(answer) == norm(reference))


def run_genome(g: Genome, task: Task, backend: Backend, models: list[ModelSpec],
               roles: list[str], sample: int = 0, *, coalition: set[int] | None = None,
               node_cache: dict | None = None,
               external_inputs: dict[int, str] | None = None) -> RunResult:
    """Execute an induced coalition; optional memoization assumes deterministic calls.

    external_inputs restricts raw task data to explicitly listed nodes. Without it,
    all nodes receive the task (the original general-purpose behavior).
    Cache scope must be one fixed backend/checkpoint configuration.
    """
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
    for i in sorted(need):
        ctx = "".join(f"\n[input from agent {p}]: {outputs[p]}" for p in g.parents(i) if p in outputs)
        raw = task.prompt if external_inputs is None else external_inputs.get(i, "")
        prompt = f"{roles[g.roles[i]]}\n\nTask: {raw}\n{ctx}"
        model = models[g.models[i]]
        key = (i, model, sample, prompt)
        if node_cache is not None and key in node_cache:
            r = node_cache[key]
        else:
            r = backend.complete(prompt, model, sample)
            calls += 1
            if node_cache is not None:
                node_cache[key] = r
        outputs[i] = r.text
        cost += r.cost
    return RunResult(extract_answer(outputs[g.n - 1]), cost, calls, outputs)

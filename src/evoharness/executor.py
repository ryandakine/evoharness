"""Run a genome on a task and report the answer plus nominal cost."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .genome import Genome
from .llm import Backend, BudgetExceeded, ModelSpec, Response
from .tools.base import ExecutionContext, ToolRegistry, ToolResult

logger = logging.getLogger("evoharness.executor")


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
    llm_calls: int = 0
    tool_calls: int = 0


class AgentError(RuntimeError):
    """A node's model call failed. ``node`` and ``model_name`` say which one."""

    def __init__(self, node: int, model_name: str, cause: BaseException):
        super().__init__(f"agent {node} ({model_name}) failed: {cause}")
        self.node, self.model_name = node, model_name


_ANSWER = re.compile(r"(?im)\b(?:final\s+)?answer\s*[*_`]*\s*[:：]\s*(.+?)\s*$")
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_WRAPPERS = " \t*_`\"'"


def extract_answer(text: str) -> str:
    """Pull the final answer out of a model reply."""
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


def _log_audit(
    audit_file: Path | str | None,
    node: int,
    model: str,
    tool_name: str,
    args: dict[str, Any],
    res: ToolResult,
    duration: float,
) -> None:
    if not audit_file:
        return
    path = Path(audit_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "node": node,
            "model": model,
            "tool": tool_name,
            "args": args,
            "error": res.error,
            "is_dry_run": res.is_dry_run,
            "duration_s": round(duration, 4),
            "output_hash": hashlib.sha256(res.output.encode("utf-8")).hexdigest()[:16],
            "output_len": len(res.output),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as exc:
        logger.warning("Failed to write tool audit log: %s", exc)


def run_react_node(
    node: int,
    prompt: str,
    model: ModelSpec,
    sample: int,
    backend: Backend,
    tool_registry: ToolRegistry,
    allowed_tools: list[str] | None,
    context: ExecutionContext,
    max_tool_turns: int = 10,
    on_tool_call: Callable[[int, str, dict[str, Any], ToolResult], None] | None = None,
    audit_file: Path | str | None = None,
) -> Response:
    """Execute a multi-turn ReAct loop for a node with tools."""
    all_tools = tool_registry.list_tools()
    if allowed_tools is not None:
        active_tools = [t for t in all_tools if t.name in allowed_tools]
    else:
        active_tools = all_tools

    if not active_tools or not hasattr(backend, "complete_messages"):
        return backend.complete(prompt, model, sample)

    litellm_tools = [t.to_litellm_tool() for t in active_tools]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": prompt},
    ]

    total_cost = 0.0
    total_tin = 0
    total_tout = 0
    turn = 0
    last_resp: Response | None = None

    while turn < max_tool_turns:
        resp = backend.complete_messages(messages, model, sample, tools=litellm_tools)
        total_cost += resp.cost
        total_tin += resp.input_tokens
        total_tout += resp.output_tokens
        last_resp = resp

        if not resp.tool_calls:
            return Response(resp.text, total_tin, total_tout, total_cost)

        # Handle parallel tool calls
        if turn == max_tool_turns - 1:
            # Reached max turns: force final synthesis without invoking tools
            messages.append({"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls})
            for tc in resp.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Turn limit reached. Synthesize your final answer now based on current findings.",
                })
            synth_resp = backend.complete_messages(messages, model, sample, tool_choice="none")
            total_cost += synth_resp.cost
            total_tin += synth_resp.input_tokens
            total_tout += synth_resp.output_tokens
            return Response(synth_resp.text, total_tin, total_tout, total_cost)

        messages.append({"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls})

        for tc in resp.tool_calls:
            tool_name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", "{}")
            parse_error = None
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(args, dict):
                    args = {}
            except Exception as exc:
                args = {}
                parse_error = str(exc)

            t0 = time.time()
            if parse_error:
                res = ToolResult(
                    output=f"Error: malformed JSON arguments for tool '{tool_name}': {parse_error}. Please provide valid JSON parameters.",
                    error=True,
                )
            else:
                res = tool_registry.execute(tool_name, args, context)
            duration = time.time() - t0

            if on_tool_call:
                try:
                    on_tool_call(node, tool_name, args, res)
                except Exception:
                    pass

            _log_audit(audit_file, node, model.name, tool_name, args, res, duration)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": res.output,
            })

        turn += 1

    return Response(last_resp.text if last_resp else "", total_tin, total_tout, total_cost)


def run_genome(
    g: Genome,
    task: Task,
    backend: Backend,
    models: list[ModelSpec],
    roles: list[str],
    sample: int = 0,
    *,
    coalition: set[int] | None = None,
    node_cache: dict | None = None,
    external_inputs: dict[int, str] | None = None,
    max_workers: int = 1,
    tool_registry: ToolRegistry | None = None,
    node_tools: dict[int, list[str]] | None = None,
    context: ExecutionContext | None = None,
    max_tool_turns: int = 10,
    on_tool_call: Callable[[int, str, dict[str, Any], ToolResult], None] | None = None,
    audit_file: Path | str | None = None,
) -> RunResult:
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
    ctx_env = context or ExecutionContext()

    def call(job: tuple[int, str, ModelSpec]) -> Response:
        i, prompt, model = job
        try:
            if tool_registry is not None and (node_tools is None or i in node_tools):
                allowed = node_tools.get(i) if node_tools else None
                return run_react_node(
                    node=i,
                    prompt=prompt,
                    model=model,
                    sample=sample,
                    backend=backend,
                    tool_registry=tool_registry,
                    allowed_tools=allowed,
                    context=ctx_env,
                    max_tool_turns=max_tool_turns,
                    on_tool_call=on_tool_call,
                    audit_file=audit_file,
                )
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

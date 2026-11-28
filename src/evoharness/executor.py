"""Run a genome on a task and report the answer plus nominal cost."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from .genome import Genome
from .llm import Backend, BudgetExceeded, ModelSpec, Response
from .tools.base import SENSITIVE_ENV_PATTERN, ExecutionContext, ToolRegistry, ToolResult, redact_sensitive_text

logger = logging.getLogger("evoharness.executor")
_AUDIT_LOCK = threading.Lock()


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
    # Nominal totals, like cost: a node served from node_cache adds its tokens and cost
    # here but not to calls, which counts only the calls this run actually made.
    input_tokens: int = 0
    output_tokens: int = 0


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


_PLAIN = r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+"
# A sign counts only at the start of a number: after a digit, letter, '.' or '^' it is a
# range, a hyphen or an exponent ('3-5' ends in 5; the 2 in 'm^2' is not an answer).
_NUMBER = re.compile(
    r"(?<![\w.^])(?P<sign>[-+]?)\$?(?:"
    r"(?P<whole>\d+)\s+(?P<mnum>\d+)/(?P<mden>\d+)"   # mixed fraction: 2 1/2
    r"|(?P<num>\d+)/(?P<den>\d+)"                      # simple fraction: 1/2
    rf"|(?P<plain>{_PLAIN})"
    r")"
)


_PLACES = 6                    # answers are compared at 6 decimal places
_SCALE = 10 ** _PLACES
_MAX_DIGITS = 309              # input limit: longest digit run accepted, for every spelling
_MAX_ADJUSTED = _MAX_DIGITS - 1  # magnitude limit: |value| < 10**309


def canonical_number(text: str) -> Hashable | None:
    """One exact value for the last number in the text, or None. Equal answers compare with ==.

    Supported grammar (the _NUMBER regex, unchanged): optional sign (ASCII or U+2212) at the
    start of a number, optional '$', then a mixed fraction '2 1/2', a fraction '1/2', or a
    decimal with thousands commas ('1,234.5') and an optional exponent ('1e-3'). The LAST
    token wins, and if it is invalid the result is None; there is no fallback to an earlier
    number. Known parser defects, pinned in tests and tracked in Taskmaster: '.5e-3' reads 3,
    '.5e3' reads 0.5, '1/-2' reads -2, '1,234 1/2' reads 0.5, '1,2' reads 2.

    Value: integer text becomes an exact int. Anything else is an exact Fraction rounded
    half-even to 6 places (no float involved); a whole result is returned as int, so
    '18.000000' is 18 and '-0.0000004' is 0. Policy: 0.1 matches 0.1000004, +/-0.0000004
    share the zero bucket, and values within 1e-6 can still land in different buckets
    (0.3333334 vs 0.3333335 differ). Integers are never rounded.

    Size policy, the same for integer, decimal, scientific and fraction spellings: every
    digit run (commas removed) may have at most 309 digits, checked before any conversion,
    and the rounded value must satisfy |value| < 10**309. Otherwise None. A nonzero value
    below 1e-7 is 0, reached without building the huge Fraction.
    """
    last = deque(_NUMBER.finditer(text.replace("\u2212", "-")), maxlen=1)
    if not last:
        return None
    m = last[0]
    neg = m["sign"] == "-"
    try:
        if m["whole"] is not None:
            if max(len(m["whole"]), len(m["mnum"]), len(m["mden"])) > _MAX_DIGITS:
                return None
            q = Fraction(int(m["whole"])) + Fraction(int(m["mnum"]), int(m["mden"]))
        elif m["num"] is not None:
            if max(len(m["num"]), len(m["den"])) > _MAX_DIGITS:
                return None
            q = Fraction(int(m["num"]), int(m["den"]))
        else:
            coef, _, exp = m["plain"].replace(",", "").lower().partition("e")
            whole, _, frac = coef.partition(".")
            if max(len(whole), len(frac), len(exp.lstrip("+-"))) > _MAX_DIGITS:
                return None
            if not frac and "." not in coef and not exp:
                v = int(whole)
                return -v if neg else v
            d = Decimal(coef)
            e = int(exp) if exp else 0
            if d == 0:
                q = Fraction(0)
            elif d.adjusted() + e > _MAX_ADJUSTED:
                return None
            elif d.adjusted() + e < -(_PLACES + 1):
                q = Fraction(0)
            else:
                q = Fraction(d) * Fraction(10) ** e
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None
    n = round((-q if neg else q) * _SCALE)
    if abs(n) >= 10 ** _MAX_DIGITS * _SCALE:
        return None
    return n // _SCALE if n % _SCALE == 0 else Fraction(n, _SCALE)


def numeric_match(answer: str, reference: str) -> float:
    """1.0 when the last number in each string has the same canonical_number, else 0.0.

    '$18.00' matches '18', '1/2' matches '0.5'. A string with no valid number never matches,
    not even itself, so text benchmarks need exact_match as the scorer. The relation is an
    equivalence on numeric answers and agrees with numeric_key. Known limits: the LAST
    number wins ('$18 for 3 days' reads as 3), '3:30' reads as 30, '50%' is 50, not 0.5.
    Pass the extracted answer, not a whole reply.
    """
    a, r = canonical_number(answer), canonical_number(reference)
    return float(a is not None and r is not None and a == r)


def numeric_key(s: str) -> Hashable:
    """Vote-grouping key: canonical_number(s), or the stripped raw text when there is none.

    For any fixed reference, answers sharing a key get the same numeric_match score. That does
    not make the winner right: blank or invalid answers share a key with each other and score
    0 against every reference. A numeric and a raw-text key never collide (number vs str).
    """
    c = canonical_number(s)
    return c if c is not None else s.strip()


def _layers(g: Genome, need: set[int]) -> list[list[int]]:
    """Group the needed nodes so that every node's included parents sit in an earlier layer."""
    level: dict[int, int] = {}
    for i in sorted(need):
        level[i] = 1 + max((level[p] for p in g.parents(i) if p in need), default=0)
    layers: dict[int, list[int]] = {}
    for i, lv in level.items():
        layers.setdefault(lv, []).append(i)
    return [sorted(layers[lv]) for lv in sorted(layers)]


def _sanitize_audit_args(obj: Any, env: dict[str, str] | None = None) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if SENSITIVE_ENV_PATTERN.search(str(k)):
                out[k] = "[REDACTED]"
            else:
                out[k] = _sanitize_audit_args(v, env)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize_audit_args(v, env) for v in obj]
    if isinstance(obj, str):
        return redact_sensitive_text(obj, env)
    return obj

def _log_audit(
    audit_file: Path | str | None,
    node: int,
    model: str,
    tool_name: str,
    args: dict[str, Any],
    res: ToolResult,
    duration: float,
    env: dict[str, str] | None = None,
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
            "args": _sanitize_audit_args(args, env),
            "error": res.error,
            "is_dry_run": res.is_dry_run,
            "duration_s": round(duration, 4),
            "output_hash": hashlib.sha256(res.output.encode("utf-8")).hexdigest()[:16],
            "output_len": len(res.output),
        }
        with _AUDIT_LOCK:
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
    all_tool_calls: list[dict[str, Any]] = []
    turn = 0
    last_resp: Response | None = None

    while turn < max_tool_turns:
        resp = backend.complete_messages(messages, model, sample, tools=litellm_tools)
        total_cost += resp.cost
        total_tin += resp.input_tokens
        total_tout += resp.output_tokens
        last_resp = resp

        if not resp.tool_calls:
            return Response(resp.text, total_tin, total_tout, total_cost, tool_calls=all_tool_calls)

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
            return Response(synth_resp.text, total_tin, total_tout, total_cost, tool_calls=all_tool_calls)

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
                    output=(
                        f"Error: malformed JSON arguments for tool '{tool_name}': {parse_error}. "
                        "Please provide valid JSON parameters."
                    ),
                    error=True,
                )
            elif allowed_tools is not None and tool_name not in allowed_tools:
                res = ToolResult(
                    output=(
                        f"Error: tool '{tool_name}' is not permitted for this agent node. "
                        f"Allowed tools: {sorted(allowed_tools)}"
                    ),
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

            _log_audit(audit_file, node, model.name, tool_name, args, res, duration, env=context.env)
            all_tool_calls.append(tc)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": res.output,
            })

        turn += 1

    return Response(last_resp.text if last_resp else "", total_tin, total_tout, total_cost, tool_calls=all_tool_calls)


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
    cost, calls, total_tool_calls = 0.0, 0, 0
    tokens_in = tokens_out = 0
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
        mutating_nodes: set[int] = set()
        if tool_registry is not None:
            for i in layer:
                allowed_names = node_tools.get(i) if node_tools is not None else None
                if allowed_names is not None:
                    tools_to_check = [tool_registry.get(t) for t in allowed_names if tool_registry.get(t) is not None]
                else:
                    tools_to_check = tool_registry.list_tools()
                if any(t.is_mutating for t in tools_to_check):
                    mutating_nodes.add(i)

        jobs: list[tuple[int, str, ModelSpec]] = []
        for i in layer:
            ctx = "".join(f"\n[input from agent {p}]: {outputs[p]}" for p in g.parents(i) if p in outputs)
            raw = task.prompt if external_inputs is None else external_inputs.get(i, "")
            prompt = f"{roles[g.roles[i]]}\n\nTask: {raw}\n{ctx}"
            model = models[g.models[i]]

            key = (i, model, sample, prompt)
            if node_cache is not None and i not in mutating_nodes and key in node_cache:
                r = node_cache[key]
                outputs[i], cost = r.text, cost + r.cost
                total_tool_calls += len(r.tool_calls)
                tokens_in, tokens_out = tokens_in + r.input_tokens, tokens_out + r.output_tokens
            else:
                jobs.append((i, prompt, model))
        if len(jobs) > 1 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
                results = list(pool.map(call, jobs))
        else:
            results = [call(job) for job in jobs]
        for (i, prompt, model), r in zip(jobs, results):
            calls += 1
            total_tool_calls += len(r.tool_calls)
            tokens_in, tokens_out = tokens_in + r.input_tokens, tokens_out + r.output_tokens
            if node_cache is not None and i not in mutating_nodes:
                node_cache[(i, model, sample, prompt)] = r
            outputs[i], cost = r.text, cost + r.cost
    outputs = dict(sorted(outputs.items()))
    return RunResult(
        extract_answer(outputs[g.n - 1]), cost, calls, outputs,
        llm_calls=calls, tool_calls=total_tool_calls,
        input_tokens=tokens_in, output_tokens=tokens_out,
    )

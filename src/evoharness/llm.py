"""LLM backends, response cache, spend budget, and tool-calling support."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelSpec:
    name: str
    price_in: float   # USD per 1M input tokens (you supply this; check provider pricing)
    price_out: float  # USD per 1M output tokens
    api_base: str | None = None
    api_key_env: str | None = None  # environment variable NAME, never the secret


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class Backend(Protocol):
    def complete(self, prompt: str, model: ModelSpec, sample: int = 0) -> Response: ...


def price(model: ModelSpec, tin: int, tout: int) -> float:
    return (tin * model.price_in + tout * model.price_out) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


class EmptyResponse(ValueError):
    """The provider answered, but with no usable text. Often transient on shared/free routers."""


class CachedBackend:
    """Wraps a backend with a response cache and a recorded-spend cutoff.

    Cache hits are free for the budget but still report their nominal cost, so
    fitness always reflects what a deployed swarm would cost. The key covers the
    model name, endpoint, sample id and prompt; prices are deliberately excluded,
    so correcting a price re-costs cached responses from their token counts
    instead of re-spending on the API.

    Pass ``cache_path`` to persist responses as JSON lines across runs. The
    cache does not know about generation settings (temperature, max_tokens);
    use a different path when you change them.

    One call may overshoot the cutoff before subsequent calls are blocked.
    Safe to share between threads.
    """

    def __init__(self, inner: Backend, budget_usd: float | None = None,
                 cache_path: str | Path | None = None):
        self.inner, self.budget, self.spent = inner, budget_usd, 0.0
        self._cache: dict[str, tuple[Response, float, float]] = {}
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        self._path = Path(cache_path) if cache_path is not None else None
        if self._path is not None:
            self._load()

    @staticmethod
    def key(prompt: str, model: ModelSpec, sample: int) -> str:
        raw = f"{model.name}|{model.api_base}|{sample}|{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                    r = Response(
                        e["text"],
                        int(e["input_tokens"]),
                        int(e["output_tokens"]),
                        float(e["cost"]),
                        tool_calls=e.get("tool_calls", []),
                    )
                    self._cache[e["key"]] = (r, float(e["price_in"]), float(e["price_out"]))
                except (ValueError, KeyError, TypeError):
                    continue  # tolerate a truncated or foreign line

    def _persist(self, key: str, r: Response, model: ModelSpec) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(key=key, text=r.text, input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                     cost=r.cost, price_in=model.price_in, price_out=model.price_out,
                     tool_calls=r.tool_calls)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def __len__(self) -> int:
        return len(self._cache)

    def complete(self, prompt: str, model: ModelSpec, sample: int = 0) -> Response:
        key = self.key(prompt, model, sample)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self.hits += 1
                r, pin, pout = hit
                if (pin, pout) != (model.price_in, model.price_out):
                    r = Response(r.text, r.input_tokens, r.output_tokens,
                                 price(model, r.input_tokens, r.output_tokens),
                                 tool_calls=r.tool_calls)
                return r
            if self.budget is not None and self.spent >= self.budget:
                raise BudgetExceeded(f"spent ${self.spent:.4f} >= cap ${self.budget:.4f}")
        r = self.inner.complete(prompt, model, sample)
        with self._lock:
            self.spent += r.cost
            self.misses += 1
            self._cache[key] = (r, model.price_in, model.price_out)
            self._persist(key, r, model)
        return r

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        model: ModelSpec,
        sample: int = 0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Response:
        if hasattr(self.inner, "complete_messages"):
            return self.inner.complete_messages(
                messages, model, sample, tools=tools, tool_choice=tool_choice
            )
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("content"))
        return self.complete(prompt, model, sample)


_TRANSIENT = {"RateLimitError", "RouterRateLimitError", "Timeout", "APITimeoutError", "APIConnectionError",
              "ServiceUnavailableError", "InternalServerError", "BadGatewayError"}


def is_transient(exc: BaseException) -> bool:
    """Heuristic: provider errors that usually succeed on retry (rate limits, timeouts, 5xx)."""
    if isinstance(exc, (TimeoutError, ConnectionError, EmptyResponse)):
        return True
    if {c.__name__ for c in type(exc).__mro__} & _TRANSIENT:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status in (408, 409, 425, 429) or status >= 500)


class APIBackend:
    """Provider-neutral text completions through the optional LiteLLM SDK."""

    def __init__(self, *, api_base: str | None = None, api_key: str | None = None,
                 max_tokens: int | None = 512, temperature: float | None = 0.7,
                 timeout: float = 60.0, extra_params: dict | None = None,
                 retries: int = 2, retry_backoff: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep):
        try:
            litellm = importlib.import_module("litellm")
        except ModuleNotFoundError as exc:
            if exc.name != "litellm":
                raise
            raise ImportError('APIBackend requires: pip install "evoharness[api]"') from exc
        self._completion = litellm.completion
        if hasattr(litellm, "suppress_debug_info"):
            litellm.suppress_debug_info = True  # drop the "Give Feedback / Get Help" banner on errors
        if retries < 0 or retry_backoff < 0:
            raise ValueError("retries and retry_backoff must be nonnegative")
        self.api_base, self._api_key = api_base, api_key
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.retries, self.retry_backoff, self._sleep = retries, retry_backoff, sleep
        self.extra_params = dict(extra_params or {})
        protected = {"model", "messages", "stream", "api_base", "api_key", "timeout", "num_retries"}
        if protected & self.extra_params.keys():
            raise ValueError("extra_params cannot override routing, credentials, messages, "
                             "streaming, timeout, or retries")

    def _params(
        self,
        messages: list[dict[str, Any]],
        model: ModelSpec,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict:
        params = dict(self.extra_params)
        params.update(model=model.name, messages=messages,
                      stream=False, timeout=self.timeout, num_retries=0)
        if tools:
            params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
        if self.max_tokens is not None:
            params.setdefault("max_tokens", self.max_tokens)
        if self.temperature is not None:
            params.setdefault("temperature", self.temperature)
        api_base = model.api_base if model.api_base is not None else self.api_base
        if api_base is not None:
            params["api_base"] = api_base
        if model.api_key_env is not None:
            api_key = os.environ.get(model.api_key_env)
            if not api_key:
                raise ValueError(f"missing credential environment variable: {model.api_key_env}")
        elif model.api_base is not None and model.api_base != self.api_base:
            api_key = None
        else:
            api_key = self._api_key
        if api_key is not None:
            params["api_key"] = api_key
        return params

    def _once(self, params: dict, model: ModelSpec) -> Response:
        result = self._completion(**params)
        if not getattr(result, "choices", None):
            raise EmptyResponse("model API returned no completion choices")
        msg = result.choices[0].message
        content = msg.content
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        text = content if isinstance(content, str) else ""

        tool_calls: list[dict[str, Any]] = []
        raw_tc = getattr(msg, "tool_calls", None)
        if raw_tc:
            for tc in raw_tc:
                fn = getattr(tc, "function", None)
                fn_name = getattr(fn, "name", "") if fn else ""
                fn_args = getattr(fn, "arguments", "{}") if fn else "{}"
                tc_id = getattr(tc, "id", "") or f"call_{len(tool_calls)}"
                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    },
                })

        if not text.strip() and not tool_calls:
            raise EmptyResponse("model API returned no text or tool calls; choose a text-capable model "
                                "or increase its output limit")
        usage = getattr(result, "usage", None)
        tin, tout = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
        if not isinstance(tin, int) or not isinstance(tout, int) or tin < 0 or tout < 0:
            raise ValueError("model API omitted valid token usage; cannot report inference cost")
        return Response(text, tin, tout, price(model, tin, tout), tool_calls=tool_calls)

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        model: ModelSpec,
        sample: int = 0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Response:
        params = self._params(messages, model, tools=tools, tool_choice=tool_choice)
        attempt = 0
        while True:
            try:
                return self._once(params, model)
            except Exception as exc:
                if attempt >= self.retries or not is_transient(exc):
                    raise
                self._sleep(self.retry_backoff * (2 ** attempt))
                attempt += 1

    def complete(self, prompt: str, model: ModelSpec, sample: int = 0) -> Response:
        return self.complete_messages([{"role": "user", "content": prompt}], model, sample)


class MockBackend:
    """Deterministic offline backend for tests and demos. NOT a model of real LLM behavior."""

    def __init__(self, answers: dict[str, str], base: list[float], input_bonus: float = 0.06,
                 model_names: list[str] | None = None):
        if not base:
            raise ValueError("base must list at least one probability")
        self.answers, self.base, self.bonus = answers, base, input_bonus
        self.models: list[str] = list(model_names or [])
        self._lock = threading.Lock()

    def complete(self, prompt: str, model: ModelSpec, sample: int = 0) -> Response:
        with self._lock:
            if model.name not in self.models:
                self.models.append(model.name)
            idx = self.models.index(model.name)
        ref = next((a for q, a in self.answers.items() if f"Task: {q}\n" in prompt), "?")
        n_in = min(prompt.count("[input from agent"), 3)
        p = min(0.98, self.base[idx % len(self.base)] + self.bonus * n_in)
        h = int(hashlib.sha256(f"{prompt}|{model.name}|{sample}".encode()).hexdigest(), 16)
        ok = (h % 10_000) / 10_000 < p
        text = f"Answer: {ref if ok else 'wrong'}"
        tin, tout = len(prompt) // 4, len(text) // 4 + 1
        return Response(text, tin, tout, price(model, tin, tout))

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        model: ModelSpec,
        sample: int = 0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Response:
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("content"))
        return self.complete(prompt, model, sample)

"""LLM backends, response cache, and spend budget."""
from __future__ import annotations
import hashlib
import importlib
import os
from dataclasses import dataclass
from typing import Protocol


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


class Backend(Protocol):
    def complete(self, prompt: str, model: ModelSpec, sample: int = 0) -> Response: ...


def price(model: ModelSpec, tin: int, tout: int) -> float:
    return (tin * model.price_in + tout * model.price_out) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


class CachedBackend:
    """Wraps a backend with a response cache and a recorded-spend cutoff.

    Cache hits are free for the budget but still report their nominal cost, so
    fitness always reflects what a deployed swarm would cost.
    One call may overshoot the cutoff before subsequent calls are blocked.
    """

    def __init__(self, inner: Backend, budget_usd: float | None = None):
        self.inner, self.budget, self.spent = inner, budget_usd, 0.0
        self._cache: dict[str, Response] = {}
        self.hits = self.misses = 0

    def complete(self, prompt: str, model: ModelSpec, sample: int = 0) -> Response:
        key = hashlib.sha256(f"{model!r}|{sample}|{prompt}".encode()).hexdigest()
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        if self.budget is not None and self.spent >= self.budget:
            raise BudgetExceeded(f"spent ${self.spent:.4f} >= cap ${self.budget:.4f}")
        r = self.inner.complete(prompt, model, sample)
        self.spent += r.cost
        self.misses += 1
        self._cache[key] = r
        return r


class APIBackend:
    """Provider-neutral text completions through the optional LiteLLM SDK.

    Use provider/model names, or openai/<model-id> with a compatible api_base.
    Credentials fall back to LiteLLM's native provider environment discovery.
    Per-model routes override backend defaults. sample is a cache discriminator,
    not a portable provider seed. Missing usage is rejected to avoid false costs.
    """

    def __init__(self, *, api_base: str | None = None, api_key: str | None = None,
                 max_tokens: int | None = 512, temperature: float | None = 0.7,
                 timeout: float = 60.0, extra_params: dict | None = None):
        try:
            self._completion = importlib.import_module("litellm").completion
        except ModuleNotFoundError as exc:
            if exc.name != "litellm":
                raise
            raise ImportError('APIBackend requires: pip install "evoharness[api]"') from exc
        self.api_base, self._api_key = api_base, api_key
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.extra_params = dict(extra_params or {})
        protected = {"model", "messages", "stream", "api_base", "api_key", "timeout", "num_retries"}
        if protected & self.extra_params.keys():
            raise ValueError("extra_params cannot override routing, credentials, messages, streaming, timeout, or retries")

    def complete(self, prompt: str, model: ModelSpec, sample: int = 0) -> Response:
        params = dict(self.extra_params)
        params.update(model=model.name, messages=[{"role": "user", "content": prompt}],
                      stream=False, timeout=self.timeout, num_retries=0)
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
            # A new per-model endpoint must not inherit a different endpoint's
            # backend credential. Native provider discovery may still apply.
            api_key = None
        else:
            api_key = self._api_key
        if api_key is not None:
            params["api_key"] = api_key
        result = self._completion(**params)
        if not result.choices:
            raise ValueError("model API returned no completion choices")
        content = result.choices[0].message.content
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model API returned no text; choose a text-capable model or increase its output limit")
        usage = getattr(result, "usage", None)
        tin, tout = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
        if (not isinstance(tin, int) or not isinstance(tout, int) or tin < 0 or tout < 0):
            raise ValueError("model API omitted valid token usage; cannot report inference cost")
        return Response(content, tin, tout, price(model, tin, tout))


class MockBackend:
    """Deterministic offline backend for tests and demos. NOT a model of real LLM behavior.

    P(correct) rises with model index and with the number of upstream inputs.
    """

    def __init__(self, answers: dict[str, str], base: list[float], input_bonus: float = 0.06):
        self.answers, self.base, self.bonus = answers, base, input_bonus
        self.models: list[str] = []

    def complete(self, prompt: str, model: ModelSpec, sample: int = 0) -> Response:
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

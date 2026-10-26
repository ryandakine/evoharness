"""Base tool interfaces, execution context, path validation, and registry."""
from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExecutionContext:
    cwd: Path = field(default_factory=Path.cwd)
    dry_run: bool = False
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolResult:
    output: str
    error: bool = False
    is_dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def truncate_output(
    text: str,
    max_bytes: int = 32768,
    head_lines: int = 100,
    tail_lines: int = 150,
) -> str:
    """Preserve head and tail of output if byte size exceeds max_bytes."""
    raw_bytes = text.encode("utf-8", errors="replace")
    if len(raw_bytes) <= max_bytes:
        return text

    lines = text.splitlines(keepends=True)
    if len(lines) <= (head_lines + tail_lines):
        # Truncate purely by bytes if lines are few but long
        cut = text[: max_bytes // 2] + "\n[... truncated ...]\n" + text[-max_bytes // 2 :]
        return cut

    head = "".join(lines[:head_lines])
    tail = "".join(lines[-tail_lines:])
    omitted_lines = len(lines) - head_lines - tail_lines
    omitted_bytes = len(raw_bytes) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    marker = f"\n[... {omitted_lines} lines / {omitted_bytes} bytes truncated ...]\n"
    return f"{head}{marker}{tail}"



SENSITIVE_ENV_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL|PRIVATE)", re.I)
DANGEROUS_ENV_PREFIXES = ("LD_", "BASH_")
DANGEROUS_ENV_KEYS = {"PYTHONPATH"}


def sanitize_env(base_env: dict[str, str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Sanitize environment variables for subprocess execution using pattern matching."""
    env = dict(base_env)
    if overrides:
        for k, v in overrides.items():
            if k in DANGEROUS_ENV_KEYS or any(k.startswith(p) for p in DANGEROUS_ENV_PREFIXES):
                continue
            if SENSITIVE_ENV_PATTERN.search(k):
                continue
            env[k] = v
    return {
        k: v
        for k, v in env.items()
        if not SENSITIVE_ENV_PATTERN.search(k)
        and not any(k.startswith(p) for p in DANGEROUS_ENV_PREFIXES)
        and k not in DANGEROUS_ENV_KEYS
    }

class PathGuard:
    """Validates that target paths strictly resolve within the workspace boundary."""

    @staticmethod
    def resolve_and_verify(path_str: str | Path, cwd: str | Path) -> Path:
        root = Path(cwd).resolve()
        if root == Path("/"):
            raise ValueError("Root directory '/' is not permitted as a tool workspace")
        target = (root / path_str).resolve()
        try:
            if not target.is_relative_to(root):
                raise ValueError(
                    f"Path traversal detected: '{path_str}' escapes workspace root '{root}'"
                )
        except AttributeError:
            # Fallback for Python < 3.9
            if root != target and root not in target.parents:
                raise ValueError(
                    f"Path traversal detected: '{path_str}' escapes workspace root '{root}'"
                ) from None
        return target


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]
    is_mutating: bool = False

    @abstractmethod
    def execute(self, args: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the tool with given arguments within context."""
        ...

    def to_litellm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Thread-safe registry of tools for the swarm."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.Lock()

    def register(self, tool: Tool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        with self._lock:
            return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        with self._lock:
            return list(self._tools.values())

    def to_litellm_tools(self) -> list[dict[str, Any]]:
        with self._lock:
            return [tool.to_litellm_tool() for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any], context: ExecutionContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                output=f"Error: Tool '{name}' not found. Available tools: {list(self._tools.keys())}",
                error=True,
            )
        try:
            return tool.execute(args, context)
        except Exception as exc:
            return ToolResult(
                output=f"Error executing tool '{name}': {type(exc).__name__}: {exc}",
                error=True,
            )

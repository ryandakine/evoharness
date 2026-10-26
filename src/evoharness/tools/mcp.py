"""Thread-safe MCP (Model Context Protocol) stdio client and tool bridge."""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import signal
import subprocess
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

from .base import ExecutionContext, Tool, ToolResult, sanitize_env, truncate_output

logger = logging.getLogger("evoharness.mcp")


def _sanitize_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure schema is a valid JSON schema object for LiteLLM / model providers."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out = dict(schema)
    if out.get("type") != "object":
        out["type"] = "object"
    if "properties" not in out or not isinstance(out["properties"], dict):
        out["properties"] = {}
    return out


class MCPStdioClient:
    """Thread-safe JSON-RPC 2.0 client managing an MCP server subprocess over stdio."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.command = command
        self.args = list(args or [])
        self.env = env
        self.cwd = cwd
        self.proc: subprocess.Popen | None = None
        self._next_id = 1
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, Future[dict[str, Any]]] = {}
        self._closing = False
        self._stderr_lines: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self, timeout: float = 15.0) -> None:
        cmd = [self.command] + self.args
        clean_env = sanitize_env(os.environ, self.env)

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self.cwd) if self.cwd else None,
            env=clean_env,
            start_new_session=True,  # Isolate process group
        )

        # Background thread to continuously drain stderr and prevent pipe deadlock
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        # Background thread to read stdout JSON-RPC lines
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_thread.start()

        # Perform MCP initialization handshake
        try:
            init_res = self._request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "evoharness", "version": "0.3.0"},
                },
                timeout=timeout,
            )
            if "error" in init_res:
                raise RuntimeError(f"MCP initialization failed: {init_res['error']}")

            # Mandatory notifications/initialized
            self._notify("notifications/initialized", {})
        except Exception:
            self.close()
            raise

    def _drain_stderr(self) -> None:
        if not self.proc or not self.proc.stderr:
            return
        for line in self.proc.stderr:
            clean_line = line.strip()
            if clean_line:
                self._stderr_lines.append(clean_line)
                if len(self._stderr_lines) > 50:
                    self._stderr_lines.pop(0)
            if self._closing:
                break
            logger.debug("[mcp stderr]: %s", clean_line)

    def _read_stdout(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_id = msg.get("id")
            if msg_id is not None:
                with self._pending_lock:
                    fut = self._pending.pop(msg_id, None)
                if fut and not fut.done():
                    fut.set_result(msg)

        # Process closed; reject any remaining pending requests
        code = self.proc.poll() if self.proc else None
        err_msg = f"MCP process terminated (exit code {code})."
        if self._stderr_lines:
            err_msg += f" Stderr: {' '.join(self._stderr_lines[-5:])}"

        with self._pending_lock:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError(err_msg))
            self._pending.clear()

    def _send(self, payload: dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin or self.proc.poll() is not None:
            raise RuntimeError("MCP process is not running")
        data = json.dumps(payload) + "\n"
        with self._write_lock:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._send(payload)

    def _request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
        with self._write_lock:
            req_id = self._next_id
            self._next_id += 1

        fut: Future[dict[str, Any]] = Future()
        with self._pending_lock:
            self._pending[req_id] = fut

        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        try:
            self._send(payload)
            return fut.result(timeout=timeout)
        except FuturesTimeoutError:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise TimeoutError(f"MCP request {method} (id={req_id}) timed out after {timeout}s") from None

    def list_tools(self, timeout: float = 30.0) -> list[dict[str, Any]]:
        res = self._request("tools/list", {}, timeout=timeout)
        if "error" in res:
            raise RuntimeError(f"MCP tools/list error: {res['error']}")
        return res.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        res = self._request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
        if "error" in res:
            raise RuntimeError(f"MCP tools/call error for {name}: {res['error']}")
        return res.get("result", {})

    def close(self) -> None:
        self._closing = True
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                self.proc.wait()


class MCPProcessPool:
    """Reuses active MCPStdioClient instances across DAG nodes and cleans them up on shutdown."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPStdioClient] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> MCPStdioClient:
        with self._lock:
            client = self._clients.get(name)
            if client is None or (client.proc and client.proc.poll() is not None):
                client = MCPStdioClient(command, args, env, cwd)
                client.start()
                self._clients[name] = client
            return client

    def close_all(self) -> None:
        with self._lock:
            for client in self._clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()


# Global pool singleton with atexit hook
_GLOBAL_POOL = MCPProcessPool()
atexit.register(_GLOBAL_POOL.close_all)


def get_mcp_pool() -> MCPProcessPool:
    return _GLOBAL_POOL


class MCPToolAdapter(Tool):
    """Adapts an MCP server's tool definition to the evoharness Tool interface."""

    def __init__(self, server_name: str, tool_def: dict[str, Any], client: MCPStdioClient) -> None:
        self.server_name = server_name
        self.mcp_name = tool_def["name"]
        safe_server = re.sub(r"[^a-zA-Z0-9_]", "_", server_name)
        safe_tool = re.sub(r"[^a-zA-Z0-9_]", "_", self.mcp_name)
        self.name = f"mcp__{safe_server}__{safe_tool}"[:64]
        self.description = tool_def.get("description") or f"MCP tool '{self.mcp_name}' from server '{server_name}'"
        self.parameters = _sanitize_schema(tool_def.get("inputSchema"))
        self.client = client
        self.is_mutating = True  # Treat remote tool calls as mutating by default

    def execute(self, args: dict[str, Any], context: ExecutionContext) -> ToolResult:
        if context.dry_run:
            return ToolResult(
                output=f"[dry-run: would call MCP tool '{self.name}' with args {json.dumps(args)}]",
                is_dry_run=True,
            )
        try:
            res = self.client.call_tool(self.mcp_name, args, timeout=30.0)
            is_error = bool(res.get("isError", False))
            content_blocks = res.get("content", [])
            out_texts: list[str] = []
            for block in content_blocks:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        out_texts.append(block.get("text", ""))
                    else:
                        out_texts.append(json.dumps(block))
                else:
                    out_texts.append(str(block))
            output = "\n".join(out_texts).strip() or "[MCP tool returned empty result]"
            return ToolResult(output=truncate_output(output), error=is_error)
        except Exception as exc:
            return ToolResult(output=f"MCP tool call failed ({self.name}): {exc}", error=True)


class MCPConfigLoader:
    """Loads MCP servers from mcp.json files."""

    @staticmethod
    def load(path: Path | str) -> dict[str, dict[str, Any]]:
        p = Path(path)
        if not p.is_file():
            return {}
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("mcpServers", {})
        except Exception as exc:
            logger.warning("Failed to load MCP config from %s: %s", p, exc)
            return {}

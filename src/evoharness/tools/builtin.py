"""Built-in tools: bash execution, file read, file write, and surgical editing."""
from __future__ import annotations

import difflib
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import ExecutionContext, PathGuard, Tool, ToolResult, truncate_output

# Sensitive environment variables stripped from subprocess execution by default
SENSITIVE_ENV_KEYS = {
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "KIMI_API_KEY",
    "GROK_API_KEY",
}


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command within the workspace directory. "
        "Returns standard output, standard error, and exit status."
    )
    is_mutating = True
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command string to execute in bash.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def execute(self, args: dict[str, Any], context: ExecutionContext) -> ToolResult:
        command = args.get("command", "").strip()
        if not command:
            return ToolResult(output="Error: 'command' argument cannot be empty", error=True)

        if context.dry_run:
            return ToolResult(
                output=f"[dry-run: would execute shell command in '{context.cwd}']:\n$ {command}",
                is_dry_run=True,
            )

        clean_env = {
            k: v for k, v in (os.environ | context.env).items() if k not in SENSITIVE_ENV_KEYS
        }

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(context.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=clean_env,
                start_new_session=True,  # Isolate process group for clean killpg on timeout
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                proc.communicate()
                return ToolResult(
                    output=f"Command timed out after {self.timeout}s: {command}",
                    error=True,
                )

            out_parts: list[str] = []
            if stdout:
                out_parts.append(stdout)
            if stderr:
                out_parts.append(f"[stderr]\n{stderr}")
            if proc.returncode != 0:
                out_parts.append(f"[exit code: {proc.returncode}]")

            combined = "\n".join(out_parts).strip() if out_parts else "[command executed with no output]"
            return ToolResult(
                output=truncate_output(combined),
                error=(proc.returncode != 0),
                metadata={"exit_code": proc.returncode},
            )

        except Exception as exc:
            return ToolResult(
                output=f"Subprocess execution error: {type(exc).__name__}: {exc}",
                error=True,
            )


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read content from a file inside the workspace boundary. "
        "Supports 1-based offset line and maximum limit lines."
    )
    is_mutating = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file inside workspace.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from (optional).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (optional).",
            },
        },
        "required": ["path"],
    }

    def execute(self, args: dict[str, Any], context: ExecutionContext) -> ToolResult:
        path_str = args.get("path", "")
        try:
            target = PathGuard.resolve_and_verify(path_str, context.cwd)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)

        if not target.exists():
            return ToolResult(output=f"File not found: {path_str}", error=True)
        if target.is_dir():
            return ToolResult(output=f"Target is a directory, not a file: {path_str}", error=True)

        try:
            with open(target, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            offset = max(1, int(args.get("offset") or 1))
            limit = int(args.get("limit")) if args.get("limit") is not None else None

            start_idx = offset - 1
            selected = lines[start_idx : (start_idx + limit) if limit else None]

            content = "".join(selected)
            return ToolResult(output=truncate_output(content))
        except Exception as exc:
            return ToolResult(output=f"Failed to read '{path_str}': {exc}", error=True)


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a file atomically inside the workspace boundary."
    is_mutating = True
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file inside workspace.",
            },
            "content": {
                "type": "string",
                "description": "Full text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def execute(self, args: dict[str, Any], context: ExecutionContext) -> ToolResult:
        path_str = args.get("path", "")
        content = args.get("content", "")
        try:
            target = PathGuard.resolve_and_verify(path_str, context.cwd)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)

        if context.dry_run:
            return ToolResult(
                output=f"[dry-run: would write {len(content)} characters to '{path_str}']",
                is_dry_run=True,
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write via temporary file in the same directory
            with tempfile.NamedTemporaryFile("w", dir=str(target.parent), delete=False, encoding="utf-8") as tf:
                tf.write(content)
                temp_path = Path(tf.name)
            os.replace(temp_path, target)
            return ToolResult(output=f"Successfully wrote {len(content)} characters to '{path_str}'")
        except Exception as exc:
            return ToolResult(output=f"Failed to write '{path_str}': {exc}", error=True)


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Surgically replace a unique block of text in a file inside workspace. "
        "The target 'old_content' must appear exactly once in the file."
    )
    is_mutating = True
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file inside workspace.",
            },
            "old_content": {
                "type": "string",
                "description": "Exact text chunk to be replaced (must match uniquely).",
            },
            "new_content": {
                "type": "string",
                "description": "New replacement text chunk.",
            },
        },
        "required": ["path", "old_content", "new_content"],
    }

    def execute(self, args: dict[str, Any], context: ExecutionContext) -> ToolResult:
        path_str = args.get("path", "")
        old_content = args.get("old_content", "")
        new_content = args.get("new_content", "")

        try:
            target = PathGuard.resolve_and_verify(path_str, context.cwd)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)

        if not target.exists():
            return ToolResult(output=f"File not found: {path_str}", error=True)

        try:
            with open(target, encoding="utf-8", errors="replace") as f:
                content = f.read()

            count = content.count(old_content)
            if count == 0:
                return ToolResult(
                    output=f"Error: old_content not found in '{path_str}'. Verify exact whitespace and lines.",
                    error=True,
                )
            if count > 1:
                return ToolResult(
                    output=f"Error: old_content appears {count} times in '{path_str}'. Must be unique.",
                    error=True,
                )

            new_full_content = content.replace(old_content, new_content, 1)

            if context.dry_run:
                diff = difflib.unified_diff(
                    content.splitlines(keepends=True),
                    new_full_content.splitlines(keepends=True),
                    fromfile=f"a/{path_str}",
                    tofile=f"b/{path_str}",
                )
                return ToolResult(
                    output=f"[dry-run diff for '{path_str}']:\n" + "".join(diff),
                    is_dry_run=True,
                )

            # Atomic replace
            with tempfile.NamedTemporaryFile("w", dir=str(target.parent), delete=False, encoding="utf-8") as tf:
                tf.write(new_full_content)
                temp_path = Path(tf.name)
            os.replace(temp_path, target)
            return ToolResult(output=f"Successfully replaced content in '{path_str}'")
        except Exception as exc:
            return ToolResult(output=f"Failed to edit '{path_str}': {exc}", error=True)

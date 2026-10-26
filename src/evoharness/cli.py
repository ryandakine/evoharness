"""Command-line interface for evoharness."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .executor import AgentError, RunResult, Task, run_genome
from .genome import Genome
from .llm import APIBackend, BudgetExceeded, EmptyResponse, ModelSpec
from .tools.base import ExecutionContext, ToolRegistry, ToolResult
from .tools.builtin import BashTool, EditFileTool, ReadFileTool, WriteFileTool
from .tools.mcp import MCPConfigLoader, MCPToolAdapter, get_mcp_pool

logger = logging.getLogger("evoharness.cli")


def _optional(cast):
    def parse(val: str):
        if val.lower() == "omit":
            return None
        return cast(val)
    return parse


FINAL = "Produce the final response. Finish with 'Answer: <final answer>'."

PRESETS: dict[str, dict[str, Any]] = {
    "solo": {
        "description": "One agent answers directly.",
        "roles": ["Solve the task carefully. Finish with 'Answer: <final answer>'."],
        "nodes": [0],
        "edges": [],
    },
    "pipeline": {
        "description": "Solver -> verifier -> finalizer.",
        "roles": [
            "Solve the task carefully and explain your solution.",
            "Check the supplied solution step by step. Point out and fix any errors.",
            FINAL,
        ],
        "nodes": [0, 1, 2],
        "edges": [[0, 1], [1, 2]],
    },
    "debate": {
        "description": "Two independent solvers -> reviewer -> finalizer (the original four-agent demo).",
        "roles": [
            "Solve the task carefully and explain your solution.",
            "Develop an independent alternative solution.",
            "Review the supplied solutions, correct errors, and resolve disagreements.",
            FINAL,
        ],
        "nodes": [0, 1, 2, 3],
        "edges": [[0, 2], [1, 2], [2, 3]],
    },
    "committee": {
        "description": "Three independent solvers -> judge who picks the best-supported answer.",
        "roles": [
            "Solve the task carefully and explain your solution.",
            "Solve the task using a different approach from the obvious one, and explain it.",
            "Solve the task, then double-check your own work for mistakes.",
            "You are the judge. Compare the supplied solutions, weigh their reasoning, and "
            "decide. Finish with 'Answer: <final answer>'.",
        ],
        "nodes": [0, 1, 2, 3],
        "edges": [[0, 3], [1, 3], [2, 3]],
    },
    "audit": {
        "description": "Researcher (read/bash) -> Implementer (edit/write) -> Verifier (bash) -> Judge.",
        "roles": [
            "You are a researcher. Inspect the workspace, read files, or run read-only commands to diagnose the issue.",
            "You are an implementer. Write or edit files to fix the issue or implement the requirement.",
            "You are a verifier. Run tests or build commands via bash to verify the fix works. Report pass/fail.",
            "You are the judge. Review the audit trace and summarize findings. Finish with 'Answer: <final answer>'.",
        ],
        "nodes": [
            {"role": 0, "tools": ["read_file", "bash"]},
            {"role": 1, "tools": ["read_file", "write_file", "edit_file"]},
            {"role": 2, "tools": ["read_file", "bash"]},
            {"role": 3, "tools": []},
        ],
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
    },
}


def _model_from(entry: Any, default: ModelSpec) -> ModelSpec:
    if isinstance(entry, str):
        return ModelSpec(entry, default.price_in, default.price_out)
    if isinstance(entry, dict) and isinstance(entry.get("name"), str):
        return ModelSpec(entry["name"], float(entry.get("price_in", default.price_in)),
                         float(entry.get("price_out", default.price_out)),
                         entry.get("api_base"), entry.get("api_key_env"))
    raise ValueError("each model must be a provider/model string or an object with a 'name'")


def load_spec(data: dict[str, Any], default_model: ModelSpec) -> tuple[Genome, list[str], list[ModelSpec]]:
    """Turn a spec dict into (genome, roles, models). Raises ValueError with a readable message."""
    roles = data.get("roles")
    if not isinstance(roles, list) or not roles or not all(isinstance(r, str) and r.strip() for r in roles):
        raise ValueError("spec needs a non-empty 'roles' list of instruction strings")
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("spec needs a non-empty 'nodes' list")
    models = [_model_from(m, default_model) for m in data["models"]] if data.get("models") else [default_model]
    node_roles, node_models, node_tools = [], [], []
    has_tools = False
    for k, node in enumerate(nodes):
        if isinstance(node, int):
            role, model, tools_cfg = node, 0, None
        elif isinstance(node, dict):
            role, model, tools_cfg = node.get("role"), node.get("model", 0), node.get("tools")
        else:
            raise ValueError(f"node {k} must be a role index or an object with 'role'")
        if not isinstance(role, int) or not isinstance(model, int):
            raise ValueError(f"node {k}: 'role' and 'model' must be integer indices")
        node_roles.append(role)
        node_models.append(model)
        if tools_cfg is not None:
            has_tools = True
            node_tools.append(list(tools_cfg))
        else:
            node_tools.append([])
    try:
        edges = {(int(a), int(b)) for a, b in data.get("edges", [])}
    except (TypeError, ValueError) as exc:
        raise ValueError("'edges' must be a list of [from, to] pairs") from exc
    genome = Genome(node_roles, node_models, edges, tools=node_tools if has_tools else None)
    genome.validate_against(len(roles), len(models))
    return genome, roles, models


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evoharness", description="Run a swarm of LLM agents on one prompt.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Presets: " + "; ".join(f"{k}: {v['description']}" for k, v in PRESETS.items()))
    p.add_argument("--prompt", help="the task for the swarm")
    p.add_argument("--model", help="LiteLLM provider/model, e.g. openai/gpt-4o-mini (default: $LLM_MODEL)")
    p.add_argument("--api-base", help="custom OpenAI-compatible base URL (default: $LLM_API_BASE)")
    p.add_argument("--env-file", help="dotenv file to load (default: ./.env)")
    graph = p.add_mutually_exclusive_group()
    graph.add_argument("--preset", choices=sorted(PRESETS), default="debate", help="built-in swarm (default: debate)")
    graph.add_argument("--spec", help="JSON file describing a custom swarm")
    p.add_argument("--max-tokens", type=_optional(int), default=1024, help="output limit per agent, or 'omit'")
    p.add_argument("--temperature", type=_optional(float), default=0.7, help="sampling temperature, or 'omit'")
    p.add_argument("--timeout", type=float, default=120.0, help="seconds per model call (default 120)")
    p.add_argument("--retries", type=int, default=2, help="retries per agent on transient errors (default 2)")
    p.add_argument("--parallel", type=int, default=2, help="max agents to run at once (default 2)")
    p.add_argument("--price-in", type=float, default=0, help="USD per million input tokens")
    p.add_argument("--price-out", type=float, default=0, help="USD per million output tokens")
    p.add_argument("--json", action="store_true", help="print a JSON report instead of text")
    p.add_argument("--quiet", action="store_true", help="print only the final answer")
    p.add_argument("--list-presets", action="store_true", help="list built-in swarms and exit")
    p.add_argument("--version", action="version", version=f"evoharness {__version__}")

    # Tool options
    p.add_argument("--tools", default="none", help="tools available: none, all, builtin, bash, files, mcp, or comma-separated list")
    p.add_argument("--mcp", help="path to MCP config file (.mcp.json)")
    p.add_argument("--cwd", help="workspace directory for tools (default: current directory)")
    p.add_argument("--max-tool-turns", type=int, default=10, help="max tool iterations per agent node (default: 10)")
    p.add_argument("--dry-run", action="store_true", help="simulate tool mutations without modifying files")
    p.add_argument("--audit-file", help="path to JSON lines tool audit file (default: ./.evoharness/audit.jsonl when tools active)")
    return p


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit('Install API dependencies: python -m pip install -e ".[api]"') from exc
    # Keep environment overrides; do not expand ${...} inside secrets.
    load_dotenv(path, override=False, interpolate=False)


def _setup_tools(tools_arg: str, mcp_config: str | None, cwd: Path, dry_run: bool) -> tuple[ToolRegistry | None, ExecutionContext]:
    clean_arg = tools_arg.strip().lower()
    ctx = ExecutionContext(cwd=cwd, dry_run=dry_run)
    if clean_arg in ("none", ""):
        return None, ctx

    registry = ToolRegistry()
    parts = [s.strip() for s in clean_arg.split(",")]

    if "all" in parts or "builtin" in parts or "bash" in parts:
        registry.register(BashTool())
    if "all" in parts or "builtin" in parts or "files" in parts:
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())

    if "all" in parts or "mcp" in parts or mcp_config:
        cfg_path = Path(mcp_config) if mcp_config else (cwd / ".mcp.json")
        if not cfg_path.is_file() and not mcp_config:
            cfg_path = Path.home() / ".claude" / "mcp.json"
        if cfg_path.is_file():
            servers = MCPConfigLoader.load(cfg_path)
            pool = get_mcp_pool()
            for s_name, s_cfg in servers.items():
                cmd = s_cfg.get("command")
                if cmd:
                    args = s_cfg.get("args", [])
                    env = s_cfg.get("env")
                    try:
                        client = pool.get_or_create(s_name, cmd, args, env, cwd=cwd)
                        for t_def in client.list_tools(timeout=10.0):
                            registry.register(MCPToolAdapter(s_name, t_def, client))
                    except Exception as exc:
                        logger.warning("Failed to initialize MCP server '%s': %s", s_name, exc)
    return registry, ctx


def _report(result: RunResult, roles: list[str], genome: Genome, models: list[ModelSpec],
            args: argparse.Namespace, elapsed: float) -> None:
    if args.json:
        print(json.dumps({
            "answer": result.answer, "calls": result.calls, "cost": result.cost,
            "elapsed_s": round(elapsed, 3), "models": [m.name for m in models],
            "preset": None if args.spec else args.preset, "roles": roles, "genome": genome.to_dict(),
            "outputs": {str(k): v for k, v in result.outputs.items()},
        }, indent=2, ensure_ascii=False))
        return
    if args.quiet:
        print(result.answer)
        return
    for node, output in result.outputs.items():
        title = roles[genome.roles[node]].splitlines()[0][:72]
        print(f"\n--- Agent {node} [{models[genome.models[node]].name}]: {title} ---\n{output}")
    print(f"\nFINAL: {result.answer}")
    print(f"Model calls: {result.calls} | elapsed {elapsed:.1f}s")
    if args.price_in or args.price_out:
        print(f"Nominal inference cost: ${result.cost:.6f}")
    else:
        print("Cost tracking disabled: supply token prices with --price-in and --price-out.")


def main(argv: Sequence[str] | None = None, env_file: str | Path | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_presets:
        for name, preset in PRESETS.items():
            print(f"{name:10s} {len(preset['nodes'])} agents  {preset['description']}")
        return 0
    _load_env(Path(args.env_file or env_file or ".env"))
    args.model = args.model or os.environ.get("LLM_MODEL")
    args.api_base = args.api_base or os.environ.get("LLM_API_BASE") or None
    if not args.prompt:
        parser.error("--prompt is required")
    if not args.model:
        parser.error("set LLM_MODEL in .env or pass --model provider/model")
    if args.price_in < 0 or args.price_out < 0:
        parser.error("token prices must be nonnegative")
    if args.retries < 0 or args.parallel < 1 or args.timeout <= 0:
        parser.error("--retries must be >= 0, --parallel >= 1 and --timeout > 0")

    default_model = ModelSpec(args.model, args.price_in, args.price_out)
    try:
        if args.spec:
            with open(args.spec, encoding="utf-8") as fh:
                spec = json.load(fh)
        else:
            spec = PRESETS[args.preset]
        genome, roles, models = load_spec(spec, default_model)
    except (OSError, ValueError) as exc:
        parser.error(f"bad swarm spec: {exc}")

    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    tools_arg = args.tools
    if tools_arg == "none" and genome.tools is not None and any(genome.tools):
        # Auto-enable tools if the spec explicitly defines node tools
        tools_arg = "all"

    tool_registry, ctx = _setup_tools(tools_arg, args.mcp, cwd, args.dry_run)
    audit_path = Path(args.audit_file) if args.audit_file else (cwd / ".evoharness" / "audit.jsonl" if tool_registry else None)

    backend = APIBackend(api_base=args.api_base, api_key=os.environ.get("LLM_API_KEY") or None,
                         max_tokens=args.max_tokens, temperature=args.temperature,
                         timeout=args.timeout, retries=args.retries)
    if not args.json and not args.quiet:
        label = args.spec or f"preset '{args.preset}'"
        names = ", ".join(sorted({m.name for m in models}))
        tool_desc = f", tools: {len(tool_registry.list_tools())} loaded" if tool_registry else ""
        print(f"Running {label}: {genome.n} agents, {len(genome.edges)} edges, model(s) {names}{tool_desc} ...",
              file=sys.stderr, flush=True)

    def _on_tool_call(node: int, tool_name: str, t_args: dict[str, Any], res: ToolResult) -> None:
        if not args.quiet and not args.json:
            status = "[error]" if res.error else ("[dry-run]" if res.is_dry_run else "[ok]")
            arg_summary = json.dumps(t_args, ensure_ascii=False)
            if len(arg_summary) > 70:
                arg_summary = arg_summary[:67] + "..."
            print(f"  -> Agent {node} called {tool_name}({arg_summary}) {status}", file=sys.stderr, flush=True)

    start = time.monotonic()
    node_tools_map = {i: genome.tools[i] for i in range(genome.n)} if genome.tools is not None else None
    try:
        result = run_genome(
            genome,
            Task(args.prompt, ""),
            backend,
            models,
            roles,
            max_workers=args.parallel,
            tool_registry=tool_registry,
            node_tools=node_tools_map,
            context=ctx,
            max_tool_turns=args.max_tool_turns,
            on_tool_call=_on_tool_call,
            audit_file=audit_path,
        )
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if isinstance(exc.__cause__, EmptyResponse):
            print("hint: the model returned no text. Shared/free routers do this under load; "
                  "retry, pick a specific --model, or raise --max-tokens.", file=sys.stderr)
        return 1
    except BudgetExceeded as exc:
        print(f"error: budget exhausted: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    _report(result, roles, genome, models, args, time.monotonic() - start)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

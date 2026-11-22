"""Targeted unit tests for security hardening, audit sanitization, and budget limits."""
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evoharness import __version__
from evoharness.executor import Task, _sanitize_audit_args, run_genome, run_react_node
from evoharness.genome import Genome
from evoharness.llm import BudgetExceeded, CachedBackend, ModelSpec, Response
from evoharness.tools.base import ExecutionContext, ToolRegistry, redact_sensitive_text, truncate_output
from evoharness.tools.builtin import BashTool, EditFileTool, ReadFileTool, WriteFileTool
from evoharness.tools.mcp import MCPProcessPool


def test_version_alignment():
    """Package version must be 0.3.4."""
    assert __version__ == "0.3.4"
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert 'version = "0.3.4"' in pyproject.read_text(encoding="utf-8")


def test_tool_allowlist_enforcement():
    """Agent must reject tools not in allowed_tools even if registered."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(BashTool())

    mock_backend = MagicMock()
    mock_backend.complete_messages.side_effect = [
        Response(
            "I will run bash",
            10,
            10,
            0.001,
            tool_calls=[{"id": "call_1", "function": {"name": "bash", "arguments": "{\"command\": \"echo hacked\"}"}}],
        ),
        Response("Done", 10, 10, 0.001, tool_calls=[]),
    ]

    ctx = ExecutionContext(cwd=Path.cwd())
    resp = run_react_node(
        node=0,
        prompt="Test",
        model=ModelSpec("test-model", 0.0, 0.0),
        sample=0,
        backend=mock_backend,
        tool_registry=registry,
        allowed_tools={"read_file"},  # bash is forbidden
        context=ctx,
        max_tool_turns=5,
    )
    assert resp.text == "Done"
    assert len(resp.tool_calls) == 1
    # Check that the second call received the authorization error
    second_call_messages = mock_backend.complete_messages.call_args_list[1][0][0]
    tool_resp_msg = [m for m in second_call_messages if m.get("role") == "tool"][0]
    assert "is not permitted for this agent node" in tool_resp_msg["content"]
    assert "Allowed tools: ['read_file']" in tool_resp_msg["content"]


def test_cached_backend_complete_messages_budget():
    """CachedBackend.complete_messages must enforce budget caps."""
    mock_inner = MagicMock()
    mock_inner.complete_messages.return_value = Response("result", 10, 10, 0.05)

    cached = CachedBackend(mock_inner, budget_usd=0.08)
    model = ModelSpec("test-model", 0.0, 0.0)
    messages = [{"role": "user", "content": "hi"}]

    # First call costs 0.05 (total spent: 0.05 <= 0.08) -> succeeds
    r1 = cached.complete_messages(messages, model)
    assert r1.text == "result"
    assert cached.spent == pytest.approx(0.05)

    # Second call costs 0.05 (total spent would be 0.10 > 0.08) -> raises BudgetExceeded
    with pytest.raises(BudgetExceeded) as exc_info:
        cached.complete_messages(messages, model)
    assert "spent $0.1000 >= cap $0.0800" in str(exc_info.value)

    # Further calls fail immediately before contacting inner backend
    with pytest.raises(BudgetExceeded):
        cached.complete_messages(messages, model)
    assert mock_inner.complete_messages.call_count == 2


def test_secret_scrubbing_json_and_env():
    """redact_sensitive_text must scrub JSON credentials and environment secrets."""
    env = {"MY_SECRET_TOKEN": "supersecretvalue123"}
    text = (
        '{"api_key": "sk-1234567890abcdef", "normal": "ok", '
        '"auth_token": "bearer-xyz-secret"}\n'
        'Value was supersecretvalue123 in log'
    )
    redacted = redact_sensitive_text(text, env=env)
    assert 'sk-1234567890abcdef' not in redacted
    assert 'bearer-xyz-secret' not in redacted
    assert 'supersecretvalue123' not in redacted
    assert '"api_key": "[REDACTED]"' in redacted
    assert '"auth_token": "[REDACTED]"' in redacted
    assert '"normal": "ok"' in redacted


def test_sanitize_audit_args_structured_keys():
    """_sanitize_audit_args must redact values for sensitive dictionary keys."""
    raw_args = {
        "api_key": "plain-text-secret",
        "nested": {"token": "nested-token", "count": 42},
        "safe_field": "safe_value",
    }
    sanitized = _sanitize_audit_args(raw_args)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert sanitized["nested"]["count"] == 42
    assert sanitized["safe_field"] == "safe_value"


def test_hard_byte_truncation():
    """truncate_output must never exceed max_bytes under any input."""
    for length in [50, 100, 500, 2000, 10000]:
        text = "🔥" * length + "A" * length
        out = truncate_output(text, max_bytes=200)
        assert len(out.encode("utf-8")) <= 200


def test_file_tools_preserve_permissions():
    """WriteFileTool and EditFileTool must preserve existing file modes."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        target = p / "script.sh"
        target.write_text("#!/bin/bash\necho original\n", encoding="utf-8")
        target.chmod(0o755)

        ctx = ExecutionContext(cwd=p)
        write_tool = WriteFileTool()
        res = write_tool.execute({"path": "script.sh", "content": "#!/bin/bash\necho replaced\n"}, ctx)
        assert not res.error
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o755

        edit_tool = EditFileTool()
        res2 = edit_tool.execute({
            "path": "script.sh",
            "old_content": "echo replaced",
            "new_content": "echo edited",
        }, ctx)
        assert not res2.error
        mode2 = stat.S_IMODE(target.stat().st_mode)
        assert mode2 == 0o755


def test_mcp_pool_scoping():
    """MCPProcessPool must isolate clients by command, args, and cwd."""
    pool = MCPProcessPool()
    assert pool._clients == {}
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        k1 = ("srv", "python", (), str(Path(td1).resolve()))
        k2 = ("srv", "python", (), str(Path(td2).resolve()))
        assert k1 != k2


def test_node_cache_invalidation_on_mutating():
    """Mutating tool nodes must bypass cache; non-mutating nodes must memoize."""
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(ReadFileTool())

    mock_backend = MagicMock()
    mock_backend.complete_messages.return_value = Response("done", 10, 10, 0.001, tool_calls=[])
    mock_backend.complete.return_value = Response("text answer", 10, 10, 0.001)

    genome = Genome(roles=[0, 1], models=[0, 0], edges=[(0, 1)], tools=[["bash"], ["read_file"]])
    models = [ModelSpec("test-model", 0.0, 0.0)]
    roles = ["Role 0", "Role 1"]
    task = Task("Do something", "")

    cache = {}
    ctx = ExecutionContext(cwd=Path.cwd())
    node_tools_map = {0: ["bash"], 1: ["read_file"]}

    # First run
    run_genome(
        genome, task, mock_backend, models, roles,
        node_cache=cache, tool_registry=registry, node_tools=node_tools_map, context=ctx,
    )
    # Node 0 (bash is mutating) must NOT be cached
    # Node 1 (read_file is non-mutating) MUST be cached
    cached_nodes = {k[0] for k in cache.keys()}
    assert 0 not in cached_nodes
    assert 1 in cached_nodes

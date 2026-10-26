import json
from pathlib import Path
import pytest

from evoharness.executor import run_genome, run_react_node, Task
from evoharness.genome import Genome
from evoharness.llm import ModelSpec, Response
from evoharness.tools.base import ExecutionContext, ToolRegistry
from evoharness.tools.builtin import ReadFileTool, WriteFileTool


class MockReActBackend:
    """Mock backend simulating multi-turn tool calling."""

    def __init__(self, script: list[Response]):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def complete_messages(self, messages, model, sample=0, tools=None, tool_choice=None):
        self.calls.append(list(messages))
        if self.script:
            return self.script.pop(0)
        return Response("Answer: default fallback", 10, 10, 0.001)

    def complete(self, prompt, model, sample=0):
        return self.complete_messages([{"role": "user", "content": prompt}], model, sample)


def test_react_node_execution(tmp_path):
    (tmp_path / "secret.txt").write_text("PIN: 9876")
    audit_file = tmp_path / "audit.jsonl"
    ctx = ExecutionContext(cwd=tmp_path)

    registry = ToolRegistry()
    registry.register(ReadFileTool())

    # Step 1: Model requests read_file
    call_1 = Response(
        text="",
        input_tokens=10,
        output_tokens=5,
        cost=0.0001,
        tool_calls=[{
            "id": "tc_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "secret.txt"})},
        }],
    )
    # Step 2: Model sees result and outputs answer
    call_2 = Response(
        text="The PIN is 9876.\nAnswer: 9876",
        input_tokens=25,
        output_tokens=10,
        cost=0.0002,
    )

    backend = MockReActBackend([call_1, call_2])
    model = ModelSpec("test-model", 1.0, 1.0)

    res = run_react_node(
        node=0,
        prompt="What is the PIN in secret.txt?",
        model=model,
        sample=0,
        backend=backend,
        tool_registry=registry,
        allowed_tools=None,
        context=ctx,
        max_tool_turns=5,
        audit_file=audit_file,
    )

    assert "The PIN is 9876" in res.text
    assert "9876" in res.text
    assert res.cost > 0.0002
    assert audit_file.exists()
    audit_lines = audit_file.read_text().strip().splitlines()
    assert len(audit_lines) == 1
    log_rec = json.loads(audit_lines[0])
    assert log_rec["tool"] == "read_file"
    assert log_rec["node"] == 0


def test_react_parallel_tool_calls(tmp_path):
    (tmp_path / "a.txt").write_text("Alpha")
    (tmp_path / "b.txt").write_text("Beta")
    ctx = ExecutionContext(cwd=tmp_path)

    registry = ToolRegistry()
    registry.register(ReadFileTool())

    # Parallel tool calls in 1 turn
    call_1 = Response(
        text="",
        input_tokens=10,
        output_tokens=10,
        cost=0.0001,
        tool_calls=[
            {
                "id": "tc_a",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt"})},
            },
            {
                "id": "tc_b",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "b.txt"})},
            },
        ],
    )
    call_2 = Response(
        text="Combined: Alpha Beta\nAnswer: Alpha Beta",
        input_tokens=40,
        output_tokens=15,
        cost=0.0003,
    )

    backend = MockReActBackend([call_1, call_2])
    model = ModelSpec("test-model", 1.0, 1.0)

    res = run_react_node(
        node=0,
        prompt="Combine a.txt and b.txt",
        model=model,
        sample=0,
        backend=backend,
        tool_registry=registry,
        allowed_tools=None,
        context=ctx,
        max_tool_turns=5,
    )

    assert "Combined: Alpha Beta" in res.text
    # Verify backend received both tool results in messages
    last_call_messages = backend.calls[-1]
    tool_messages = [m for m in last_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["tool_call_id"] == "tc_a"
    assert "Alpha" in tool_messages[0]["content"]
    assert tool_messages[1]["tool_call_id"] == "tc_b"
    assert "Beta" in tool_messages[1]["content"]


def test_react_forced_synthesis_on_max_turns(tmp_path):
    ctx = ExecutionContext(cwd=tmp_path)
    registry = ToolRegistry()
    registry.register(ReadFileTool())

    infinite_call = Response(
        text="",
        input_tokens=10,
        output_tokens=5,
        cost=0.0001,
        tool_calls=[{
            "id": "tc_loop",
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "nonexistent"})},
        }],
    )
    forced_synth_call = Response(
        text="I hit the turn limit, best answer is unknown.\nAnswer: unknown",
        input_tokens=50,
        output_tokens=10,
        cost=0.0002,
    )

    backend = MockReActBackend([infinite_call, infinite_call, forced_synth_call])
    model = ModelSpec("test-model", 1.0, 1.0)

    res = run_react_node(
        node=0,
        prompt="Looping task",
        model=model,
        sample=0,
        backend=backend,
        tool_registry=registry,
        allowed_tools=None,
        context=ctx,
        max_tool_turns=2,  # Max 2 turns
    )

    assert "I hit the turn limit" in res.text
    assert "unknown" in res.text

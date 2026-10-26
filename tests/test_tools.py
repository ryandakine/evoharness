
import pytest

from evoharness.tools.base import ExecutionContext, PathGuard, ToolRegistry, truncate_output
from evoharness.tools.builtin import BashTool, EditFileTool, ReadFileTool, WriteFileTool


def test_truncate_output():
    short = "line 1\nline 2\nline 3"
    assert truncate_output(short, max_bytes=1000) == short

    # Generate text > 1000 bytes
    many_lines = "\n".join(f"line {i}" for i in range(500))
    truncated = truncate_output(many_lines, max_bytes=500, head_lines=5, tail_lines=5)
    assert "[..." in truncated
    assert "truncated ...]" in truncated
    assert truncated.startswith("line 0\nline 1\n")
    assert truncated.endswith("line 499")


def test_path_guard_confinement(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "valid.txt").write_text("ok")

    resolved = PathGuard.resolve_and_verify("valid.txt", workspace)
    assert resolved == (workspace / "valid.txt").resolve()

    # Traversal test
    with pytest.raises(ValueError, match="escapes workspace root"):
        PathGuard.resolve_and_verify("../../etc/passwd", workspace)

    with pytest.raises(ValueError, match="escapes workspace root"):
        PathGuard.resolve_and_verify("/etc/passwd", workspace)


def test_file_tools(tmp_path):
    ctx = ExecutionContext(cwd=tmp_path)
    write_tool = WriteFileTool()
    read_tool = ReadFileTool()
    edit_tool = EditFileTool()

    # 1. Write file
    res = write_tool.execute({"path": "nested/test.txt", "content": "hello\nworld\nfoo\nbar"}, ctx)
    assert not res.error
    assert (tmp_path / "nested/test.txt").read_text() == "hello\nworld\nfoo\nbar"

    # 2. Read file with offset and limit
    res_read = read_tool.execute({"path": "nested/test.txt", "offset": 2, "limit": 2}, ctx)
    assert not res_read.error
    assert res_read.output == "world\nfoo\n"

    # 3. Edit file (dry-run)
    dry_ctx = ExecutionContext(cwd=tmp_path, dry_run=True)
    res_edit_dry = edit_tool.execute({
        "path": "nested/test.txt",
        "old_content": "foo",
        "new_content": "qux"
    }, dry_ctx)
    assert res_edit_dry.is_dry_run
    assert "+qux" in res_edit_dry.output
    # Assert file untouched
    assert "qux" not in (tmp_path / "nested/test.txt").read_text()

    # 4. Edit file live
    res_edit = edit_tool.execute({
        "path": "nested/test.txt",
        "old_content": "foo",
        "new_content": "qux"
    }, ctx)
    assert not res_edit.error
    assert (tmp_path / "nested/test.txt").read_text() == "hello\nworld\nqux\nbar"

    # 5. Non-unique error
    (tmp_path / "dup.txt").write_text("aaa bbb aaa")
    res_dup = edit_tool.execute({"path": "dup.txt", "old_content": "aaa", "new_content": "zzz"}, ctx)
    assert res_dup.error
    assert "appears 2 times" in res_dup.output


def test_bash_tool(tmp_path):
    ctx = ExecutionContext(cwd=tmp_path)
    tool = BashTool(timeout=5.0)

    # Simple command
    res = tool.execute({"command": "echo 'hello shell'"}, ctx)
    assert not res.error
    assert "hello shell" in res.output

    # Dry run
    dry_ctx = ExecutionContext(cwd=tmp_path, dry_run=True)
    res_dry = tool.execute({"command": "touch bad.txt"}, dry_ctx)
    assert res_dry.is_dry_run
    assert not (tmp_path / "bad.txt").exists()

    # Timeout handling
    res_timeout = BashTool(timeout=0.2).execute({"command": "sleep 2"}, ctx)
    assert res_timeout.error
    assert "timed out" in res_timeout.output


def test_tool_registry():
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(ReadFileTool())

    assert registry.get("bash") is not None
    assert len(registry.list_tools()) == 2
    schemas = registry.to_litellm_tools()
    assert len(schemas) == 2
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] in ("bash", "read_file")

    # Unknown tool dispatch
    res = registry.execute("nonexistent", {}, ExecutionContext())
    assert res.error
    assert "not found" in res.output


def test_path_guard_rejects_root():
    with pytest.raises(ValueError, match="Root directory.*not permitted"):
        PathGuard.resolve_and_verify("foo.txt", "/")


def test_sanitize_env_pattern():
    from evoharness.tools.base import sanitize_env
    raw_env = {
        "PATH": "/usr/bin",
        "MY_SECRET_KEY": "hidden",
        "AUTH_TOKEN": "hidden",
        "DB_PASSWORD": "hidden",
        "LD_PRELOAD": "/bad.so",
        "PYTHONPATH": "/bad/path",
        "SAFE_VAR": "visible",
    }
    cleaned = sanitize_env(raw_env, overrides={"USER_TOKEN": "xyz", "FOO": "bar"})
    assert "SAFE_VAR" in cleaned
    assert "FOO" in cleaned
    assert "MY_SECRET_KEY" not in cleaned
    assert "AUTH_TOKEN" not in cleaned
    assert "DB_PASSWORD" not in cleaned
    assert "LD_PRELOAD" not in cleaned
    assert "PYTHONPATH" not in cleaned
    assert "USER_TOKEN" not in cleaned

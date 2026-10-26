import json
import sys

from evoharness.tools.base import ExecutionContext
from evoharness.tools.mcp import MCPConfigLoader, MCPStdioClient, MCPToolAdapter, _sanitize_schema

# Mock MCP server script written with explicit chr(10) newlines
MOCK_SERVER_SCRIPT = '''
import json
import sys

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue

    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-server", "version": "1.0.0"}
            }
        }
        sys.stdout.write(json.dumps(resp) + chr(10))
        sys.stdout.flush()
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back input message",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"}
                            },
                            "required": ["message"]
                        }
                    }
                ]
            }
        }
        sys.stdout.write(json.dumps(resp) + chr(10))
        sys.stdout.flush()
    elif method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "echo":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"ECHO: {args.get('message', '')}"}],
                    "isError": False
                }
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Tool {tool_name} not found"}
            }
        sys.stdout.write(json.dumps(resp) + chr(10))
        sys.stdout.flush()
'''


def test_sanitize_schema():
    assert _sanitize_schema(None) == {"type": "object", "properties": {}}
    assert _sanitize_schema({}) == {"type": "object", "properties": {}}
    assert _sanitize_schema({"type": "string"}) == {"type": "object", "properties": {}}
    valid = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert _sanitize_schema(valid) == valid


def test_mcp_client_e2e(tmp_path):
    server_file = tmp_path / "mock_server.py"
    server_file.write_text(MOCK_SERVER_SCRIPT.strip())

    client = MCPStdioClient(sys.executable, ["-u", str(server_file)], cwd=tmp_path)
    try:
        client.start(timeout=5.0)
        assert client.proc is not None
        assert client.proc.poll() is None

        # 1. List tools
        tools = client.list_tools(timeout=5.0)
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"

        # 2. Call tool directly
        res = client.call_tool("echo", {"message": "test ping"}, timeout=5.0)
        assert not res.get("isError")
        assert "ECHO: test ping" in res["content"][0]["text"]

        # 3. Tool adapter integration
        adapter = MCPToolAdapter("mock", tools[0], client)
        assert adapter.name == "mcp__mock__echo"
        exec_res = adapter.execute({"message": "adapter ping"}, ExecutionContext(cwd=tmp_path))
        assert not exec_res.error
        assert exec_res.output == "ECHO: adapter ping"

        # 4. Dry run
        dry_res = adapter.execute({"message": "adapter ping"}, ExecutionContext(cwd=tmp_path, dry_run=True))
        assert dry_res.is_dry_run
        assert "[dry-run: would call MCP tool" in dry_res.output

    finally:
        client.close()
        assert client.proc.poll() is not None


def test_mcp_config_loader(tmp_path):
    cfg_file = tmp_path / "mcp.json"
    cfg_file.write_text(json.dumps({
        "mcpServers": {
            "osi": {
                "command": "uv",
                "args": ["run", "osi-runtime-mcp"],
                "env": {"FOO": "BAR"}
            }
        }
    }))

    servers = MCPConfigLoader.load(cfg_file)
    assert "osi" in servers
    assert servers["osi"]["command"] == "uv"
    assert servers["osi"]["env"]["FOO"] == "BAR"

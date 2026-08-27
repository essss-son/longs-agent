"""MCP client 测试（stub，不真实启动 MCP server）。"""
from __future__ import annotations

from agent.mcp_client import MCPToolWrapper


class _FakeMCPTool:
    name = "fake_tool"
    description = "a fake mcp tool"
    inputSchema = {"type": "object", "properties": {}, "additionalProperties": False}


class _FakeSession:
    async def call_tool(self, name, args):
        return f"called {name} with {args}"


def test_mcp_wrapper_properties():
    w = MCPToolWrapper(_FakeMCPTool(), _FakeSession())
    assert w.name == "fake_tool"
    assert w.description == "a fake mcp tool"
    assert w.schema == _FakeMCPTool.inputSchema
    assert w.read_only is False


async def test_mcp_wrapper_execute_returns_text():
    w = MCPToolWrapper(_FakeMCPTool(), _FakeSession())
    result = await w.execute(arg1="x")
    assert "fake_tool" in result
    assert "x" in result
    assert isinstance(result, str)

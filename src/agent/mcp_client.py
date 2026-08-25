"""MCP client（D14）。包装 MCP 工具为 Tool 注册进 Registry（统一抽象红利）。

注册进同一 Registry → MCP 工具自动获权限引擎 + trace + plan mode 过滤。
load_mcp_server 需 mcp 包（optional [mcp]，pip install -e ".[mcp]"）。
"""
from __future__ import annotations

from typing import Any

from .tools import Tool


class MCPToolWrapper(Tool):
    """包装 MCP 工具为 agent Tool。read_only 默认 False（走 ask）。"""

    def __init__(self, mcp_tool, session):
        self._tool = mcp_tool
        self._session = session
        self.name = mcp_tool.name
        self.description = mcp_tool.description
        self.schema = mcp_tool.inputSchema
        self.read_only = False  # MCP 工具默认 ask；可在 config 标注

    async def execute(self, **kwargs: Any) -> str:
        result = await self._session.call_tool(self._tool.name, kwargs)
        return str(result)


async def load_mcp_server(name: str, command: str, args: list[str] | None = None) -> list:
    """启动 MCP server（stdio），返回 MCPToolWrapper 列表。需 mcp 包。

    未装 mcp 时调用会 ImportError（提示 pip install -e ".[mcp]"）。
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args or [])
    tools: list = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            for t in resp.tools:
                tools.append(MCPToolWrapper(t, session))
    return tools

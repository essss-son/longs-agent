"""MCP client（D14）。包装 MCP 工具为 Tool 注册进 Registry（统一抽象红利）。

注册进同一 Registry → MCP 工具自动获权限引擎 + trace + plan mode 过滤。
load_mcp_server 需 mcp 包（optional [mcp]，pip install -e ".[mcp]"）。
除 tools 外，可选拉取 prompts / resources：不逐个注册，只各注册一个元工具
（use_prompt_<server> / read_mcp_resource_<server>），可用清单写进 description。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tools import Tool


def _extract_text(result: Any) -> str:
    """从 MCP 返回里只提取 text 块拼接（忽略 image/blob 等非文本块）。

    兼容三类结构：call_tool 的 .content、get_prompt 的 .messages、
    read_resource 的 .contents——每条优先取 .text，否则取 .content.text。
    若本身就是字符串则原样返回（兜底，兼容 stub/纯文本响应）。
    """
    if isinstance(result, str):
        return result
    parts: list[str] = []
    items = (
        getattr(result, "messages", None)
        or getattr(result, "contents", None)
        or getattr(result, "content", None)
        or []
    )
    if isinstance(items, str):
        items = [items]
    for item in items:
        text = getattr(item, "text", None)
        if not isinstance(text, str) or not text:
            text = getattr(getattr(item, "content", None), "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n\n".join(parts)


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
        return _extract_text(result)  # 只取 text 块，非文本块丢弃（不再 str() 一把梭）


class MCPPromptTool(Tool):
    """元工具：渲染 MCP prompt 模板（get_prompt）。read_only，plan 模式也可用。

    不逐个注册 prompt，只注册一个 use_prompt 元工具；可用模板清单写进 description。
    """

    def __init__(self, prompts, session, server_name: str):
        self.name = f"use_prompt_{server_name}"
        listing = ", ".join(
            f"{p.name}({', '.join(a.name for a in p.arguments)})" for p in prompts
        )
        self.description = f"渲染 MCP server 提供的 prompt 模板并返回现成话术。可用模板：{listing}"
        self.schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [p.name for p in prompts],
                    "description": "要渲染的模板名",
                },
                "arguments": {"type": "object", "description": "模板参数（可选）"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        self.read_only = True
        self._prompts = {p.name: p for p in prompts}
        self._session = session

    async def execute(self, name: str, arguments: dict | None = None, **_: Any) -> str:
        result = await self._session.get_prompt(name, arguments or {})
        return _extract_text(result)


class MCPResourceTool(Tool):
    """元工具：按 URI 读取 MCP resource（read_resource）。read_only，plan 模式也可用。"""

    def __init__(self, resources, session, server_name: str):
        self.name = f"read_mcp_resource_{server_name}"
        listing = ", ".join(r.uri for r in resources)
        self.description = f"读取 MCP server 暴露的资源（传 URI）。可用 URI：{listing}"
        self.schema = {
            "type": "object",
            "properties": {"uri": {"type": "string", "description": "资源 URI"}},
            "required": ["uri"],
            "additionalProperties": False,
        }
        self.read_only = True
        self._session = session

    async def execute(self, uri: str, **_: Any) -> str:
        result = await self._session.read_resource(uri)
        return _extract_text(result)


@dataclass
class MCPLoadResult:
    """load_mcp_server 返回值：该 server 的全部工具（含 prompt/resource 元工具）。"""

    tools: list[Tool] = field(default_factory=list)


async def load_mcp_server(
    name: str, command: str, args: list[str] | None = None, url: str = "", exit_stack=None
) -> MCPLoadResult:
    """启动 MCP server，返回 MCPLoadResult（tools + 可选 prompt/resource 元工具）。

    传输：url 非空走 streamable HTTP（云端/远程 server），否则走 stdio（本地子进程）。
    连接生命周期由调用方的 exit_stack 管理（常驻到 agent 退出），本函数不再自己开/关——
    否则返回的 wrapper 持有已关闭的 session，工具调用必失败。
    未装 mcp 时调用会 ImportError（提示 pip install -e ".[mcp]"）。
    prompt/resource 可选：server 没提供时跳过，不注册对应元工具。
    """
    from mcp import ClientSession

    if url:
        from mcp.client.streamable_http import streamablehttp_client

        read, write = await exit_stack.enter_async_context(streamablehttp_client(url))
    else:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=command, args=args or [])
        read, write = await exit_stack.enter_async_context(stdio_client(params))
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    resp = await session.list_tools()
    tools: list[Tool] = [MCPToolWrapper(t, session) for t in resp.tools]

    # prompt：可选，有就注册 use_prompt 元工具（失败跳过不报错）
    try:
        prompts = (await session.list_prompts()).prompts
    except Exception:
        prompts = []
    if prompts:
        tools.append(MCPPromptTool(prompts, session, name))

    # resource：可选，有就注册 read_mcp_resource 元工具
    try:
        resources = (await session.list_resources()).resources
    except Exception:
        resources = []
    if resources:
        tools.append(MCPResourceTool(resources, session, name))

    return MCPLoadResult(tools=tools)

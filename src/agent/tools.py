"""Tool 基类 + 注册表。

Tool.read_only 标记是 plan mode 门控依据（D7）。
ToolRegistry.schemas() 产出 OpenAI tool defs 供 Provider。
注册表过滤（readonly()）是 plan mode 第一层安全（D7）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """工具基类。execute 返回文本结果（回喂模型）。"""

    name: str = ""
    description: str = ""
    schema: dict = {}          # JSON schema，parameters.type 必须是 "object"
    read_only: bool = False    # plan mode 门控依据

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具，返回文本结果。异常由 loop.dispatch 捕获转 [error: ...] 回喂。"""
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def readonly(self) -> list[Tool]:
        """plan mode 用：只读工具 + ExitPlanMode（退出 plan 的通道）。"""
        return [t for t in self._tools.values() if t.read_only or t.name == "ExitPlanMode"]

    def schemas(self) -> list[dict]:
        """OpenAI tool defs: [{type:function, function:{name, description, parameters}}]。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

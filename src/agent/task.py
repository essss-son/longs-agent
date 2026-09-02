"""Task 子代理（简化子循环）：上下文隔离 + 工具白名单 + 模型路由。

不复用 AgentLoop，独立写一个最小子循环，方便后续做注入/拦截（steering 等）。
子代理：干净消息历史（不落盘）、只读白名单工具（Read/Glob/Grep）、
禁止写工具 / 禁止嵌套 Task / 不加载 MCP（防密钥外泄）。
工具结果不走信封（截断会丢失精度、子代理又无取回手段，残缺信息会污染
最终报告），只留 200k 字符硬截断防内存爆。
"""
from __future__ import annotations

from .builtin_tools import Glob, Grep, Read
from .messages import Message
from .tools import Tool
from .utils import truncate

# 子代理白名单（problems.md ALLOWED_TOOLS 对齐，本项目用 Read 代 LS，暂不含 TodoWrite）
ALLOWED_SUBAGENT_TOOLS = ("Read", "Glob", "Grep")

_SUBAGENT_SYSTEM = (
    "You are a read-only subagent. Analyze the codebase and return a structured report. "
    "You may ONLY use read-only tools: Read, Glob, Grep. Never write, edit, or run commands. "
    "When done, output the final report as plain text (no tool calls)."
)


class Task(Tool):
    """把子任务外包给只读子代理。Task 本身 read_only：plan 模式也可用。"""

    name = "Task"
    description = (
        "把子任务外包给只读子代理，避免主上下文被细节污染。"
        "子代理只读分析（Read/Glob/Grep）并返回报告，不能写文件、不能嵌套 Task、不能用 Bash。"
    )
    schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "子任务一句话描述"},
            "prompt": {"type": "string", "description": "给子代理的完整任务说明"},
            "subagent_type": {
                "type": "string",
                "enum": ["general", "explore", "summary", "plan"],
                "default": "general",
                "description": "子代理角色",
            },
            "model": {
                "type": "string",
                "enum": ["light", "main"],
                "default": "light",
                "description": "light 用轻量模型省 token，main 用主模型",
            },
        },
        "required": ["description", "prompt"],
        "additionalProperties": False,
    }
    read_only = True

    def __init__(self, provider, light_provider=None, max_steps: int = 20):
        self.provider = provider
        self.light_provider = light_provider  # None 时 light 退化用主模型
        self.max_steps = max_steps
        self._tools = {"Read": Read(), "Glob": Glob(), "Grep": Grep()}

    def _tool_schemas(self) -> list[dict]:
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

    async def execute(
        self,
        description: str,
        prompt: str,
        subagent_type: str = "general",
        model: str = "light",
        **_: object,
    ) -> str:
        provider = (
            self.light_provider if model == "light" and self.light_provider else self.provider
        )
        return await self._run_subagent(prompt, provider)

    async def _run_subagent(self, prompt: str, provider) -> str:
        """简化子循环：干净消息历史 + 白名单工具，最多 max_steps 步。"""
        messages: list[Message] = [Message("user", prompt)]
        tools = self._tool_schemas()
        for _ in range(self.max_steps):
            resp = await provider.chat(messages, tools=tools, system=_SUBAGENT_SYSTEM)
            messages.append(Message("assistant", resp.content, tool_calls=resp.tool_calls))
            if not resp.tool_calls:
                return resp.content or "(subagent returned empty)"
            for tc in resp.tool_calls:
                tool = self._tools.get(tc.name)
                if tool is None:
                    result = f"[error: tool '{tc.name}' not allowed in subagent]"
                else:
                    try:
                        result = await tool.execute(**tc.arguments)
                    except Exception as e:
                        result = f"[error: {type(e).__name__}: {e}]"
                    if not isinstance(result, str):
                        result = str(result)
                    # 子代理不走工具信封：截断会丢失精度，且子代理没有取回手段，
                    # 残缺信息会污染最终报告。只留 200k 字符硬截断防内存爆
                    # （与主循环 Bash 工具同款兜底），正常大小结果原样进上下文。
                    result = truncate(result, 200000)
                messages.append(
                    Message("tool", content=result, tool_call_id=tc.id, name=tc.name)
                )
        return messages[-1].content or "(subagent reached max steps)"

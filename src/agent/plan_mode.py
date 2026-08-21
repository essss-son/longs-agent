"""Plan mode（D7）。

双重安全（对说明书的修正 2）：
1. loop._active_tools() plan 模式只暴露 readonly（registry.readonly）——模型看不到写工具
2. dispatch 兜底：if mode==PLAN and not tool.read_only: return DENY——幻觉写工具也拦

ExitPlanMode 必须 read_only=True 才能通过过滤器。
plan 批准后立即落 todos（compaction 压不掉）+ 退出 plan 模式。
"""
from __future__ import annotations

import re
from typing import Any

from .todo import Todo, TodoStatus
from .tools import Tool


def parse_plan_to_todos(plan: str) -> list[Todo]:
    """解析 markdown checklist（- [ ] task / - [x] done）成 todos。

    无 checklist 时落单个 todo "Execute plan"（plan 仍有价值，只是无结构化步骤）。
    """
    todos: list[Todo] = []
    for line in plan.splitlines():
        m = re.match(r"\s*[-*]\s+\[([ x])\]\s+(.+)", line)
        if m:
            content = m.group(2).strip()
            status = TodoStatus.COMPLETED if m.group(1) == "x" else TodoStatus.PENDING
            todos.append(Todo(content=content, status=status))
    if not todos:
        todos = [Todo(content="Execute plan", status=TodoStatus.PENDING)]
    return todos


class ExitPlanMode(Tool):
    name = "ExitPlanMode"
    description = (
        "提交计划供用户审批。plan 参数为计划文本（markdown checklist 推荐）。"
        "调用此工具即结束 plan 模式，等待用户批准。"
    )
    schema = {
        "type": "object",
        "properties": {
            "plan": {"type": "string", "description": "计划文本"},
        },
        "required": ["plan"],
        "additionalProperties": False,
    }
    read_only = True  # 关键：read_only=True 才能通过 plan 模式注册表过滤

    async def execute(self, plan: str, **_: Any) -> str:
        # execute 不直接审批；loop.dispatch 检测到 ExitPlanMode 调用后触发 repl.approve_plan
        return f"[plan submitted]\n{plan}"


class EnterPlanMode(Tool):
    """主动进入 plan mode。进入后只读探索，完成后用 ExitPlanMode 提交计划。"""

    name = "EnterPlanMode"
    description = (
        "主动进入计划模式。进入后只读探索（Read/Glob/Grep），"
        "完成后用 ExitPlanMode 提交计划供审批。适用于：任务复杂或多文件改动、"
        "需求模糊有多种方案、有破坏性风险、需先探索代码才能确定做法。"
    )
    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    read_only = False  # 改变模式状态（非只读）；plan 模式下被过滤 + dispatch 兜底拦

    async def execute(self, **_: Any) -> str:
        # 真正的模式切换在 loop.dispatch 检测到 EnterPlanMode 后执行
        return "[entering plan mode]"

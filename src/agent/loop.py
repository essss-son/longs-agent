"""AgentLoop 主循环（D9：+ compaction 轮末检查）。

每次 LLM 请求前检查 should_compact，超 80% effective 则压缩（配对不破）。
plan 双重安全（D7）+ trace（D5）保持。
"""
from __future__ import annotations

from .messages import Message, ToolCall
from .permissions import Mode, PermissionConfig, PermissionEngine, Verdict
from .tools import ToolRegistry


class AgentLoop:
    # "只说不做"检测关键词：模型 finish=stop 无 tool_calls，但 content 含这些词说明
    # 它仍在描述计划（如 "Let me look at X"）而非真正完成，harness 应提示它动手。
    _CONTINUATION_WORDS = (
        "let me", "i will", "i'll", "let's", "lets",
        "look at", "looking at", "接下来", "下一步",
    )

    def __init__(
        self,
        provider,
        registry: ToolRegistry,
        session,
        permissions: PermissionEngine | None = None,
        permission_config: PermissionConfig | None = None,
        mode: Mode = Mode.MANUAL,
        repl=None,
        todo_store=None,
        compactor=None,
        system_prompt: str | None = None,
        max_steps: int = 50,
        loop_detection_threshold: int = 5,
    ):
        self.provider = provider
        self.registry = registry
        self.session = session
        self.permissions = permissions or PermissionEngine()
        self.permission_config = permission_config or PermissionConfig()
        self.mode = mode
        self.repl = repl
        self.todo_store = todo_store
        self.compactor = compactor
        self.system_prompt = system_prompt
        self.max_steps = max_steps  # 单轮最大 LLM 调用次数，超了优雅退出
        self.loop_detection_threshold = loop_detection_threshold  # 连续 N 次相同工具调用触发循环检测
        self.messages: list[Message] = []
        self._mode_before_plan: Mode | None = None  # 进入 plan 前的模式，批准后据此恢复
        self._last_tool_sig: str | None = None  # 上一轮 tool_calls 签名（循环检测）
        self._repeat_count = 0  # 连续相同签名次数

    def _active_tools(self) -> list[dict]:
        if self.mode == Mode.PLAN:
            active = self.registry.readonly()
        else:
            active = [t for t in self.registry.all() if t.name != "ExitPlanMode"]
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            for t in active
        ]

    def _current_system_prompt(self) -> str | None:
        """每轮请求前刷新：基础 system_prompt + 当前 todos 段。"""
        base = self.system_prompt
        if not self.todo_store:
            return base
        from .memory import _todos_block

        todos_b = _todos_block([t.to_dict() for t in self.todo_store.all()])
        if not todos_b:
            return base
        if base:
            return base + "\n\n" + todos_b
        return todos_b

    def _enter_plan(self) -> None:
        """进入 plan 模式：先记住当前模式，批准后据此恢复。"""
        self._mode_before_plan = self.mode
        self.mode = Mode.PLAN

    @staticmethod
    def _tool_calls_signature(tool_calls: list[ToolCall]) -> str:
        """tool_calls 稳定签名：并行调用排序后拼接，用于检测连续重复（死循环）。"""
        import json

        parts = sorted(
            json.dumps({"name": tc.name, "args": tc.arguments}, sort_keys=True, ensure_ascii=False)
            for tc in tool_calls
        )
        return "|".join(parts)

    @staticmethod
    def _continuation_intent(content: str | None) -> bool:
        """检测"只说不做"：content 暗示还要继续探索/动手，但模型没调工具。

        防御小模型"描述计划"与"实际调用工具"脱节——它输出 "Let me look at X" 却
        stop 了，harness 会误判为"任务完成"。命中时由 run_turn 注入提示让模型真正调工具。
        """
        if not content:
            return False
        low = content.lower()
        return any(w in low for w in AgentLoop._CONTINUATION_WORDS)

    async def run_turn(self, user_input: str, on_event=None) -> str:
        if self.compactor:
            self.compactor.reset()  # 新一轮对话：重置滞回，重新允许压缩
        user_msg = Message("user", user_input)
        self.messages.append(user_msg)
        self.session.append_message(user_msg)
        self.session.append_trace({"type": "turn_start", "data": {"user_input": user_input[:200]}})

        step_count = 0
        hinted = False  # "只说不做"提示是否已注入（限一次，避免死循环）
        while True:
            step_count += 1
            if step_count > self.max_steps:
                # 预算耗尽：优雅退出并汇报，而不是无限跑下去
                msg = (
                    f"[达到最大步数 {self.max_steps}，已停止。]"
                    "已完成的工作记录在上方对话和 trace 中，可据此继续。"
                )
                self.session.append_trace({"type": "max_steps", "data": {"steps": step_count}})
                return msg

            tools = self._active_tools()
            # 请求前压缩：超 80% effective 则 compact（配对不破）
            if self.compactor and self.compactor.should_compact(self.messages, tools):
                self.messages = await self.compactor.compact(self.messages, tools)
                self.session.append_trace(
                    {"type": "compaction", "data": {"remaining": len(self.messages)}}
                )

            # 每轮刷新 system_prompt 的 todos 段（TodoWrite 可能改了 todos）
            system = self._current_system_prompt()
            if on_event is not None:
                on_event("thinking")
                resp = await self.provider.stream(
                    self.messages, tools=tools, system=system,
                    on_delta=lambda c: on_event("delta", c),
                on_reasoning=lambda c: on_event("reasoning", c),
                )
            else:
                resp = await self.provider.chat(self.messages, tools=tools, system=system)
            assistant_msg = Message("assistant", resp.content, tool_calls=resp.tool_calls)
            self.messages.append(assistant_msg)
            self.session.append_message(assistant_msg)
            self.session.append_trace(
                {
                    "type": "llm_response",
                    "data": {
                        "model": resp.model,
                        "usage": resp.usage.to_dict(),
                        "finish": resp.finish_reason,
                        "tool_calls": len(resp.tool_calls or []),
                    },
                }
            )

            if not resp.tool_calls:
                # "只说不做"防御：模型 stop 但 content 还暗示要继续，注入提示让它真正调工具
                if not hinted and self._continuation_intent(resp.content):
                    hinted = True
                    hint = (
                        "[hint] You described a next step but did not actually call a tool. "
                        "Please call the corresponding tool (Read/Grep/Edit/Bash, etc.) directly "
                        "instead of just describing what you plan to do. If the task is truly "
                        "done, say Done or give your conclusion explicitly."
                    )
                    self.messages.append(Message("user", hint))
                    self.session.append_message(Message("user", hint))
                    self.session.append_trace(
                        {"type": "continuation_hint", "data": {"content": (resp.content or "")[:200]}}
                    )
                    continue
                return resp.content or ""

            # 循环检测：与上一轮 tool_calls 签名比较（同 name + 同 args 才算重复）
            sig = self._tool_calls_signature(resp.tool_calls)
            if sig == self._last_tool_sig:
                self._repeat_count += 1
            else:
                self._repeat_count = 1
                self._last_tool_sig = sig

            for tc in resp.tool_calls:
                if on_event is not None:
                    on_event("tool_start", tc.name, tc.arguments)
                # Write/Edit：执行前快照旧内容，执行后生成 diff
                old_snap = self._snapshot(tc) if on_event is not None else None
                tool_msg = await self.dispatch(tc)
                if on_event is not None:
                    on_event("tool_end", tc.name, tool_msg.content or "")
                    if old_snap is not None:
                        diff = self._make_diff(old_snap)
                        if diff is not None:
                            on_event("tool_diff", *diff)
                self.messages.append(tool_msg)
                self.session.append_message(tool_msg)

            # 死循环打断：连续 N 次相同 tool_calls，注入提示让模型换策略
            if self._repeat_count >= self.loop_detection_threshold:
                hint = (
                    f"[循环检测] 你已连续 {self._repeat_count} 次调用完全相同的工具和参数，"
                    "但没有得到新的进展。请停下来检查：1) 工具返回是否与预期不符 2) 是否该换一种查找方式 "
                    "3) 是否该直接说明卡住的原因。"
                )
                self.messages.append(Message("user", hint))
                self.session.append_message(Message("user", hint))
                self.session.append_trace(
                    {"type": "loop_detection", "data": {"repeat": self._repeat_count, "sig": sig[:200]}}
                )
                self._repeat_count = 0  # 重置，允许新一轮再检测
                self._last_tool_sig = None

    async def dispatch(self, tc: ToolCall) -> Message:
        tool = self.registry.get(tc.name)
        if tool is None:
            self.session.append_trace(
                {"type": "tool_call", "data": {"name": tc.name, "unknown": True}}
            )
            return Message(
                "tool",
                content=f"[error: unknown tool '{tc.name}']",
                tool_call_id=tc.id,
                name=tc.name,
            )

        # plan 模式兜底校验（双重安全第二层）
        if self.mode == Mode.PLAN and not tool.read_only:
            self.session.append_trace(
                {"type": "permission", "data": {"verdict": "DENY", "reason": "plan mode readonly"}}
            )
            return Message(
                "tool",
                content="[denied: plan mode only allows read-only tools]",
                tool_call_id=tc.id,
                name=tc.name,
            )

        # EnterPlanMode 特殊：主动进入 PLAN 模式（变严格，无需审批）
        if tc.name == "EnterPlanMode":
            self.session.append_trace(
                {"type": "tool_call", "data": {"name": "EnterPlanMode"}}
            )
            try:
                result = await tool.execute(**tc.arguments)
            except Exception as e:
                result = f"[error: {type(e).__name__}: {e}]"
            if not isinstance(result, str):
                result = str(result)
            self._enter_plan()
            self.session.append_trace(
                {"type": "mode_change", "data": {"mode": "PLAN", "via": "EnterPlanMode"}}
            )
            return Message(
                "tool", content="[entering plan mode]", tool_call_id=tc.id, name=tc.name
            )

        # ExitPlanMode 特殊：不走工具权限 ASK，直接执行 + 触发 plan 审批
        if tc.name == "ExitPlanMode":
            self.session.append_trace(
                {"type": "tool_call", "data": {"name": "ExitPlanMode"}}
            )
            try:
                result = await tool.execute(**tc.arguments)
            except Exception as e:
                result = f"[error: {type(e).__name__}: {e}]"
            if not isinstance(result, str):
                result = str(result)
            if self.repl is not None and self.todo_store is not None:
                plan = tc.arguments.get("plan", "")
                approved = await self.repl.approve_plan(plan)
                if approved:
                    from .plan_mode import parse_plan_to_todos

                    todos = parse_plan_to_todos(plan)
                    self.todo_store.set(todos)
                    self.mode = self._mode_before_plan or Mode.MANUAL
                    self._mode_before_plan = None
                    result = "[plan approved, exiting plan mode]"
                else:
                    result = "[plan rejected, staying in plan mode]"
            self.session.append_trace(
                {"type": "tool_result", "data": {"content_preview": result[:200]}}
            )
            return Message("tool", content=result, tool_call_id=tc.id, name=tc.name)

        # 权限检查
        verdict, reason = self.permissions.check(tc, self.permission_config, self.mode)
        if verdict == Verdict.DENY:
            self.session.append_trace(
                {"type": "permission", "data": {"verdict": "DENY", "reason": reason}}
            )
            return Message("tool", content=f"[denied: {reason}]", tool_call_id=tc.id, name=tc.name)
        if verdict == Verdict.ASK:
            if self.repl is None:
                self.session.append_trace(
                    {"type": "permission", "data": {"verdict": "DENY", "reason": "no repl"}}
                )
                return Message(
                    "tool",
                    content="[denied: no repl to confirm, default deny]",
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            decision = await self.repl.ask_permission(tc)
            if decision == "n":
                self.session.append_trace(
                    {"type": "permission", "data": {"verdict": "DENY", "reason": "user"}}
                )
                return Message("tool", content="[denied by user]", tool_call_id=tc.id, name=tc.name)
            if decision == "always":
                self.permission_config.always_grants.add(tc.name)
                self.session.append_trace(
                    {"type": "permission", "data": {"verdict": "ALLOW", "reason": "always"}}
                )
            else:
                self.session.append_trace(
                    {"type": "permission", "data": {"verdict": "ALLOW", "reason": "user"}}
                )

        # 执行
        self.session.append_trace(
            {"type": "tool_call", "data": {"name": tc.name, "arguments": tc.arguments}}
        )
        try:
            result = await tool.execute(**tc.arguments)
        except Exception as e:
            result = f"[error: {type(e).__name__}: {e}]"
            self.session.append_trace(
                {"type": "error", "data": {"tool": tc.name, "error": str(e)[:200]}}
            )
        if not isinstance(result, str):
            result = str(result)
        self.session.append_trace(
            {"type": "tool_result", "data": {"content_preview": result[:200]}}
        )
        return Message("tool", content=result, tool_call_id=tc.id, name=tc.name)

    @staticmethod
    def _snapshot(tc):
        """Write/Edit 执行前读旧文件内容，返回 (file_path, old_text) 或 None。"""
        if tc.name not in ("Write", "Edit"):
            return None
        fp = tc.arguments.get("file_path")
        if not fp:
            return None
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return (fp, f.read())
        except (FileNotFoundError, OSError):
            return (fp, "")  # 新文件

    @staticmethod
    def _make_diff(old_snap):
        """执行后读新内容，生成带行号的 diff，返回 (file_path, lines, added, removed) 或 None。

        每行 (sign, lineno, text)：- 用旧文件行号，+ 和 context 用新文件行号。
        """
        import difflib
        import re

        fp, old_text = old_snap
        try:
            with open(fp, "r", encoding="utf-8") as f:
                new_text = f.read()
        except (FileNotFoundError, OSError):
            new_text = ""
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        added = removed = 0
        out = []
        old_no = new_no = 0
        for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=2):
            if line.startswith(("+++", "---")):
                continue
            if line.startswith("@@"):
                m = re.search(r"-(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?", line)
                if m:
                    old_no = int(m.group(1)) - 1
                    new_no = int(m.group(2)) - 1
                continue
            if line.startswith("+"):
                new_no += 1
                added += 1
                out.append(("+", new_no, line[1:]))
            elif line.startswith("-"):
                old_no += 1
                removed += 1
                out.append(("-", old_no, line[1:]))
            else:
                old_no += 1
                new_no += 1
                out.append((" ", new_no, line[1:]))
        return (fp, out, added, removed) if out else None

"""prompt_toolkit REPL（D9：+ /compact /context token 估算）。

D8：/mode。D9：/compact 手动压缩 /context 显示 token 用量。
"""
from __future__ import annotations

from pathlib import Path

from .loop import AgentLoop
from .permissions import Mode
from .session import SessionStore


class REPL:
    def __init__(self, loop: AgentLoop, mode: str = "NORMAL"):
        self.loop = loop
        self.mode = mode
        self._session = None

    async def run(self) -> None:
        from prompt_toolkit import PromptSession

        self._session = PromptSession()
        self.loop.repl = self
        print("输入 /exit 退出，/help 查看命令，/plan 计划，/mode 切换，/resume 恢复，/compact 压缩")
        while True:
            try:
                user_input = await self._session.prompt_async("you> ", bottom_toolbar=self._toolbar)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                if await self._slash(user_input) == "exit":
                    break
                continue
            try:
                answer = await self.loop.run_turn(user_input)
                print(f"assistant> {answer}")
                self._render_todos()
            except KeyboardInterrupt:
                print("\n[interrupted]")
            except asyncio.CancelledError:
                print("\n[interrupted]")
            except Exception as e:
                print(f"\n[error: {type(e).__name__}: {e}]")

    def _toolbar(self) -> str:
        mode_name = getattr(self.loop, "mode", None)
        mode_str = mode_name.name if mode_name else self.mode
        return f" mode={mode_str} | session={self.loop.session.sid} | /exit /help /plan /mode /resume /compact "

    async def _slash(self, cmd: str) -> str | None:
        cmd_lower = cmd.lower()
        if cmd_lower in ("/exit", "/quit"):
            return "exit"
        if cmd_lower == "/help":
            print(
                "命令: /exit /help /plan /mode /resume [sid] /compact /context /cost\n"
                "（后续: /model /trace /skill）"
            )
            return None
        if cmd_lower == "/plan":
            self.loop._enter_plan()
            print("[plan mode] 只读探索，ExitPlanMode 提交计划供审批")
            return None
        if cmd_lower == "/mode":
            if self.loop.mode == Mode.PLAN:
                print("[plan mode] 用 ExitPlanMode 提交计划退出，或继续探索")
            elif self.loop.mode == Mode.MANUAL:
                self.loop.mode = Mode.AUTO
            else:
                self.loop.mode = Mode.MANUAL
            print(f"[mode] {self.loop.mode.name}（hard deny 仍生效）")
            return None
        if cmd_lower.startswith("/resume"):
            await self._resume(cmd)
            return None
        if cmd_lower == "/compact":
            await self._compact()
            return None
        if cmd_lower == "/context":
            self._show_context()
            return None
        if cmd_lower == "/cost":
            self._show_cost()
            return None
        if cmd_lower == "/trace":
            print(self._trace_view())
            return None
        if cmd_lower.startswith("/model"):
            await self._switch_model(cmd)
            return None
        print(f"未知命令: {cmd}（/help 查看）")
        return None

    async def _compact(self) -> None:
        if not self.loop.compactor:
            print("无 compactor")
            return
        before = len(self.loop.messages)
        self.loop.messages = await self.loop.compactor.compact(
            self.loop.messages, self.loop._active_tools()
        )
        print(f"[compacted] {before} → {len(self.loop.messages)} 条消息")

    def _show_context(self) -> None:
        from .utils import estimate_messages_tokens

        tools = self.loop._active_tools()
        used = estimate_messages_tokens(self.loop.messages, tools)
        win = getattr(self.loop.provider, "context_window", 32768)
        pct = 100 * used // max(win, 1)
        print(f"token: {used}/{win} ({pct}%)，{len(self.loop.messages)} 条消息")

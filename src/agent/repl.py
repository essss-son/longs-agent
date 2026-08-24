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

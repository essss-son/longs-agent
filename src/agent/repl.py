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
                "命令: /exit /help /plan /mode /resume [sid] /compact /context /cost /rename <名字>\n"
                "（后续: /model /trace /skill /undo /rewind [n]）"
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
        if cmd_lower == "/undo":
            self._undo()
            return None
        if cmd_lower.startswith("/rewind"):
            await self._rewind(cmd)
            return None
        if cmd_lower.startswith("/rename"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print("usage: /rename <名字>")
                return None
            self._rename(parts[1].strip())
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

    async def _resume(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=1)
        sid = parts[1].strip() if len(parts) > 1 else None
        if sid:
            self._load_session(sid)
            return
        sids = SessionStore.list_sessions()
        if not sids:
            print("无历史会话")
            return
        print("历史会话（最近修改在前）：")
        for i, s in enumerate(sids[:5]):
            name = SessionStore(sid=s).get_name()
            print(f"  {i + 1}. {s}  {name}" if name else f"  {i + 1}. {s}")
        try:
            choice = await self._session.prompt_async("选择恢复 (1-5，回车取消): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        choice = choice.strip()
        if choice.isdigit() and 1 <= int(choice) <= min(5, len(sids)):
            self._load_session(sids[int(choice) - 1])
        else:
            print("已取消")

    def _load_session(self, sid: str) -> None:
        new_session = SessionStore(sid=sid)
        msgs = new_session.read_messages()
        if not msgs:
            print(f"会话 {sid} 无消息或不存在")
            return
        self.loop.session = new_session
        self.loop.messages = msgs
        self.loop._file_hashes = new_session.read_file_hashes()  # 乐观锁状态随会话恢复
        name = new_session.get_name()
        suffix = f"（{name}）" if name else ""
        print(f"resume 会话 {sid}{suffix}，恢复 {len(msgs)} 条消息")
        self._render_todos()

    def _render_todos(self) -> None:
        todos = self.loop.session.read_todos()
        if not todos:
            return
        print("\n[todos]")
        marks = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
        for i, t in enumerate(todos):
            status = t.get("status", "pending")
            mark = marks.get(status, "[ ]")
            label = t.get("active_form") or t.get("content", "")
            print(f"  {i + 1}. {mark} {label}")
        print()

    def _show_cost(self) -> None:
        from .trace import TraceStore

        cost = TraceStore(self.loop.session.trace_path).cost()
        print(
            f"cost: prompt={cost['prompt_tokens']} completion={cost['completion_tokens']} "
            f"total={cost['total_tokens']}"
        )

    def _trace_view(self) -> str:
        from .trace import TraceStore

        return TraceStore(self.loop.session.trace_path).timeline_view()

    def _rename(self, name: str) -> None:
        """重命名当前会话（display name，写 meta.json）。sid 目录名不变。"""
        self.loop.session.set_name(name)
        print(f"[rename] 会话 {self.loop.session.sid} 命名为: {name}")

    def _undo(self) -> None:
        """回滚最近一次 Write/Edit（三线：文件 + todo + messages）。"""
        msg = self.loop.session.undo_last_write()
        self._sync_after_rollback()
        print(msg)

    async def _rewind(self, cmd: str) -> None:
        """回退到某条用户消息处理完成后的状态（用户消息粒度）。"""
        parts = cmd.split(maxsplit=1)
        targets = self.loop.session.list_rewind_targets()
        if not targets:
            print("(没有可回退的用户消息)")
            return
        # /rewind <n>：n 是第几条用户消息（1-based，对应下面列表的编号）
        if len(parts) >= 2 and parts[1].strip().isdigit():
            n = int(parts[1].strip())
            if not (1 <= n <= len(targets)):
                print(f"(编号需在 1~{len(targets)} 之间)")
                return
            self._do_rewind(targets[n - 1], n)
            return
        # 不带参数：交互列出选项
        print("可回退到的用户消息（回退到最后一条 = 无变化）：")
        for i, t in enumerate(targets, 1):
            print(f"  {i}. {t['preview']}")
        try:
            ans = await self._session.prompt_async("选择 (1-N，回车取消): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        ans = ans.strip()
        if ans.isdigit() and 1 <= int(ans) <= len(targets):
            self._do_rewind(targets[int(ans) - 1], int(ans))
        else:
            print("已取消")

    def _do_rewind(self, target: dict, n: int) -> None:
        msg = self.loop.session.rewind_to_user(target["idx"])
        self._sync_after_rollback()
        print(f"[rewind] 回到第 {n} 条消息：{msg}")

    def _sync_after_rollback(self) -> None:
        """回滚后同步内存态：loop.messages 重读、todo_store 重载。"""
        self.loop.messages = self.loop.session.read_messages()
        if self.loop.todo_store is not None:
            self.loop.todo_store.load()
        self._render_todos()

    async def _switch_model(self, cmd: str) -> None:
        """热切换供应商：换 loop.provider（归一化历史可重序列化）。"""
        from .config import Config
        from .provider import AnthropicProvider, OpenAICompatibleProvider

        parts = cmd.split(maxsplit=1)
        alias = parts[1].strip() if len(parts) > 1 else None
        if not alias:
            print("usage: /model <alias>")
            return
        cfg = Config.load()
        m = cfg.get(alias)
        key = cfg.api_key(alias)
        if not m or not key:
            print(f"unknown model or missing api_key: {alias}")
            return
        if m.provider == "anthropic":
            new_provider = AnthropicProvider(
                api_key=key, model=m.model, context_window=m.context_window, max_tokens=m.max_tokens
            )
        else:
            new_provider = OpenAICompatibleProvider(
                base_url=m.base_url, api_key=key, model=m.model, context_window=m.context_window
            )
        self.loop.provider = new_provider
        if self.loop.compactor:
            self.loop.compactor.provider = new_provider
        print(f"[model] switched to {m.model} @ {m.base_url}")

    async def ask_permission(self, tool_call) -> str:
        from .utils import render_diff

        print(f"\n[ask permission] {tool_call.name}")
        if tool_call.name == "Bash":
            print(f"  command: {tool_call.arguments.get('command', '')}")
        elif tool_call.name in ("Write", "Edit"):
            fp = tool_call.arguments.get("file_path", "")
            print(f"  file: {fp}")
            try:
                old = Path(fp).read_text(encoding="utf-8")
            except Exception:
                old = ""
            if tool_call.name == "Write":
                new = tool_call.arguments.get("content", "")
                diff = render_diff(old, new)
                print(f"  diff:\n{diff[:2000]}")
            else:
                print(f"  old_string: {tool_call.arguments.get('old_string', '')[:200]}")
                print(f"  new_string: {tool_call.arguments.get('new_string', '')[:200]}")
        else:
            print(f"  args: {tool_call.arguments}")
        try:
            ans = await self._session.prompt_async("(y/n/always): ")
        except (EOFError, KeyboardInterrupt):
            return "n"
        ans = ans.strip().lower()
        if ans == "always":
            return "always"
        return "y" if ans == "y" else "n"

    async def approve_plan(self, plan: str) -> bool:
        print("\n[plan for approval]")
        print(plan[:3000])
        try:
            ans = await self._session.prompt_async("(a)ccept / (r)eject: ")
        except (EOFError, KeyboardInterrupt):
            return False
        return ans.strip().lower() == "a"

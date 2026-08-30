"""全屏 TUI Application（稳定版：logo 固定 + 对话区 BufferControl 滚动）。

布局（HSplit，从上到下）：
  logo 固定顶部（有色 +-+| 边框，左 logo 右提示）→ 对话区（BufferControl 可滚轮翻历史）
  → spinner → 输入框上横线（会话名）→ 输入框（/ 补全）
对话区可滚轮翻历史；logo 固定不动（TUI 常态）。
"""
from __future__ import annotations

import asyncio
import time

from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.layout.scrollable_pane import ScrollablePane

from ..permissions import Mode
from .completer import SlashCompleter


class _LogoProcessor(Processor):
    """给 buffer 的 logo 行复刻 _logo_ft 的样式，对话行用 history 样式。"""

    def __init__(self, logo_ft):
        self._lines = self._split_lines(logo_ft)
        # 给每行 logo 文字加青→紫渐变色（#00ffff → #ff00ff）
        logo_frags = [
            (i, j)
            for i, line in enumerate(self._lines)
            for j, (s, _) in enumerate(line)
            if s == "class:logo"
        ]
        n = len(logo_frags)
        for k, (i, j) in enumerate(logo_frags):
            t = k / (n - 1) if n > 1 else 0.0
            r = int((255 - 0) * t)  # 0 → 255
            g = int(255 - (255 - 0) * t)  # 255 → 0
            b = 255
            _, text = self._lines[i][j]
            self._lines[i][j] = (f"fg:#{r:02x}{g:02x}{b:02x} bold", text)

    @staticmethod
    def _split_lines(fragments):
        lines = [[]]
        for style, text in fragments:
            for i, part in enumerate(text.split("\n")):
                if i:
                    lines.append([])
                if part:
                    lines[-1].append((style, part))
        return lines

    def apply_transformation(self, ti):
        if ti.lineno < len(self._lines):
            return Transformation(fragments=list(self._lines[ti.lineno]))
        text = "".join(f[1] for f in ti.fragments)
        if text.startswith("  ⚙ "):
            return Transformation(fragments=[("class:tool-start", text)])
        if text.startswith("  → "):
            return Transformation(fragments=[("class:tool-result", text)])
        if text.startswith("  💭 "):
            return Transformation(fragments=[("class:reasoning", text)])
        if text.startswith("  ⚠ "):
            return Transformation(fragments=[("class:ask", text)])
        if text.startswith("● 你："):
            return Transformation(fragments=[("class:user-prompt", text)])
        if text.startswith("  ✎ "):
            return Transformation(fragments=[("class:diff-head", text)])
        if text.startswith("  + "):
            return Transformation(fragments=[("class:diff-add", text)])
        if text.startswith("  - "):
            return Transformation(fragments=[("class:diff-del", text)])
        return Transformation(fragments=[("class:history", text)])


class _HistoryControl(BufferControl):
    """BufferControl + 拦截滚轮：滚轮改 ScrollablePane.vertical_scroll，而非 buffer cursor。

    cursor 驱动的滚动在长内容时会被 Window.do_scroll 的 cursor clamp 拉回底部，手动滚动失效。
    改用 ScrollablePane + 自管 vertical_scroll 规避。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pane = None  # 由 TUIApp._build_layout 构造后回填
        self._app = None  # TUIApp 实例，供 invalidate

    def mouse_handler(self, mouse_event):
        from prompt_toolkit.mouse_events import MouseEventType

        et = mouse_event.event_type
        if et == MouseEventType.SCROLL_UP:
            self._scroll(-3)
            return None  # 不回落 Window._scroll_up（避免改 buffer cursor）
        if et == MouseEventType.SCROLL_DOWN:
            self._scroll(3)
            return None
        return super().mouse_handler(mouse_event)

    def _scroll(self, delta):
        pane = getattr(self, "_pane", None)
        app = getattr(self, "_app", None)
        if pane is not None:
            pane.vertical_scroll += delta  # 渲染时由 _HistoryPane.write_to_screen clamp 到合法范围
            if app is not None:
                app._invalidate()


class _HistoryPane(ScrollablePane):
    """ScrollablePane + cursor 不驱动（keep_cursor_visible=False）+ 渲染前安全 clamp 防越界。"""

    def write_to_screen(self, screen, mouse_handlers, write_position, parent_style, erase_bg, z_index):
        # super 前先算 virtual_height 并 clamp vertical_scroll，防 data_buffer 越界 IndexError
        show_sb = self.show_scrollbar()
        vw = write_position.width - (1 if show_sb else 0)
        vh = self.content.preferred_height(vw, self.max_available_height).preferred
        vh = max(vh, write_position.height)
        self._max_scroll = max(0, vh - write_position.height)
        if self.vertical_scroll > self._max_scroll:
            self.vertical_scroll = self._max_scroll
        elif self.vertical_scroll < 0:
            self.vertical_scroll = 0
        super().write_to_screen(screen, mouse_handlers, write_position, parent_style, erase_bg, z_index)


class TUIApp:
    """全屏 TUI 控制器，实现 REPL 接口供 loop.repl 注入。"""

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, loop, version: str = "0.1.0"):
        from prompt_toolkit.buffer import Buffer

        self.loop = loop
        self.version = version
        self.history_buffer = Buffer(name="history")
        self.thinking = False
        self._spinner_idx = 0
        self._app = None
        self._min_width = 80
        self._start = time.time()
        self._ctx_n = -1
        self._ctx_used = 0
        self._ctx_win = 32768
        self._reasoning_buf = ""
        self._pending_ask = None
        self._pending_plan_approval = None
        self._current_task = None  # 当前运行的生成任务，Ctrl+C 取消用
        self._thoughts: list[str] = []
        self._reasoning_t0 = None

    # ---- 文本片段 ----
    def _logo_ft(self):
        """单栏 logo 框：收紧到 logo 宽度；version 右下角灰色斜体。"""
        from .logo import _render_logo_art

        logo_lines = _render_logo_art("slant")
        version_text = f"version: v{self.version}"
        logo_w = max(self._disp_width(l) for l in logo_lines)
        version_w = self._disp_width(version_text)
        inner_w = max(logo_w, version_w)  # 内宽取 logo 与 version 较大者，自然缩放
        width = inner_w + 4  # 2 边框 + 左右各 1 padding

        ft: list = []
        ft.append(("class:logo-box", "+" + "-" * (width - 2) + "+\n"))
        for line in logo_lines:
            ft.append(("class:logo-box", "| "))
            ft.append(("class:logo", self._pad(line, inner_w)))
            ft.append(("class:logo-box", " |\n"))
        # version 右下角：右对齐，灰色斜体
        ft.append(("class:logo-box", "| "))
        ft.append(("class:logo-version", self._pad("", inner_w - version_w) + version_text))
        ft.append(("class:logo-box", " |\n"))
        ft.append(("class:logo-box", "+" + "-" * (width - 2) + "+\n"))
        return ft

    def _spinner_ft(self):
        if self.thinking:
            s = self.SPINNER[self._spinner_idx]
            buf = self._reasoning_buf
            if buf.strip():
                # 取最后一行（最新思考），截断 80 + 省略号代表还有更多
                line = buf.splitlines()[-1].strip()
                line = (line[:77] + "...") if len(line) > 77 else (line + "...")
                return [("class:spinner", f" {s} think: "), ("class:reasoning", line)]
            return [("class:spinner", f" {s} think...")]
        return [("class:dim", "")]

    # ---- 布局/样式 ----
    def _build_layout(self):
        from prompt_toolkit.layout.containers import HSplit, VSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.dimension import D
        from prompt_toolkit.layout.layout import Layout

        # logo 文本作为 buffer 前缀，与对话一起在 BufferControl 中滚动（滚轮整体翻）
        logo_ft = self._logo_ft()
        if not self.history_buffer.text:
            self.history_buffer.text = "".join(t for _, t in logo_ft) + "\n\n"
            self.history_buffer.cursor_position = len(self.history_buffer.text)
        control = _HistoryControl(
            buffer=self.history_buffer,
            focusable=False,
            input_processors=[_LogoProcessor(logo_ft)],
        )
        self._history_window = Window(control, wrap_lines=True)
        self._history_pane = _HistoryPane(
            self._history_window, keep_cursor_visible=False, show_scrollbar=True
        )
        control._pane = self._history_pane
        control._app = self
        root = HSplit([
            self._history_pane,
            Window(FormattedTextControl(self._spinner_ft), height=D.exact(1)),
            Window(FormattedTextControl(lambda: self._input_box_ft()), height=D.exact(1), style="class:logo-box"),
            VSplit([
                Window(FormattedTextControl(text=[("class:prompt", "❯ ")]), width=D.exact(2), style="class:input-row"),
                Window(BufferControl(buffer=self._input_buffer_with_completer()), style="class:input-row"),
            ], height=D.exact(1)),
            Window(FormattedTextControl(lambda: self._bot_line_ft()), height=D.exact(1), style="class:logo-box"),
            Window(FormattedTextControl(self._status_ft), height=D.exact(2)),
        ])
        layout = Layout(root)
        layout.focus(self._input_buffer_with_completer())
        return layout

    def _input_buffer_with_completer(self):
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.history import InMemoryHistory

        if not hasattr(self, "_input_w_completer"):
            buf = Buffer(
                name="input",
                completer=self._make_completer(),
                history=InMemoryHistory(),
                complete_while_typing=False,
            )

            def _accept(buf) -> bool:
                self._accept_input(buf)
                return False

            buf.accept_handler = _accept
            self._input_w_completer = buf
        return self._input_w_completer

    def _make_completer(self):
        from ..config import Config
        from ..session import SessionStore

        cfg = Config.load()
        models = list(cfg.models.keys()) if cfg.models else []
        return SlashCompleter(
            list_sessions=lambda: SessionStore.list_sessions(),
            list_models=lambda: models,
        )

    def _styles(self):
        from prompt_toolkit.styles import Style

        return Style([
            ("logo", "fg:ansimagenta bold"),
            ("logo-box", "fg:ansibrightblack"),
            ("logo-side", "fg:ansibrightblack"),
            ("logo-version", "fg:ansibrightblack italic"),
            ("spinner", "fg:ansicyan bold"),
            ("history", "fg:ansiwhite"),
            ("input-row", "fg:ansibrightgreen"),
            ("prompt", "fg:ansibrightgreen bold"),
            ("user-prompt", "fg:ansibrightgreen"),
            ("status-dim", "fg:ansibrightblack"),
            ("status-model", "fg:ansicyan"),
            ("status-warn", "fg:ansired bold"),
            ("tool-start", "fg:ansicyan"),
            ("tool-result", "fg:ansibrightblack"),
            ("reasoning", "fg:ansibrightblack italic"),
            ("ask", "fg:ansiyellow bold"),
            ("diff-head", "fg:ansiyellow bold"),
            ("diff-add", "fg:ansigreen"),
            ("diff-del", "fg:ansired"),
            ("mode-manual", "fg:ansiyellow bold"),
            ("mode-plan", "fg:ansicyan bold"),
            ("mode-auto", "fg:ansigreen bold"),
            ("error", "fg:ansired bold"),
            ("dim", "fg:ansibrightblack"),
        ])

    def _keybindings(self):
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys

        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            self._accept_input(self._input_buffer_with_completer())

        @kb.add("s-tab")
        def _(event):
            self._cycle_mode()

        @kb.add(Keys.ScrollUp)
        def _(event):
            pane = getattr(self, "_history_pane", None)
            if pane is not None:
                pane.vertical_scroll = max(0, pane.vertical_scroll - 3)
                self._invalidate()

        @kb.add(Keys.ScrollDown)
        def _(event):
            pane = getattr(self, "_history_pane", None)
            if pane is not None:
                pane.vertical_scroll += 3
                self._invalidate()

        @kb.add("c-c")
        def _(event):
            """Ctrl+C：取消当前正在生成的回复，不退出应用。"""
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()

        return kb

    # ---- 输入处理 ----
    def _accept_input(self, buf) -> None:
        """统一提交：存历史（非 ask 决策）→ reset（重置 working lines，触发 history 重新加载）→ 走 _submit_input。

        关键：append_to_history 只存到 history 对象，不更新 _working_lines（history_backward 用的列表）。
        必须调 reset 触发 load_history_if_not_yet_loaded，下次重绘才把新条目加载进 _working_lines。
        """
        text = buf.text
        if text.strip() and not getattr(self, "_pending_ask", None):
            buf.append_to_history()
        buf.reset()  # 清空文本 + 重置 _working_lines，下次重绘异步加载 history（含新条目）
        self._submit_input(text)

    def _submit_input(self, text: str) -> None:
        """统一输入提交入口：ask 等待中 → plan 审批中 → 否则走对话。"""
        if getattr(self, "_pending_ask", None):
            _tc, fut = self._pending_ask
            decision = self._parse_ask(text)
            if not fut.done():
                fut.set_result(decision)
            return
        if getattr(self, "_pending_plan_approval", None):
            approved = text.strip().lower() == "a"
            if not self._pending_plan_approval.done():
                self._pending_plan_approval.set_result(approved)
            return
        asyncio.create_task(self._handle_input(text))

    @staticmethod
    def _parse_ask(text: str) -> str:
        t = text.strip().lower()
        if t in ("a", "always"):
            return "always"
        if t in ("y", "yes"):
            return "y"
        return "n"  # 空/其他默认拒绝

    async def _handle_input(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith("/"):
            await self._slash(text)
            return
        self._append_raw(f"● 你：{text}\n\n")
        self._ai_started = False
        task = asyncio.current_task()
        self._current_task = task
        try:
            self.thinking = True
            self._invalidate()
            await self.loop.run_turn(text, on_event=self._on_event)
        except asyncio.CancelledError:
            self._append_raw("\n[interrupted]\n\n", flush=True)
        except Exception as e:
            self._append_raw(f"\n● error：{type(e).__name__}: {e}\n\n")
        finally:
            if getattr(self, "_ai_started", False):
                self._append_raw("\n\n", flush=True)
            self.thinking = False
            self._current_task = None
            self._invalidate()

    def _on_event(self, typ: str, *args) -> None:
        if typ == "reasoning":
            # 思考累积到 buf，首个 content delta 时只打印标记行（默认折叠，/thought 查全文）
            if self._reasoning_t0 is None:
                self._reasoning_t0 = time.time()
            self._reasoning_buf += args[0] if args else ""
            return
        if typ == "tool_start":
            name = args[0] if args else ""
            arguments = args[1] if len(args) > 1 else {}
            self._normalize_trailing(2)  # 工具块上方保证 1 空行
            self._append_raw(f"  ⚙ {name}: {self._fmt_tool_args(arguments)}\n\n", flush=True)  # 下方留 1 空行
            return
        if typ == "tool_end":
            result = args[1] if len(args) > 1 else ""
            self._append_raw(f"  → {self._fmt_tool_result(result)}\n\n", flush=True)  # 下方留 1 空行
            self._ai_started = False  # 下一轮 content 重新加 ● 前缀
            return
        if typ == "tool_diff":
            fp = args[0] if args else ""
            lines = args[1] if len(args) > 1 else []
            added = args[2] if len(args) > 2 else 0
            removed = args[3] if len(args) > 3 else 0
            self._append_raw(f"  ✎ {fp} (+{added} -{removed})\n")
            show = lines[:40]
            max_no = max((no for _, no, _ in show), default=0)
            w = len(str(max_no)) if max_no else 1
            for sign, no, text in show:
                self._append_raw(f"  {sign} {no:>{w}} {text}\n")
            if len(lines) > 40:
                self._append_raw(f"  ... ({len(lines) - 40} more)\n")
            self._normalize_trailing(2)  # diff 块下方留 1 空行
            self._invalidate()
            return
        if typ == "delta":
            chunk = args[0] if args else ""
            if not getattr(self, "_ai_started", False):
                # 思考过程已在 spinner 行（_spinner_ft）动态展示，这里只存全文供 /thought
                if self._reasoning_buf:
                    self._thoughts.append(self._reasoning_buf)
                    self._reasoning_buf = ""
                    self._reasoning_t0 = None
                self._append_raw("● longs-agent：", flush=True)
                self._ai_started = True
                self.thinking = False  # 思考结束，think 行清空
                self._invalidate()
            self._append_raw(chunk, flush=True)

    async def _slash(self, cmd: str) -> None:
        from ..permissions import Mode

        c = cmd.lower().strip()
        if c in ("/exit", "/quit"):
            self._append_raw("\n[exit]\n")
            if self._app:
                self._app.exit()
            return
        if c == "/help":
            self._append_raw("\n命令: /exit /help /plan /mode /model /resume /compact /context /cost /trace /thought /rename /undo /rewind [n]\n\n")
            return
        if c == "/plan":
            self.loop._enter_plan()
            self._append_raw("\n[plan mode] 只读探索，ExitPlanMode 提交计划\n")
            self._invalidate()
            return
        if c == "/mode":
            self._cycle_mode()
            return
        if c == "/context":
            from ..utils import estimate_messages_tokens

            used = estimate_messages_tokens(self.loop.messages, self.loop._active_tools())
            win = getattr(self.loop.provider, "context_window", 32768)
            self._append_raw(f"\n[token] {used}/{win} ({100*used//max(win,1)}%)，{len(self.loop.messages)} 条消息\n")
            return
        if c == "/compact":
            if self.loop.compactor:
                before = len(self.loop.messages)
                self.loop.messages = await self.loop.compactor.compact(
                    self.loop.messages, self.loop._active_tools()
                )
                self._append_raw(f"\n[compacted] {before} → {len(self.loop.messages)} 条\n")
            return
        if c == "/cost":
            from ..trace import TraceStore

            cost = TraceStore(self.loop.session.trace_path).cost()
            self._append_raw(
                f"\n[cost] prompt={cost['prompt_tokens']} completion={cost['completion_tokens']} total={cost['total_tokens']}\n"
            )
            return
        if c == "/trace":
            from ..trace import TraceStore

            self._append_raw("\n" + TraceStore(self.loop.session.trace_path).timeline_view() + "\n")
            return
        if c == "/undo":
            self._undo()
            return
        if c.startswith("/rewind"):
            await self._rewind(cmd)
            return
        if c == "/thought":
            if not getattr(self, "_thoughts", []):
                self._append_raw("\n(无思考记录)\n")
                return
            last = self._thoughts[-1]
            self._append_raw(f"\n[思考 {len(self._thoughts)}/{len(self._thoughts)}] 最近一条:\n{last}\n")
            return
        if c.startswith("/resume"):
            parts = cmd.split(maxsplit=1)
            sid = parts[1].strip() if len(parts) > 1 else None
            if sid:
                self._load_session(sid)
            else:
                from ..session import SessionStore

                sids = SessionStore.list_sessions()[:8]
                if sids:
                    lines = []
                    for s in sids:
                        name = SessionStore(sid=s).get_name()
                        lines.append(f"  {s}  {name}" if name else f"  {s}")
                    self._append_raw("\n历史会话:\n" + "\n".join(lines) + "\n（用 /resume <sid> 恢复）\n")
                else:
                    self._append_raw("\n无历史会话\n")
            return
        if c.startswith("/rename"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                self._append_raw("\nusage: /rename <名字>\n")
                return
            name = parts[1].strip()
            self.loop.session.set_name(name)
            self._append_raw(f"\n[rename] 会话 {self.loop.session.sid} 命名为: {name}\n")
            self._invalidate()
            return
        if c.startswith("/model"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                self._append_raw("\nusage: /model <alias>\n")
                return
            await self._switch_model(parts[1].strip())
            return
        self._append_raw(f"\n未知命令: {cmd}（/help）\n")

    async def _switch_model(self, alias: str) -> None:
        from ..config import Config
        from ..provider import OpenAICompatibleProvider

        cfg = Config.load()
        m = cfg.get(alias)
        key = cfg.api_key(alias)
        if not m or not key:
            self._append_raw(f"\n未知模型或缺 api_key: {alias}\n")
            return
        self.loop.provider = OpenAICompatibleProvider(
            base_url=m.base_url, api_key=key, model=m.model, context_window=m.context_window
        )
        if self.loop.compactor:
            self.loop.compactor.provider = self.loop.provider
        self._append_raw(f"\n[model] 切换到 {m.model}\n")
        self._invalidate()

    def _load_session(self, sid: str) -> None:
        from ..session import SessionStore

        new_session = SessionStore(sid=sid)
        msgs = new_session.read_messages()
        if not msgs:
            self._append_raw(f"\n会话 {sid} 无消息\n")
            return
        self.loop.session = new_session
        self.loop.messages = msgs
        name = new_session.get_name()
        self._append_raw(f"\n[resume] {sid} {name}，恢复 {len(msgs)} 条消息\n")
        self._invalidate()

    # ---- 回滚（/undo /rewind）----
    def _undo(self) -> None:
        """回滚最近一次 Write/Edit（三线：文件 + todo + messages）。"""
        msg = self.loop.session.undo_last_write()
        self._sync_after_rollback()
        self._append_raw(f"\n{msg}\n", flush=True)

    async def _rewind(self, cmd: str) -> None:
        """回退到某条用户消息处理完成后的状态（用户消息粒度）。

        TUI 无 prompt_toolkit 二次输入，交互式选择改由 /rewind <n> 带编号完成；
        不带参数只列候选列表。
        """
        parts = cmd.split(maxsplit=1)
        targets = self.loop.session.list_rewind_targets()
        if not targets:
            self._append_raw("\n(没有可回退的用户消息)\n", flush=True)
            return
        if len(parts) >= 2 and parts[1].strip().isdigit():
            n = int(parts[1].strip())
            if not (1 <= n <= len(targets)):
                self._append_raw(f"\n(编号需在 1~{len(targets)} 之间)\n", flush=True)
                return
            self._do_rewind(targets[n - 1], n)
            return
        lines = ["可回退到的用户消息（用 /rewind <n> 选择）："]
        for i, t in enumerate(targets, 1):
            lines.append(f"  {i}. {t['preview']}")
        self._append_raw("\n" + "\n".join(lines) + "\n", flush=True)

    def _do_rewind(self, target: dict, n: int) -> None:
        msg = self.loop.session.rewind_to_user(target["idx"])
        self._sync_after_rollback()
        self._append_raw(f"\n[rewind] 回到第 {n} 条消息：{msg}\n", flush=True)

    def _sync_after_rollback(self) -> None:
        """回滚后同步内存态：loop.messages 重读、todo_store 重载。"""
        self.loop.messages = self.loop.session.read_messages()
        if self.loop.todo_store is not None:
            self.loop.todo_store.load()
        self._invalidate()

    # ---- 工具方法 ----
    @staticmethod
    def _disp_width(s: str) -> int:
        try:
            from wcwidth import wcswidth

            w = wcswidth(s)
            return w if w >= 0 else len(s)
        except ImportError:
            return len(s)

    def _pad(self, s: str, width: int) -> str:
        dw = self._disp_width(s)
        if dw >= width:
            return s
        return s + " " * (width - dw)

    def _term_width(self) -> int:
        if self._app:
            return max(self._min_width, self._app.output.get_size().columns)
        return self._min_width

    def _session_name(self) -> str:
        return self.loop.session.get_name()

    def _input_box_ft(self):
        width = self._term_width()
        name = self._session_name()
        name_part = f" {name} " if name else ""
        remain = width - self._disp_width(name_part) - 2
        left = remain // 2
        right = remain - left
        return "─" * left + name_part + "─" * right

    def _bot_line_ft(self):
        """输入框下横线，与上横线对称。"""
        return "─" * self._term_width()

    def _ctx_info(self):
        """带缓存的 token 估算：消息条数不变就用缓存，避免每帧重算。"""
        from ..utils import estimate_messages_tokens

        n = len(self.loop.messages)
        if n != self._ctx_n:
            self._ctx_n = n
            win = getattr(self.loop.provider, "context_window", 32768)
            self._ctx_used = estimate_messages_tokens(
                self.loop.messages, self.loop._active_tools()
            )
            self._ctx_win = win
        return self._ctx_used, self._ctx_win

    def _fmt_elapsed(self) -> str:
        secs = int(time.time() - self._start)
        if secs >= 3600:
            return f"{secs // 3600}h {(secs % 3600) // 60}m"
        if secs >= 60:
            return f"{secs // 60}m {secs % 60}s"
        return f"{secs}s"

    def _status_ft(self):
        """底部状态栏 2 行：[模型] │ 目录 │ ⏱️ 时长 / Context 进度条 + 右侧模式标记。"""
        from pathlib import Path

        model = getattr(self.loop.provider, "model", "?")
        cwd = Path.cwd().name
        used, win = self._ctx_info()
        pct = int(100 * used / max(win, 1))
        filled = max(0, min(10, round(used / max(win, 1) * 10)))
        bar = "█" * filled + "░" * (10 - filled)
        bar_style = "class:status-warn" if pct >= 80 else "class:status-model"
        m = self.loop.mode
        if m == Mode.MANUAL:
            sym, label, mode_cls = "⏸", "manual", "class:mode-manual"
        elif m == Mode.PLAN:
            sym, label, mode_cls = "⏸", "plan", "class:mode-plan"
        else:
            sym, label, mode_cls = "⏵⏵", "auto", "class:mode-auto"
        mode_text = f"{sym} {label} mode on"
        return [
            ("class:status-dim", "  ["),
            ("class:status-model", model),
            ("class:status-dim", "] │ "),
            ("class:status-dim", cwd),
            ("class:status-dim", " │ ⏱️  "),
            ("class:status-dim", self._fmt_elapsed()),
            ("", "\n"),
            ("class:status-dim", "  Context "),
            (bar_style, bar),
            ("class:status-dim", f" {pct}%  "),
            (mode_cls, mode_text),
        ]

    def _cycle_mode(self) -> None:
        order = [Mode.MANUAL, Mode.PLAN, Mode.AUTO]
        i = order.index(self.loop.mode)
        self.loop.mode = order[(i + 1) % len(order)]
        self._invalidate()

    def _fmt_tool_args(self, arguments: dict) -> str:
        """工具参数摘要：命令类显示 command/cmd，其他 JSON 一行，截断 80。"""
        if not arguments:
            return ""
        for key in ("command", "cmd", "script"):
            v = arguments.get(key)
            if v:
                first = str(v).splitlines()[0] if isinstance(v, str) else str(v)
                return self._truncate(first, 80)
        import json
        return self._truncate(json.dumps(arguments, ensure_ascii=False), 80)

    def _fmt_tool_result(self, result: str) -> str:
        """工具结果摘要：首行截断 100 + 行数。"""
        if not result:
            return "(empty)"
        lines = str(result).splitlines()
        first = lines[0] if lines else ""
        summary = self._truncate(first, 100)
        return f"{summary} ({len(lines)} lines)" if len(lines) > 1 else summary

    @staticmethod
    def _truncate(s: str, n: int) -> str:
        s = s.replace("\n", " ").strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    def _at_bottom(self) -> bool:
        """pane 是否在底部，用于自动跟随判断。"""
        pane = getattr(self, "_history_pane", None)
        if pane is None:
            return True  # 首次渲染前默认跟随
        ms = getattr(pane, "_max_scroll", None)
        if ms is None:
            return True  # 首次渲染前默认跟随
        return pane.vertical_scroll >= ms

    def _normalize_trailing(self, n: int) -> None:
        """规范 buffer 末尾恰好 n 个换行符（用于工具块前空行控制）。"""
        buf = self.history_buffer
        text = buf.text
        stripped = text.rstrip("\n")
        if len(text) - len(stripped) == n:
            return
        at_bottom = self._at_bottom()
        buf.text = stripped + "\n" * n
        if at_bottom:
            buf.cursor_position = len(buf.text)

    def _append_raw(self, text: str, flush: bool = False) -> None:
        """追加到 history buffer。在底部时自动跟随到末尾（新消息自动翻页）。

        cursor 不再驱动滚动（ScrollablePane keep_cursor_visible=False），
        自动跟随靠设 pane.vertical_scroll 到大值，渲染时 _HistoryPane.write_to_screen clamp 到底。
        """
        buf = self.history_buffer
        at_bottom = self._at_bottom()
        buf.text = buf.text + text
        if at_bottom:
            pane = getattr(self, "_history_pane", None)
            if pane is not None:
                pane.vertical_scroll = 10**9  # 渲染 clamp 到底
        if flush:
            self._invalidate()

    def _invalidate(self) -> None:
        if self._app:
            self._app.invalidate()

    async def _spinner_loop(self) -> None:
        while self._app and self._app.is_running:
            if self.thinking:
                self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER)
                self._invalidate()
            await asyncio.sleep(0.08)

    # ---- REPL 接口 ----
    async def ask_permission(self, tool_call) -> str:
        self.thinking = False
        self._append_raw("  ⚠ 确认执行? (y/n/always)\n", flush=True)
        fut = asyncio.get_running_loop().create_future()
        self._pending_ask = (tool_call, fut)
        try:
            return await fut
        finally:
            self._pending_ask = None

    async def approve_plan(self, plan: str) -> bool:
        self._append_raw(f"\n[plan 待审批]\n{plan[:1000]}\n输入 a 批准 / r 拒绝\n", flush=True)
        fut = asyncio.get_running_loop().create_future()
        self._pending_plan_approval = fut
        try:
            return await fut
        finally:
            self._pending_plan_approval = None

    # ---- 运行 ----
    async def run(self) -> None:
        from prompt_toolkit import Application
        from .logo import print_logo

        await print_logo(self.version)
        self.loop.repl = self
        self._app = Application(
            layout=self._build_layout(),
            style=self._styles(),
            key_bindings=self._keybindings(),
            full_screen=True,
            mouse_support=True,
        )
        asyncio.create_task(self._spinner_loop())
        self._app.layout.focus(self._input_buffer_with_completer())
        await self._app.run_async()

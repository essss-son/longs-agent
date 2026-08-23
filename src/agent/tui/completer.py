"""Slash 命令补全（/ 前缀匹配 + 二级参数）。"""
from __future__ import annotations

from prompt_toolkit.completion import Completer, Completion

COMMANDS = [
    ("/exit", "退出"),
    ("/help", "帮助"),
    ("/plan", "进入计划模式"),
    ("/mode", "切换 NORMAL/AUTO"),
    ("/model", "切换模型"),
    ("/resume", "恢复历史会话"),
    ("/compact", "压缩上下文"),
    ("/context", "查看 token 用量"),
    ("/cost", "查看累计花费"),
    ("/trace", "查看追踪时间线"),
]


class SlashCompleter(Completer):
    def __init__(self, list_sessions=None, list_models=None):
        self.list_sessions = list_sessions or (lambda: [])
        self.list_models = list_models or (lambda: [])

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        parts = text.split(maxsplit=1)
        cmd = parts[0]
        # 一级：命令名补全（未输空格时）
        if len(parts) == 1:
            for name, desc in COMMANDS:
                if name.startswith(cmd):
                    yield Completion(name, start_position=-len(cmd), display_meta=desc)
            return
        # 二级：/resume <sid> /model <alias>
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == "/resume":
            for sid in self.list_sessions()[:8]:
                if sid.startswith(arg):
                    yield Completion(sid, start_position=-len(arg))
        elif cmd == "/model":
            for alias in self.list_models():
                if alias.startswith(arg):
                    yield Completion(alias, start_position=-len(arg))

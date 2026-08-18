"""CLI bootstrap：装配 Config/Provider/Registry/Loop/REPL + 权限 + todo + plan + compaction。

D9：装配 Compactor（provider + context_window）。
"""
from __future__ import annotations

from datetime import datetime

from .builtin_tools import Bash, Edit, Glob, Grep, Read, Write
from .compaction import Compactor
from .config import Config
from .loop import AgentLoop
from .messages import NormalizedResponse, ToolCall
from .permissions import Mode, PermissionConfig, PermissionEngine
from .plan_mode import ExitPlanMode
from .provider import FakeProvider, OpenAICompatibleProvider
from .repl import REPL
from .session import SessionStore
from .todo import TodoStore, TodoWrite
from .tools import ToolRegistry


def _build_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(Read())
    r.register(Write())
    r.register(Edit())
    r.register(Bash())
    r.register(Glob())
    r.register(Grep())
    r.register(ExitPlanMode())
    return r


def _demo_script() -> list:
    return [
        NormalizedResponse(
            tool_calls=[
                ToolCall(id="demo1", name="Read", arguments={"file_path": "longs-agent项目说明.md"})
            ]
        ),
        NormalizedResponse(
            content="（demo）我读取了项目说明。配置 .agent/config.toml 后可用真实模型自由对话。"
        ),
    ]


async def main() -> None:
    cfg = Config.load()
    model = cfg.get()
    api_key = cfg.api_key()

    if model and api_key:
        provider = OpenAICompatibleProvider(
            base_url=model.base_url,
            api_key=api_key,
            model=model.model,
            context_window=model.context_window,
        )
        print(f"[longs-agent] 模型: {model.model} @ {model.base_url}")
    else:
        provider = FakeProvider(_demo_script())
        print("[longs-agent] 未配置 .agent/config.toml 或缺少 api_key，用 demo 模式。")

    session = SessionStore()
    session.write_meta(
        {
            "sid": session.sid,
            "model_alias": model.alias if model else "demo",
            "created_at": datetime.now().isoformat(),
            "mode": "NORMAL",
        }
    )
    registry = _build_registry()
    todo_store = TodoStore(path=session.todo_path)
    registry.register(TodoWrite(todo_store))
    context_window = model.context_window if model else 32768
    compactor = Compactor(provider, context_window=context_window)
    from .memory import build_system_prompt, load_agent_md
    from .skills import scan_skills, skills_prompt_block

    system_prompt = build_system_prompt(
        agent_md=load_agent_md("."),
        skills_block=skills_prompt_block(scan_skills()),
        registry=registry,
        todos=None,  # todos 由 loop._current_system_prompt 每轮动态刷新
    )
    # todo_store 给 loop，让 run_turn 每轮刷新 system_prompt 的 todos 段
    loop = AgentLoop(
        provider,
        registry,
        session,
        permissions=PermissionEngine(),
        permission_config=PermissionConfig(),
        mode=Mode.NORMAL,
        todo_store=todo_store,
        compactor=compactor,
        system_prompt=system_prompt,
    )
    repl = REPL(loop)
    loop.repl = repl
    # 优先全屏 TUI（有 rich/pyfiglet 依赖时），否则回退基础 REPL
    try:
        from .tui.app import TUIApp

        tui = TUIApp(loop, version="0.1.0")
        loop.repl = tui
        await tui.run()
    except ImportError:
        await repl.run()


def trace_cli(args: list[str]) -> None:
    """agent trace view <id> [--failed] | export <id> -o <path>"""
    from pathlib import Path

    from .session import SessionStore
    from .trace import TraceStore

    if not args:
        print("usage: agent trace view <id> [--failed] | export <id> -o <path>")
        return
    sub = args[0]
    if sub == "view":
        if len(args) < 2:
            for s in SessionStore.list_sessions()[:10]:
                print(s)
            return
        sid = args[1]
        ts = TraceStore(SessionStore(sid=sid).trace_path)
        print(ts.failed_view() if "--failed" in args else ts.timeline_view())
    elif sub == "export":
        if len(args) < 2:
            print("usage: agent trace export <id> -o <path>")
            return
        sid = args[1]
        out = "trace.md"
        if "-o" in args:
            out = args[args.index("-o") + 1]
        md = TraceStore(SessionStore(sid=sid).trace_path).export_md()
        Path(out).write_text(md, encoding="utf-8")
        print(f"exported to {out}")
    else:
        print(f"unknown subcommand: {sub}")

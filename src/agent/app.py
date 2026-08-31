"""CLI bootstrap：装配 Config/Provider/Registry/Loop/REPL + 权限 + todo + plan + compaction。

D9：装配 Compactor（provider + context_window）。
"""
from __future__ import annotations

import os
from contextlib import AsyncExitStack
from datetime import datetime

from .archive import ArchiveStore, MemoryRead
from .builtin_tools import Bash, Edit, Glob, Grep, Read, Write
from .compaction import Compactor
from .config import Config
from .loop import AgentLoop
from .messages import NormalizedResponse, ToolCall
from .permissions import Mode, PermissionConfig, PermissionEngine
from .plan_mode import EnterPlanMode, ExitPlanMode
from .provider import AnthropicProvider, FakeProvider, OpenAICompatibleProvider
from .repl import REPL
from .session import SessionStore
from .task import Task
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
    r.register(EnterPlanMode())
    r.register(ExitPlanMode())
    return r


def _build_provider(model, api_key):
    """按 model.provider 构造 Provider（默认 openai_compatible；anthropic 走 AnthropicProvider）。"""
    if model.provider == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=model.model,
            context_window=model.context_window,
            max_tokens=model.max_tokens,
        )
    return OpenAICompatibleProvider(
        base_url=model.base_url,
        api_key=api_key,
        model=model.model,
        context_window=model.context_window,
    )


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


async def _load_mcp_tools(cfg, registry, exit_stack) -> None:
    """加载 config 里的 MCP servers，注册工具到 registry。失败不中断主流程。

    exit_stack 保持 MCP 连接常驻（agent 退出时统一干净关闭）。
    """
    for name, mcp_cfg in cfg.mcp_servers.items():
        try:
            from .mcp_client import load_mcp_server

            result = await load_mcp_server(
                name, mcp_cfg.command, mcp_cfg.args, url=mcp_cfg.url, exit_stack=exit_stack
            )
            for t in result.tools:
                registry.register(t)
            print(f"[longs-agent] MCP server '{name}' 加载 {len(result.tools)} 个工具")
        except ImportError:
            print(f"[longs-agent] MCP 未安装，跳过 server '{name}'（pip install -e \".[mcp]\"）")
        except Exception as e:
            print(f"[longs-agent] MCP server '{name}' 加载失败: {type(e).__name__}: {e}")


async def main() -> None:
    cfg = Config.load()
    model = cfg.get()
    api_key = cfg.api_key()

    if model and api_key:
        provider = _build_provider(model, api_key)
        print(f"[longs-agent] 模型: {model.model} @ {model.base_url}")
    else:
        provider = FakeProvider(_demo_script())
        print("[longs-agent] 未配置 .agent/config.toml 或缺少 api_key，用 demo 模式。")

    # light 模型（Task 子代理路由）：config 配了 "light" 别名才启用，否则子代理退化用主模型
    light_provider = None
    light_cfg = cfg.get("light")
    light_key = cfg.api_key("light")
    if light_cfg and light_key and light_cfg is not model:
        light_provider = _build_provider(light_cfg, light_key)

    session = SessionStore()
    archive = ArchiveStore(session.dir)  # L2 档案层：压缩换出的内容归档，MemoryRead 可取回
    session.write_meta(
        {
            "sid": session.sid,
            "model_alias": model.alias if model else "demo",
            "created_at": datetime.now().isoformat(),
            "mode": "MANUAL",
        }
    )
    registry = _build_registry()
    # MCP 连接常驻：exit_stack 保持 stdio/session 存活到 agent 退出，统一干净关闭
    async with AsyncExitStack() as exit_stack:
        # 加载 MCP 工具（注册进同一 registry → 自动复用权限/trace/plan）
        await _load_mcp_tools(cfg, registry, exit_stack)
        todo_store = TodoStore(path=session.todo_path)
        registry.register(TodoWrite(todo_store))
        registry.register(MemoryRead(archive))
        registry.register(Task(provider, light_provider=light_provider))
        context_window = model.context_window if model else 32768
        compactor = Compactor(provider, context_window=context_window, archive=archive)
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
            permission_config=PermissionConfig(project_root=os.getcwd()),
            mode=Mode.MANUAL,
            todo_store=todo_store,
            compactor=compactor,
            archive=archive,
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

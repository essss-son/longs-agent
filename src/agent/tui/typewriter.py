"""动画 3：打字机效果（B1：真实流式 chunk，不人为节流）。

用 Rich Live 异步刷新：on_delta 回调逐 chunk 追加文本 + Live.update，
不阻塞事件循环。独立运行：python -m agent.tui.typewriter
（模拟流式 chunk，验证打字机 UX；真实接入在 app.py 把 on_delta 指过来）
"""
from __future__ import annotations

import asyncio


async def _fake_stream(text: str, on_delta):
    """模拟流式 API：把文本切成不等长 chunk 逐个吐（模拟真实网络分块）。"""
    import random

    pos = 0
    chunks = []
    while pos < len(text):
        # 真实流式 chunk 长度不固定（2-8 字符），模拟网络分块
        size = min(random.randint(2, 8), len(text) - pos)
        chunks.append(text[pos : pos + size])
        pos += size
    for c in chunks:
        on_delta(c)
        await asyncio.sleep(0.04)  # 模拟 chunk 间隔


async def typewriter_print(text: str, title: str = "longs-agent") -> None:
    """打字机式渲染 AI 回复。on_delta 追加 + Rich Live 刷新。"""
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    rendered = Text()

    def on_delta(chunk: str) -> None:
        rendered.append(chunk, style="white")

    # 青紫色：cyan(0,255,255) 与 magenta(255,0,255) 中间，取 (128,128,255)
    border_rgb = "rgb(128,128,255)"
    title_styled = f"[{border_rgb}]{title}[/]"
    with Live(Panel(rendered, title=title_styled, border_style=border_rgb), refresh_per_second=30) as live:
        await _fake_stream(text, on_delta)
        live.update(Panel(rendered, title=title_styled, border_style=border_rgb))


if __name__ == "__main__":
    demo = "longs-agent 是一个 Claude Code 风格的 async code agent CLI。\n归一化消息 + append-only JSONL + 模式状态机，三原语支撑全部功能。"
    asyncio.run(typewriter_print(demo))

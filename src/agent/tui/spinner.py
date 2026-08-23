"""动画 2：思考中 spinner。

Rich Spinner("dots") 帧序列 ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏，配 "Thinking..." 文本，青色。
Live 异步刷新不阻塞。独立运行：python -m agent.tui.spinner
"""
from __future__ import annotations

import asyncio


async def show_spinner(duration: float = 3.0, text: str = "Thinking...") -> None:
    """显示 spinner 指定时长。用 Rich Live 异步刷新。"""
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    spinner = Spinner(
        "dots",
        text=Text(f" {text}", style="cyan"),
        style="cyan",
    )
    with Live(spinner, refresh_per_second=12, transient=True) as live:
        await asyncio.sleep(duration)


if __name__ == "__main__":
    asyncio.run(show_spinner())

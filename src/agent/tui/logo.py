"""动画 1：启动 logo 渐变打印。

"longs" figlet 逐行打印，颜色从紫色渐变到蓝色，每行 50ms。
独立运行：python -m agent.tui.logo
"""
from __future__ import annotations

import asyncio


def _gradient_color(i: int, n: int) -> tuple[int, int, int]:
    """第 i 行（共 n 行）从紫 (155,0,255) 渐变到蓝 (0,100,255)。"""
    if n <= 1:
        return (155, 0, 255)
    t = i / (n - 1)
    r = int(155 + (0 - 155) * t)
    g = int(0 + (100 - 0) * t)
    b = 255
    return (r, g, b)


def _render_logo_art(font: str = "slant") -> list[str]:
    import pyfiglet

    art = pyfiglet.figlet_format("longs-agent", font=font)
    return art.rstrip("\n").splitlines()


async def print_logo(version: str = "0.1.0", font: str = "slant", interval: float = 0.05) -> None:
    """逐行渐变打印 logo + 版本号。"""
    from rich.console import Console

    console = Console()
    lines = _render_logo_art(font)
    n = len(lines)
    for i, line in enumerate(lines):
        r, g, b = _gradient_color(i, n)
        console.print(f"[rgb({r},{g},{b})]{line}[/]")
        await asyncio.sleep(interval)
    # 版本号居中（按 logo 宽度对齐）
    width = max(len(l) for l in lines) if lines else 20
    console.print(f"[dim]{'longs-agent v' + version:^{width}}[/]")


if __name__ == "__main__":
    asyncio.run(print_logo())

"""工具返回标准化协议信封：超阈值 → 截断 + 落盘 + 返回指针。

所有工具输出统一走同一套截断规则，完整输出落盘保存，随时可回溯。
截断方向三种：head（默认，前 100 行）/ tail（后 100 行）/ head_tail（头尾各 50 行）。
阈值：100 行 / 12800 字节。
落盘：默认 tool-output/ 目录，调用方可传 output_dir（loop 传 session 目录）；文件名
tool_<timestamp>_<toolname>.json，内容为完整原始输出，返回指针为绝对路径。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

MAX_LINES = 100
MAX_BYTES = 12800
HEAD_TAIL_MARGIN = 50
OUTPUT_DIR = "tool-output"
VALID_DIRECTIONS = ("head", "tail", "head_tail")


def _truncate_lines(content: str, direction: str) -> str:
    """按方向截断为预览文本。"""
    lines = content.split("\n")
    if direction == "tail":
        return "\n".join(lines[-MAX_LINES:])
    if direction == "head_tail":
        head = lines[:HEAD_TAIL_MARGIN]
        tail = lines[-HEAD_TAIL_MARGIN:]
        return "\n".join(head + ["... [中间省略] ..."] + tail)
    return "\n".join(lines[:MAX_LINES])


def _dump_path(tool_name: str, output_dir: str) -> Path:
    """唯一落盘路径：同秒同名工具加序号避免覆盖。返回绝对路径（指针不依赖 CWD）。"""
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"tool_{ts}_{tool_name}.json"
    i = 0
    while path.exists():
        i += 1
        path = Path(output_dir) / f"tool_{ts}_{tool_name}_{i}.json"
    return path


def wrap(
    content: str,
    *,
    tool_name: str,
    direction: str = "head",
    output_dir: str = OUTPUT_DIR,
    archive=None,
    dump: bool = True,
) -> str:
    """超阈值则截断；dump=True 时落盘完整输出 + 登记 archive + 返回指针。

    dump=False（子代理）：只截断，不落盘、不归档、无指针——子代理运行无痕，
    结束后其最终报告作为 Task 工具结果回主循环，由主循环信封正常处理。
    """
    n_lines = content.count("\n") + 1
    n_bytes = len(content.encode("utf-8"))
    if n_lines <= MAX_LINES and n_bytes <= MAX_BYTES:
        return content
    if direction not in VALID_DIRECTIONS:
        direction = "head"
    preview = _truncate_lines(content, direction)
    kept_lines = preview.count("\n") + 1
    if not dump:
        return (
            f"⚠️ 输出过大已截断（子代理上下文内截断，未落盘） | direction={direction} "
            f"| original_lines={n_lines} | original_bytes={n_bytes} | kept_lines={kept_lines}\n{preview}"
        )
    path = _dump_path(tool_name, output_dir)
    path.write_text(
        json.dumps({"tool": tool_name, "content": content}, ensure_ascii=False),
        encoding="utf-8",
    )
    mem_id = ""
    if archive is not None:
        mem_id = archive.archive(
            "", kind="tool", tool_name=tool_name, file_path=str(path), char_count=len(content)
        )
    mem = f"mem_id={mem_id} | " if mem_id else ""
    return (
        f"⚠️ 输出过大已截断 | {mem}direction={direction} | original_lines={n_lines} "
        f"| original_bytes={n_bytes} | kept_lines={kept_lines} "
        f"| full_output_path={path}\n{preview}"
    )

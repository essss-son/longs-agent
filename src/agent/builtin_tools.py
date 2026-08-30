"""内置工具：Read / Write / Edit（D2）。

D3 补 Bash / Glob / Grep。
Edit 用精确字符串替换 + 失败回喂（不让模型生成 diff）。
文件 IO / 目录遍历等阻塞操作统一经 asyncio.to_thread 丢线程池，避免阻塞事件循环；
Bash / Grep(rg) 走子进程，本身真异步。
"""
from __future__ import annotations

import asyncio

from .tools import Tool


def _read_file(file_path: str, offset: int, limit: int | None) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines(keepends=True)
    start = max(0, offset - 1)  # 1-based → 0-based 索引
    end = len(lines) if limit is None else min(len(lines), start + max(0, limit))
    return "".join(f"{start + 1 + i}\t{line}" for i, line in enumerate(lines[start:end]))


class Read(Tool):
    name = "Read"
    description = (
        "读取文件内容，返回带行号的文本（行号 + tab + 内容）。file_path 为绝对或相对路径。"
        "offset/limit 支持分段读（1-based 行号）。"
    )
    schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "要读取的文件路径"},
            "offset": {
                "type": "integer",
                "default": 1,
                "description": "起始行号（1-based，默认 1）。从该行开始读。",
            },
            "limit": {
                "type": "integer",
                "description": "最多读取的行数（默认读到底）。",
            },
            "truncation": {
                "type": "string",
                "enum": ["head", "tail", "head_tail"],
                "default": "head",
                "description": "输出过大时截断方向：head 前100行 / tail 后100行 / head_tail 头尾各50行",
            },
        },
        "required": ["file_path"],
        "additionalProperties": False,
    }
    read_only = True

    async def execute(
        self, file_path: str, offset: int = 1, limit: int | None = None, **_: object
    ) -> str:
        return await asyncio.to_thread(_read_file, file_path, offset, limit)


def _write_file(file_path: str, content: str) -> str:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {file_path}"


class Write(Tool):
    name = "Write"
    description = "写入文件（整体覆盖）。file_path + content。"
    schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "要写入的文件路径"},
            "content": {"type": "string", "description": "完整文件内容"},
        },
        "required": ["file_path", "content"],
        "additionalProperties": False,
    }
    read_only = False

    async def execute(self, file_path: str, content: str, **_: object) -> str:
        return await asyncio.to_thread(_write_file, file_path, content)


def _edit_file(
    file_path: str, old_string: str, new_string: str, replace_all: bool
) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    count = content.count(old_string)
    if count == 0:
        return f"[error: old_string not found in {file_path}]"
    if count > 1 and not replace_all:
        return (
            f"[error: old_string appears {count} times in {file_path}, "
            "not unique; set replace_all=true]"
        )
    new = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new)
    return f"replaced {count if replace_all else 1} occurrence(s) in {file_path}"


class Edit(Tool):
    name = "Edit"
    description = (
        "精确字符串替换。file_path + old_string + new_string。"
        "old_string 必须在文件中唯一，否则需 replace_all=true。"
        "匹配失败返回错误文本（回喂供模型自纠正），不生成 diff。"
    )
    schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "要编辑的文件路径"},
            "old_string": {"type": "string", "description": "要替换的精确字符串"},
            "new_string": {"type": "string", "description": "替换后的字符串"},
            "replace_all": {
                "type": "boolean",
                "default": False,
                "description": "是否替换所有出现",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
        "additionalProperties": False,
    }
    read_only = False

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **_: object,
    ) -> str:
        return await asyncio.to_thread(
            _edit_file, file_path, old_string, new_string, replace_all
        )


class Bash(Tool):
    name = "Bash"
    description = "执行 bash 命令，返回合并的 stdout+stderr（截断到 10k 字符）。timeout 秒。"
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "bash 命令"},
            "timeout": {"type": "integer", "default": 120, "description": "超时秒数"},
            "truncation": {
                "type": "string",
                "enum": ["head", "tail", "head_tail"],
                "default": "head",
                "description": "输出过大时截断方向：head 前100行 / tail 后100行 / head_tail 头尾各50行",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    read_only = False

    async def execute(self, command: str, timeout: int = 120, **_: object) -> str:
        from .utils import truncate

        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"[error: command timed out after {timeout}s]"
        # 兜底硬截 200k（防极端内存爆）；常规超长输出交给统一信封落盘
        return truncate(out.decode("utf-8", errors="replace"), 200000)


def _glob(pattern: str, path: str) -> str:
    from pathlib import Path

    root = Path(path)
    results: list[str] = []
    for p in root.glob(pattern):
        if ".git" in p.parts:
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        results.append(str(rel))
    results.sort()
    if not results:
        return "(no matches)"
    shown = results[:200]
    out = "\n".join(shown)
    if len(results) > 200:
        out += f"\n... [showing 200 of {len(results)}]"
    return out


class Glob(Tool):
    name = "Glob"
    description = "递归匹配文件路径（glob 模式，如 '**/*.py'）。默认忽略 .git。"
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "glob 模式"},
            "path": {"type": "string", "default": ".", "description": "根目录"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    read_only = True

    async def execute(self, pattern: str, path: str = ".", **_: object) -> str:
        return await asyncio.to_thread(_glob, pattern, path)


async def _grep_rg(pattern: str, path: str) -> list[str]:
    proc = await asyncio.create_subprocess_exec(
        "rg",
        "-n",
        "--no-heading",
        pattern,
        path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    return out.decode("utf-8", errors="replace").splitlines()


def _grep_py(pattern: str, path: str, include: str) -> list[str]:
    import re
    from pathlib import Path

    regex = re.compile(pattern)
    root = Path(path)
    lines: list[str] = []
    for p in root.rglob(include):
        if ".git" in p.parts or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                lines.append(f"{p}:{i}:{line}")
    return lines


class Grep(Tool):
    name = "Grep"
    description = "正则搜索文件内容。pattern 为正则。优先用 rg，否则纯 Python。结果截断 50 行。"
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则表达式"},
            "path": {"type": "string", "default": ".", "description": "搜索根目录"},
            "include": {"type": "string", "default": "*", "description": "文件名 glob 过滤"},
            "truncation": {
                "type": "string",
                "enum": ["head", "tail", "head_tail"],
                "default": "head",
                "description": "输出过大时截断方向：head 前100行 / tail 后100行 / head_tail 头尾各50行",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    read_only = True

    async def execute(
        self, pattern: str, path: str = ".", include: str = "*", **_: object
    ) -> str:
        import shutil

        if shutil.which("rg"):
            lines = await _grep_rg(pattern, path)          # 真异步子进程
        else:
            lines = await asyncio.to_thread(_grep_py, pattern, path, include)
        if not lines:
            return "(no matches)"
        shown = lines[:50]
        out = "\n".join(shown)
        if len(lines) > 50:
            out += f"\n... [showing 50 of {len(lines)}]"
        return out

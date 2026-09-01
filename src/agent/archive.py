"""L2 档案层：压缩换出的工具原文落盘，模型按 mem_id 取回。

单策略压缩下，archive 只存「工具输出」：信封截断的大输出落盘 tool-output/ 文件、
记录只存 file_path 指针；中等/小输出内联全文。检索入口只有 MemoryRead（按 mem_id
一次取回原文），不再有全文检索——"该读哪条"由 context 里的「早期工具调用台账」
给出，模型自己挑 mem_id 读。

存储：session 目录下 archive.jsonl，append-only + 容错读（复用 session.read_jsonl_tolerant）。
记录：{mem_id, kind, tool_name, tool_call_id, seq, char_count, preview, content, file_path?}
- mem_id: t_XXXX，全局单调编号，resume 读档续号
- file_path: 信封大输出的原文文件；有则 MemoryRead 读文件，否则读内联 content
"""
from __future__ import annotations

import json
from pathlib import Path

from .session import read_jsonl_tolerant
from .tools import Tool


class ArchiveStore:
    """会话级档案：append-only JSONL + 内存索引。mem_id 顺序编号，恢复时读档续号。"""

    def __init__(self, dir: str | Path):
        self.path = Path(dir) / "archive.jsonl"
        # 已有档案（resume 场景）→ 内存索引 + mem_id 续号
        self._records: list[dict] = read_jsonl_tolerant(self.path)

    def archive(
        self,
        content: str,
        *,
        kind: str = "tool",
        tool_name: str = "",
        tool_call_id: str = "",
        file_path: str = "",
        char_count: int | None = None,
    ) -> str:
        """归档一段内容，返回 mem_id。file_path 指向信封大输出的原文文件（原文不内联）。"""
        mem_id = f"t_{len(self._records) + 1:04d}"
        record = {
            "mem_id": mem_id,
            "kind": kind,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "seq": len(self._records) + 1,
            "char_count": len(content) if char_count is None else char_count,
            "preview": " ".join(content.split())[:80],
            "content": content,
        }
        if file_path:
            record["file_path"] = file_path
        line = json.dumps(record, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
        self._records.append(record)
        return mem_id

    def read(self, mem_id: str) -> dict | None:
        for r in self._records:
            if r["mem_id"] == mem_id:
                return r
        return None

    def has_tool_call_id(self, tool_call_id: str) -> bool:
        """这条 tool 的原文是否已归档（kind=tool 且 tool_call_id 匹配），防重复归档。"""
        if not tool_call_id:
            return False
        return any(
            r["kind"] == "tool" and r["tool_call_id"] == tool_call_id
            for r in self._records
        )


class MemoryRead(Tool):
    """按 mem_id 取回工具原文。read_only（plan 模式可用）。"""

    name = "MemoryRead"
    description = (
        "按 mem_id 从会话档案取回被压缩换出的工具输出原文。"
        "mem_id 从 context 里的「早期工具调用台账」读取；需要找回早前某次工具调用的完整输出时使用。"
    )
    schema = {
        "type": "object",
        "properties": {
            "mem_id": {"type": "string", "description": "档案 ID，如 t_0007"},
            "max_chars": {
                "type": "integer",
                "default": 16000,
                "description": "返回字符上限（防大内容回灌爆窗）",
            },
        },
        "required": ["mem_id"],
        "additionalProperties": False,
    }
    read_only = True

    def __init__(self, store: ArchiveStore):
        self.store = store

    async def execute(self, mem_id: str, max_chars: int = 16000, **_: object) -> str:
        r = self.store.read(mem_id)
        if r is None:
            return f"[error: no archive entry '{mem_id}']"
        content = r.get("content") or ""
        fp = r.get("file_path")
        if fp and Path(fp).exists():
            raw = Path(fp).read_text(encoding="utf-8")
            try:  # 信封落盘是 JSON 包装（{"tool":..., "content":...}），解包取原文
                loaded = json.loads(raw)
                content = loaded.get("content", raw) if isinstance(loaded, dict) else raw
            except json.JSONDecodeError:
                content = raw  # 非 JSON 文件：原样返回
        char_count = r.get("char_count") or len(content)
        header = (
            f"[archive {r['mem_id']} | tool={r['tool_name'] or '-'} "
            f"| seq={r['seq']} | {char_count} chars | 历史快照]\n"
        )
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return header + content

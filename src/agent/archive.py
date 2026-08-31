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
        """这条 tool 的全文是否已归档（kind=elision 且 tool_call_id 匹配）。

        用于 summary/window 换出 unit 前补归档：避免对已有全文副本的 tool 重复归档。
        """
        if not tool_call_id:
            return False
        return any(
            r["kind"] == "elision" and r["tool_call_id"] == tool_call_id
            for r in self._records
        )

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        """BM25 检索 + recency 加权，返回 [(record, score)] top-k。"""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        docs = [(_tokenize(r["content"]), r) for r in self._records]
        n = len(docs)
        if n == 0:
            return []
        # df: 每个词出现在多少条档案
        df: dict[str, int] = {}
        for tokens, _ in docs:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        avgdl = sum(len(tokens) for tokens, _ in docs) / n  # 平均文档长度
        k1, b = 1.5, 0.75  # BM25 超参：词频饱和 + 长度归一化
        scored: list[tuple[dict, float]] = []
        for tokens, r in docs:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            doc_len = len(tokens)
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                # BM25 IDF：平滑、恒正，df 越大 idf 越小
                idf = math.log(1 + (n - df[qt] + 0.5) / (df[qt] + 0.5))
                # 词频饱和 + 文档长度归一化
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avgdl))
            if score > 0:
                score *= 1 + 0.1 * (r["seq"] / n)  # recency：新的略优先
                scored.append((r, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class MemoryRead(Tool):
    """按 mem_id 取回档案原文。read_only（plan 模式可用）。"""

    name = "MemoryRead"
    description = (
        "按 mem_id 从会话档案取回被压缩换出的原始内容。"
        "当历史消息中出现 [elided ... mem_id=t_xxxx] marker、"
        "或需要找回早前工具输出/消息的完整内容时使用。"
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
        header = (
            f"[archive {r['mem_id']} | kind={r['kind']} | tool={r['tool_name'] or '-'} "
            f"| seq={r['seq']} | {r['char_count']} chars | 历史快照]\n"
        )
        content = r["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return header + content


class MemorySearch(Tool):
    """关键词检索档案，返回 top-k 候选（mem_id + preview）。read_only（plan 可用）。"""

    name = "MemorySearch"
    description = (
        "在会话档案中检索被压缩换出的历史内容（早期工具输出/消息），返回候选 mem_id + 预览。"
        "需要找回早前出现过的路径/函数名/报错/数字、或用户提到「刚才/之前」时使用；"
        "拿到 mem_id 后用 MemoryRead 取回原文。"
    )
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词（路径/函数名/报错片段等）"},
            "k": {"type": "integer", "default": 5, "description": "返回候选数"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    read_only = True

    def __init__(self, store: ArchiveStore):
        self.store = store

    async def execute(self, query: str, k: int = 5, **_: object) -> str:
        hits = self.store.search(query, k=k)
        if not hits:
            return "(no matches)"
        lines = [
            f'{r["mem_id"]} | {r["kind"]}/{r["tool_name"] or "-"} | seq={r["seq"]} '
            f"| {r['char_count']} chars | {r['preview']}"
            for r, _ in hits
        ]
        return "\n".join(lines)

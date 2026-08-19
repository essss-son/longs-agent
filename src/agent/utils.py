"""共享工具：token 估算 + diff 渲染 + 截断。

token 估算必须把工具 schema 也算进去（最常被遗忘，导致 80% 阈值偏小、compaction 滞后）。
"""
from __future__ import annotations

import difflib
import json
from typing import Any


def truncate(s: str, limit: int = 10000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated, {len(s) - limit} more chars]"


def render_diff(old: str, new: str, n: int = 3) -> str:
    """unified diff，n 行上下文。用于 D4 ask UI 展示 Write/Edit 变更。"""
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        n=n,
        lineterm="",
    )
    return "".join(diff) or "(no changes)"


def estimate_tokens(text: str, model: str | None = None) -> int:
    """tiktoken 优先：encoding_for_model → cl100k_base → len/4 兜底。"""
    try:
        import tiktoken

        enc = None
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = None
        if enc is None:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_messages_tokens(
    messages: list,
    tools: list[dict] | None = None,
    model: str | None = None,
) -> int:
    """估算 messages + 工具 schema 的 token。

    工具 schema 本身消耗 token（6 工具约 2-4k），必须算进去。
    每条消息固定开销 ~4 tokens（角色分隔）。
    """
    total = 0
    for m in messages:
        total += 4  # 每条消息固定开销
        d = m.to_dict() if hasattr(m, "to_dict") else m
        total += estimate_tokens(json.dumps(d, ensure_ascii=False), model)
    if tools:
        total += estimate_tokens(json.dumps(tools, ensure_ascii=False), model)
    return total

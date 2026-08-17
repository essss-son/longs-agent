"""归一化消息类型 —— 整个 agent 的地基。

所有 Provider 的公共交集 + 必要超集。历史永远是 list[Message] 落 messages.jsonl，
与供应商无关。Provider 职责是双向翻译：归一化 → 各家 API 格式，响应 → 归一化。
这是 /model 热切换和跨供应商 resume 的前提（第一天必须定对，否则 D14 返工）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """assistant 发起的一次工具调用。

    arguments 存已解析的 dict（非 JSON 字符串），由 Provider 决定序列化时
    json.dumps 还是直传对象。反序列化时不必二次解析，避免 str/dict 歧义。
    """
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        args = d.get("arguments", {})
        # 容错：早期数据或 OpenAI 原生残留可能是 JSON 字符串，统一成 dict
        if isinstance(args, str):
            args = json.loads(args) if args else {}
        if not isinstance(args, dict):
            args = {}
        return cls(id=d["id"], name=d["name"], arguments=args)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Usage":
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )


@dataclass
class NormalizedResponse:
    """Provider.chat / stream 返回的归一化响应。

    raw 保留原始响应供调试，但不入 JSONL（只落 Message，不落 Response）。
    """
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    model: str | None = None
    raw: dict | None = None


@dataclass
class Message:
    """归一化消息。role ∈ {"system","user","assistant","tool"}。

    - assistant 可同时带 content 和 tool_calls（一次发起多个并行工具调用）
    - tool 消息带 tool_call_id 关联 assistant.tool_calls[i].id，name 为工具名
    - P0 content 一律 str（list[ContentBlock] 留给 P1 Anthropic 多 block）
    """
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        tool_calls = None
        if d.get("tool_calls"):
            tool_calls = [ToolCall.from_dict(tc) for tc in d["tool_calls"]]
        return cls(
            role=d["role"],
            content=d.get("content"),
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


def as_text(content: str | None | list) -> str:
    """把 content（str 或 list[ContentBlock]）归一成纯文本。

    P0 content 一律 str；list 分支留给 P1 Anthropic 多 block（取所有 text 块拼接）。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
        else:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)

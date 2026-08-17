"""Provider 抽象 + OpenAICompatibleProvider 双向翻译 + FakeProvider。

关键：内部消息历史永远是归一化 list[Message]。Provider 职责是双向翻译。
OpenAICompatibleProvider 一家覆盖 OpenAI/DeepSeek/Qwen/Ollama/本地 vLLM。

D1 只实现非流式 chat；流式 stream 留 D3（渲染 delta.content + 按 index
累积 tool_calls 片段，流结束重建 ToolCall）。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from .messages import Message, NormalizedResponse, ToolCall, Usage, as_text


class Provider(Protocol):
    @property
    def model(self) -> str: ...
    @property
    def context_window(self) -> int: ...

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system: str | None = None,
        **kw,
    ) -> NormalizedResponse: ...


def to_openai_messages(messages: list[Message], system: str | None = None) -> list[dict]:
    """归一化 list[Message] → OpenAI chat 格式。

    assistant 的 tool_calls[].function.arguments 序列化成 JSON 字符串（OpenAI 要求）；
    tool 消息带 tool_call_id 关联。
    """
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "system":
            out.append({"role": "system", "content": as_text(m.content)})
        elif m.role == "user":
            out.append({"role": "user", "content": as_text(m.content)})
        elif m.role == "assistant":
            d: dict[str, Any] = {"role": "assistant", "content": as_text(m.content) or None}
            if m.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(d)
        elif m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": as_text(m.content),
                    "name": m.name or "",
                }
            )
    return out


def from_openai_response(resp: Any) -> NormalizedResponse:
    """OpenAI 响应 → 归一化。

    resp 可为 openai SDK 响应对象（有 model_dump）或 dict（测试 fixture 用）。
    tool_calls 的 function.arguments（JSON 字符串）解析成 dict；解析失败保留原文。
    """
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    elif isinstance(resp, dict):
        data = resp
    else:
        raise TypeError(f"unsupported response type: {type(resp)}")

    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content")

    tool_calls = None
    if msg.get("tool_calls"):
        tool_calls = []
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                args = {"_raw": args_raw}  # 解析失败保留原文，回喂供模型自纠正
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

    usage_data = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    )

    return NormalizedResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=choice.get("finish_reason"),
        model=data.get("model"),
        raw=data,
    )


class OpenAICompatibleProvider:
    """覆盖 OpenAI / DeepSeek / Qwen / Ollama / 本地 vLLM。

    延迟导入 openai（在 __init__），避免无依赖时 import 报错（测试用 FakeProvider 不需 openai）。
    D1 只实现非流式 chat。
    """

    def __init__(self, base_url: str, api_key: str, model: str, context_window: int):
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._context_window = context_window
        from openai import AsyncOpenAI  # 延迟导入

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system: str | None = None,
        **kw,
    ) -> NormalizedResponse:
        openai_msgs = to_openai_messages(messages, system)
        kwargs: dict[str, Any] = {"model": self._model, "messages": openai_msgs}
        if tools:
            kwargs["tools"] = tools
        resp = await self._client.chat.completions.create(**kwargs)
        return from_openai_response(resp)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        **kw,
    ) -> NormalizedResponse:
        """D3 实现：流式渲染 delta.content + 按 index 累积 tool_calls 片段。"""
        raise NotImplementedError("stream 在 D3 实现")


class ScriptExhaustedError(Exception):
    """FakeProvider 剧本耗尽。"""


class FakeProvider:
    """剧本式假 LLM —— 确定性端到端测试基石。

    script 项可为 NormalizedResponse 或 Callable[[messages, tools], NormalizedResponse]
    （动态响应，用于测试 compaction 触发后行为）。
    脚本耗尽抛 ScriptExhaustedError（明确失败而非 NoneType 崩溃）。
    记录所有调用到 self.calls 供断言。
    """

    def __init__(self, script: list, *, model: str = "fake", context_window: int = 32768):
        self.script = list(script)
        self.calls: list[tuple[list[Message], list[dict] | None]] = []
        self._model = model
        self._context_window = context_window

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    async def chat(self, messages, tools=None, system=None, **kw) -> NormalizedResponse:
        self.calls.append((list(messages), tools))
        if not self.script:
            raise ScriptExhaustedError("FakeProvider 剧本耗尽")
        item = self.script.pop(0)
        return item(messages, tools) if callable(item) else item

    async def stream(self, messages, tools=None, system=None, on_delta=None, **kw) -> NormalizedResponse:
        resp = await self.chat(messages, tools, system)
        if on_delta and resp.content:
            on_delta(resp.content)  # 一次性吐，模拟流式
        return resp

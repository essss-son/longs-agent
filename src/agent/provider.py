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
        on_reasoning: Callable[[str], None] | None = None,
        **kw,
    ) -> NormalizedResponse:
        """流式：实时渲染 delta.content，按 index 累积 tool_calls 片段，流结束重建。

        隐藏坑：首 chunk 常只有 delta.role 无 content；末 chunk 才有 finish_reason；
        arguments 跨多 chunk 按 index 累积；tool_call.id 只在首个含该 index 的 chunk。
        on_reasoning：解析 delta.reasoning_content（DeepSeek-R1 / glm 思考模型等），
        模型不返回则不触发，兼容无害。
        """
        openai_msgs = to_openai_messages(messages, system)
        kwargs: dict[str, Any] = {"model": self._model, "messages": openai_msgs, "stream": True}
        if tools:
            kwargs["tools"] = tools
        # 要求流式末尾回传 usage（OpenAI 需显式开启；兼容 provider 忽略此参数）
        kwargs["stream_options"] = {"include_usage": True}
        stream = await self._client.chat.completions.create(**kwargs)

        content_parts: list[str] = []
        tool_frags: dict[int, dict] = {}  # index → {id, name, args_parts}
        finish_reason = None
        model = None
        usage_data: dict | None = None
        async for chunk in stream:
            data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            if not model:
                model = data.get("model")
            if data.get("usage"):
                usage_data = data["usage"]  # 流式 usage 只在最后一个 chunk 顶层
            choices = data.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
                if on_delta:
                    on_delta(delta["content"])
            rc = delta.get("reasoning_content")
            if rc and on_reasoning:
                on_reasoning(rc)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                frag = tool_frags.setdefault(idx, {"id": None, "name": None, "args": []})
                if tc.get("id"):
                    frag["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    frag["name"] = fn["name"]
                if fn.get("arguments"):
                    frag["args"].append(fn["arguments"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        content = "".join(content_parts) if content_parts else None
        tool_calls = None
        if tool_frags:
            tool_calls = []
            for idx in sorted(tool_frags):
                frag = tool_frags[idx]
                args_str = "".join(frag["args"])
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {"_raw": args_str}
                tool_calls.append(ToolCall(id=frag["id"] or "", name=frag["name"] or "", arguments=args))
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ) if usage_data else Usage()
        return NormalizedResponse(
            content=content, tool_calls=tool_calls, finish_reason=finish_reason,
            model=model, usage=usage,
        )


def _map_stop_reason(stop_reason: str | None) -> str | None:
    """Anthropic stop_reason → OpenAI 风格 finish_reason。"""
    if stop_reason == "end_turn":
        return "stop"
    if stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    return stop_reason


def to_anthropic_tools(tools: list[dict] | None) -> list[dict]:
    """OpenAI tool defs → Anthropic tools 格式。input_schema.type 兜底 "object"。"""
    out: list[dict] = []
    for t in tools or []:
        fn = t.get("function", {})
        schema = dict(fn.get("parameters") or {"type": "object"})
        schema.setdefault("type", "object")
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": schema,
            }
        )
    return out


def _merge_adjacent(out: list[dict]) -> list[dict]:
    """合并连续同角色消息（Anthropic 强制 user/assistant 交替）。tool 已在 to_ 阶段合并。"""
    merged: list[dict] = []
    for msg in out:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            pc, mc = prev["content"], msg["content"]
            if isinstance(pc, list) and isinstance(mc, list):
                prev["content"] = pc + mc
            elif isinstance(pc, list):
                prev["content"] = pc + [{"type": "text", "text": mc}]
            elif isinstance(mc, list):
                prev["content"] = [{"type": "text", "text": pc}] + mc
            else:
                prev["content"] = pc + "\n" + mc
        else:
            merged.append(msg)
    return merged


def to_anthropic_messages(
    messages: list[Message], system: str | None = None
) -> tuple[str | None, list[dict]]:
    """归一化 list[Message] → Anthropic 格式 (system, messages)。

    - tool 消息（可能连续多条）→ 合并成一条 user 的 tool_result blocks
    - assistant.tool_calls → content 里的 tool_use blocks（input 是 dict，不 json 序列化）
    - system role 消息 → 合并进 system 字段
    """
    out: list[dict] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.role == "system":
            system = (system + "\n" + (m.content or "")) if system else (m.content or "")
            i += 1
        elif m.role == "tool":
            blocks: list[dict] = []
            while i < n and messages[i].role == "tool":
                t = messages[i]
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": t.tool_call_id or "",
                        "content": t.content or "",
                    }
                )
                i += 1
            out.append({"role": "user", "content": blocks})
        elif m.role == "user":
            out.append({"role": "user", "content": m.content or ""})
            i += 1
        elif m.role == "assistant":
            blocks = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id or "",
                        "name": tc.name,
                        "input": tc.arguments or {},
                    }
                )
            out.append({"role": "assistant", "content": blocks if blocks else (m.content or "")})
            i += 1
        else:
            i += 1
    return system, _merge_adjacent(out)


def from_anthropic_response(resp: Any) -> NormalizedResponse:
    """Anthropic 响应 → 归一化。resp 可为 anthropic SDK 对象或 dict（fixture）。

    content 是 list[ContentBlock]：text block 拼 content，tool_use block 转 ToolCall
    （input 本来就是 dict）。usage 拆 input/output tokens。stop_reason 映射 finish_reason。
    """
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    elif isinstance(resp, dict):
        data = resp
    else:
        raise TypeError(f"unsupported response type: {type(resp)}")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content") or []:
        if isinstance(block, str):
            text_parts.append(block)
        elif block.get("type") == "text":
            text_parts.append(block.get("text") or "")
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input") or {},
                )
            )
    usage_data = data.get("usage") or {}
    in_tok = usage_data.get("input_tokens", 0)
    out_tok = usage_data.get("output_tokens", 0)
    return NormalizedResponse(
        content="".join(text_parts) or None,
        tool_calls=tool_calls or None,
        usage=Usage(in_tok, out_tok, in_tok + out_tok),
        finish_reason=_map_stop_reason(data.get("stop_reason")),
        model=data.get("model"),
        raw=data,
    )


class AnthropicProvider:
    """Anthropic Messages API provider。双向翻译 + chat/stream。

    延迟导入 anthropic（__init__），无依赖时不报错（测试用 fixture 不打真实 API）。
    max_tokens 必传（Anthropic 强制）。
    """

    def __init__(self, api_key: str, model: str, context_window: int, max_tokens: int = 8192):
        self._model = model
        self._context_window = context_window
        self._max_tokens = max_tokens
        from anthropic import AsyncAnthropic  # 延迟导入

        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    def _kwargs(self, messages, tools, system) -> dict[str, Any]:
        system_out, msgs = to_anthropic_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": msgs,
        }
        if system_out:
            kwargs["system"] = system_out
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        return kwargs

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system: str | None = None,
        **kw,
    ) -> NormalizedResponse:
        resp = await self._client.messages.create(**self._kwargs(messages, tools, system))
        return from_anthropic_response(resp)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        **kw,
    ) -> NormalizedResponse:
        """流式：按 index 累积 text / tool_use 片段。

        坑：input_tokens 在 message_start 事件，output_tokens 在 message_delta 事件；
        tool_use 的 input 是 partial_json 按 index 累积，流结束 json.loads 重建。
        """
        text_parts: list[str] = []
        tool_by_index: dict[int, dict] = {}
        finish_reason: str | None = None
        input_tokens = output_tokens = 0
        async with self._client.messages.stream(
            **self._kwargs(messages, tools, system)
        ) as stream:
            async for event in stream:
                if event.type == "message_start":
                    if event.message and event.message.usage:
                        input_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_start":
                    cb = event.content_block
                    if getattr(cb, "type", None) == "tool_use":
                        tool_by_index[event.index] = {
                            "id": cb.id,
                            "name": cb.name,
                            "input_parts": [],
                        }
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if getattr(delta, "type", None) == "text_delta" and delta.text:
                        text_parts.append(delta.text)
                        if on_delta:
                            on_delta(delta.text)
                    elif (
                        getattr(delta, "type", None) == "input_json_delta"
                        and delta.partial_json
                    ):
                        tool_by_index.setdefault(
                            event.index, {"id": None, "name": None, "input_parts": []}
                        )["input_parts"].append(delta.partial_json)
                    elif getattr(delta, "type", None) == "thinking_delta" and on_reasoning:
                        on_reasoning(delta.thinking)
                elif event.type == "message_delta":
                    if event.delta and event.delta.stop_reason:
                        finish_reason = event.delta.stop_reason
                    if event.usage:
                        output_tokens = event.usage.output_tokens

        content = "".join(text_parts) or None
        tool_calls: list[ToolCall] | None = None
        if tool_by_index:
            tool_calls = []
            for idx in sorted(tool_by_index):
                rec = tool_by_index[idx]
                args_str = "".join(rec["input_parts"])
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {"_raw": args_str}
                tool_calls.append(
                    ToolCall(id=rec["id"] or "", name=rec["name"] or "", arguments=args)
                )
        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            usage=Usage(input_tokens, output_tokens, input_tokens + output_tokens),
            finish_reason=_map_stop_reason(finish_reason),
            model=self._model,
        )


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

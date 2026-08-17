"""归一化消息类型往返测试。"""
from __future__ import annotations

import json

from agent.messages import Message, NormalizedResponse, ToolCall, Usage, as_text


def test_message_roundtrip_plain():
    m = Message("user", "hello")
    d = m.to_dict()
    assert d == {"role": "user", "content": "hello"}
    assert Message.from_dict(d) == m


def test_message_roundtrip_assistant_with_tool_calls():
    m = Message(
        "assistant",
        "thinking...",
        tool_calls=[ToolCall(id="tc1", name="Read", arguments={"file_path": "a.py"})],
    )
    d = m.to_dict()
    assert d["role"] == "assistant"
    assert d["content"] == "thinking..."
    assert d["tool_calls"] == [
        {"id": "tc1", "name": "Read", "arguments": {"file_path": "a.py"}}
    ]
    m2 = Message.from_dict(d)
    assert m2.role == "assistant"
    # 关键：arguments 是 dict，不是 JSON 字符串
    assert m2.tool_calls[0].arguments == {"file_path": "a.py"}
    assert isinstance(m2.tool_calls[0].arguments, dict)


def test_message_roundtrip_tool_message():
    m = Message("tool", "file content...", tool_call_id="tc1", name="Read")
    d = m.to_dict()
    assert d == {
        "role": "tool",
        "content": "file content...",
        "tool_call_id": "tc1",
        "name": "Read",
    }
    m2 = Message.from_dict(d)
    assert m2.tool_call_id == "tc1"
    assert m2.name == "Read"


def test_toolcall_from_dict_arguments_string_tolerant():
    """容错：arguments 可能是 JSON 字符串（OpenAI 原生残留），统一成 dict。"""
    tc = ToolCall.from_dict({"id": "x", "name": "F", "arguments": '{"a": 1}'})
    assert tc.arguments == {"a": 1}


def test_toolcall_from_dict_arguments_empty_tolerant():
    tc = ToolCall.from_dict({"id": "x", "name": "F", "arguments": ""})
    assert tc.arguments == {}


def test_as_text():
    assert as_text(None) == ""
    assert as_text("hi") == "hi"
    assert as_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert as_text([{"type": "other"}, {"text": "c"}]) == "c"


def test_four_roles_to_dict_json_serializable():
    """四角色消息都能 JSON 序列化（落 JSONL 前提）。"""
    msgs = [
        Message("system", "you are an agent"),
        Message("user", "hi"),
        Message("assistant", "ok", tool_calls=[ToolCall(id="1", name="X", arguments={})]),
        Message("tool", "result", tool_call_id="1", name="X"),
    ]
    for m in msgs:
        json.dumps(m.to_dict())


def test_normalized_response_defaults():
    nr = NormalizedResponse()
    assert nr.content is None
    assert nr.tool_calls is None
    assert nr.usage == Usage()
    assert nr.finish_reason is None

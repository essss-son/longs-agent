"""Anthropic Provider 双向翻译测试（合成 fixture，不打真实 API）。"""
from __future__ import annotations

from agent.messages import Message, ToolCall
from agent.provider import (
    from_anthropic_response,
    to_anthropic_messages,
    to_anthropic_tools,
)


def _anthropic_resp(content_blocks, model="claude-test", stop_reason="end_turn"):
    """构造 Anthropic 响应 dict（合成 fixture）。"""
    return {
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 20, "output_tokens": 8},
    }


def test_to_anthropic_messages_tool_messages_merge_into_single_user():
    msgs = [
        Message(
            "assistant",
            None,
            tool_calls=[
                ToolCall(id="t1", name="Read", arguments={"file_path": "a.py"}),
                ToolCall(id="t2", name="Grep", arguments={"pattern": "x"}),
            ],
        ),
        Message("tool", "file content", tool_call_id="t1", name="Read"),
        Message("tool", "2 matches", tool_call_id="t2", name="Grep"),
    ]
    system, out = to_anthropic_messages(msgs)
    assert system is None
    assert len(out) == 2  # assistant 一条 + user 一条（两条 tool_result 合并）
    assert out[0]["role"] == "assistant"
    blocks = out[0]["content"]
    assert [b["type"] for b in blocks] == ["tool_use", "tool_use"]
    assert blocks[0]["input"] == {"file_path": "a.py"}  # input 是 dict，非 JSON 字符串
    assert out[1]["role"] == "user"
    results = out[1]["content"]
    assert [b["tool_use_id"] for b in results] == ["t1", "t2"]


def test_to_anthropic_messages_system_role_merged_into_system():
    system, out = to_anthropic_messages(
        [Message("system", "base"), Message("user", "hi")], system="top"
    )
    assert system == "top\nbase"
    assert out == [{"role": "user", "content": "hi"}]


def test_to_anthropic_messages_adjacent_users_merged():
    system, out = to_anthropic_messages([Message("user", "a"), Message("user", "b")])
    assert len(out) == 1
    assert out[0] == {"role": "user", "content": "a\nb"}


def test_to_anthropic_tools_schema_type_object():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "read a file",
                "parameters": {"properties": {"file_path": {"type": "string"}}},
            },
        }
    ]
    out = to_anthropic_tools(tools)
    assert out[0]["name"] == "Read"
    assert out[0]["input_schema"]["type"] == "object"  # 兜底补 type
    assert "type" not in out[0]  # 无 OpenAI 的 function 包装


def test_from_anthropic_response_text_and_tool_use():
    resp = _anthropic_resp(
        [
            {"type": "text", "text": "I'll read the file."},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.py"}},
        ],
        stop_reason="tool_use",
    )
    r = from_anthropic_response(resp)
    assert r.content == "I'll read the file."
    assert r.tool_calls[0].id == "toolu_1"
    assert r.tool_calls[0].arguments == {"file_path": "a.py"}  # input 是 dict
    assert r.finish_reason == "tool_calls"  # stop_reason 映射
    assert r.usage.total_tokens == 28


def test_from_anthropic_response_max_tokens_maps_to_length():
    resp = _anthropic_resp([{"type": "text", "text": "cut off"}], stop_reason="max_tokens")
    r = from_anthropic_response(resp)
    assert r.finish_reason == "length"

"""Provider 双向翻译测试（用合成 fixture dict，不打真实 API）。"""
from __future__ import annotations

import json

import pytest

from agent.messages import Message, NormalizedResponse, ToolCall
from agent.provider import (
    FakeProvider,
    ScriptExhaustedError,
    from_openai_response,
    to_openai_messages,
)


def _resp(content=None, tool_calls=None, model="test-model", finish=None):
    """构造 OpenAI 兼容响应 dict（合成 fixture）。"""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "model": model,
        "choices": [
            {"message": msg, "finish_reason": finish or ("tool_calls" if tool_calls else "stop")}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_to_openai_messages_system_injected():
    out = to_openai_messages([Message("user", "hi")], system="be helpful")
    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_to_openai_messages_assistant_tool_calls_arguments_is_json_string():
    m = Message(
        "assistant",
        "thinking",
        tool_calls=[ToolCall(id="tc1", name="Read", arguments={"file_path": "a.py"})],
    )
    out = to_openai_messages([m])
    tc = out[0]["tool_calls"][0]
    assert tc["id"] == "tc1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "Read"
    # arguments 必须是 JSON 字符串（OpenAI 要求）；归一化层存的是 dict
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"file_path": "a.py"}


def test_to_openai_messages_tool_message_links_by_id():
    m = Message("tool", "result", tool_call_id="tc1", name="Read")
    out = to_openai_messages([m])
    assert out[0] == {
        "role": "tool",
        "tool_call_id": "tc1",
        "content": "result",
        "name": "Read",
    }


def test_from_openai_response_parses_tool_call_arguments_to_dict():
    resp = _resp(
        content=None,
        tool_calls=[
            {
                "id": "tc1",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"file_path": "a.py"}'},
            }
        ],
    )
    nr = from_openai_response(resp)
    assert nr.content is None
    assert nr.tool_calls[0].id == "tc1"
    assert nr.tool_calls[0].name == "Read"
    # 关键：arguments 解析成 dict（不是 JSON 字符串）
    assert nr.tool_calls[0].arguments == {"file_path": "a.py"}
    assert isinstance(nr.tool_calls[0].arguments, dict)
    assert nr.usage.total_tokens == 15
    assert nr.finish_reason == "tool_calls"
    assert nr.model == "test-model"


def test_from_openai_response_plain_text():
    resp = _resp(content="hello", model="m")
    nr = from_openai_response(resp)
    assert nr.content == "hello"
    assert nr.tool_calls is None
    assert nr.finish_reason == "stop"


def test_roundtrip_to_openai_then_back_preserves_arguments_dict():
    """归一化→OpenAI→归一化 往返：arguments 仍是 dict（热切换/resume 前提）。"""
    original = [
        Message(
            "assistant",
            "ok",
            tool_calls=[ToolCall(id="1", name="X", arguments={"a": 1, "b": [2, 3]})],
        )
    ]
    openai_msgs = to_openai_messages(original)
    resp = {
        "model": "m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": openai_msgs[0]["tool_calls"],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    nr = from_openai_response(resp)
    assert nr.tool_calls[0].arguments == {"a": 1, "b": [2, 3]}


async def test_fake_provider_script_consumed_and_records_calls():
    fp = FakeProvider([NormalizedResponse(content="hi"), NormalizedResponse(content="bye")])
    r1 = await fp.chat([Message("user", "x")])
    assert r1.content == "hi"
    r2 = await fp.chat([Message("user", "y")])
    assert r2.content == "bye"
    # 脚本耗尽抛 ScriptExhaustedError（明确失败而非 NoneType 崩溃）
    with pytest.raises(ScriptExhaustedError):
        await fp.chat([Message("user", "z")])
    # calls 被记录供断言
    assert len(fp.calls) == 3


async def test_fake_provider_callable_script():
    """动态响应：script 项可为 callable，用于测试基于消息内容的响应。"""
    def _echo(messages, tools):
        return NormalizedResponse(content=f"echo:{messages[-1].content}")

    fp = FakeProvider([_echo])
    r = await fp.chat([Message("user", "ping")])
    assert r.content == "echo:ping"

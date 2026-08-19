"""utils 测试。"""
from __future__ import annotations

from agent.messages import Message, ToolCall
from agent.utils import (
    estimate_messages_tokens,
    estimate_tokens,
    render_diff,
    truncate,
)


def test_truncate_short():
    assert truncate("hi") == "hi"


def test_truncate_long():
    long = "x" * 100
    out = truncate(long, limit=10)
    assert out.startswith("x" * 10)
    assert "truncated" in out


def test_render_diff_shows_changes():
    d = render_diff("a\nb\n", "a\nc\n")
    assert "-b" in d
    assert "+c" in d


def test_render_diff_no_changes():
    assert render_diff("same\n", "same\n") == "(no changes)"


def test_estimate_tokens_positive():
    assert estimate_tokens("hello world") > 0


def test_estimate_tokens_with_model():
    assert estimate_tokens("hello", model="gpt-4") > 0


def test_estimate_messages_tokens_includes_tools():
    """工具 schema token 必须被算进去（最常被遗忘）。"""
    msgs = [
        Message("user", "hi"),
        Message("assistant", "ok", tool_calls=[ToolCall(id="1", name="X", arguments={})]),
    ]
    without = estimate_messages_tokens(msgs)
    with_tools = estimate_messages_tokens(
        msgs, tools=[{"type": "function", "function": {"name": "X", "description": "d", "parameters": {}}}]
    )
    assert with_tools > without


def test_estimate_messages_tokens_grows_with_content():
    short = [Message("user", "hi")]
    long = [Message("user", "x" * 1000)]
    assert estimate_messages_tokens(long) > estimate_messages_tokens(short)

"""TraceStore 测试。"""
from __future__ import annotations

from agent.session import SessionStore
from agent.trace import TraceStore


def test_timeline_view(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.append_trace({"type": "turn_start", "data": {"user_input": "hi"}})
    s.append_trace(
        {
            "type": "llm_response",
            "data": {"model": "m", "usage": {"total_tokens": 10}, "finish": "stop", "tool_calls": 0},
        }
    )
    ts = TraceStore(s.trace_path)
    view = ts.timeline_view()
    assert "turn_start" in view
    assert "llm_response" in view
    assert "2 events" in view


def test_failed_view_finds_error(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.append_trace({"type": "tool_call", "data": {"name": "Read"}})
    s.append_trace({"type": "error", "data": {"tool": "Read", "error": "FileNotFoundError"}})
    s.append_trace({"type": "tool_result", "data": {"content_preview": "x"}})
    ts = TraceStore(s.trace_path)
    failed = ts.failed_view()
    assert "error" in failed.lower() or "failure" in failed.lower()
    assert "FileNotFoundError" in failed


def test_failed_view_finds_permission_deny(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.append_trace({"type": "permission", "data": {"verdict": "DENY", "reason": "hard deny"}})
    ts = TraceStore(s.trace_path)
    failed = ts.failed_view()
    assert "DENY" in failed


def test_cost_aggregates_tokens(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.append_trace(
        {
            "type": "llm_response",
            "data": {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        }
    )
    s.append_trace(
        {
            "type": "llm_response",
            "data": {"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
        }
    )
    ts = TraceStore(s.trace_path)
    cost = ts.cost()
    assert cost["prompt_tokens"] == 30
    assert cost["completion_tokens"] == 15
    assert cost["total_tokens"] == 45


def test_export_md(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.append_trace({"type": "turn_start", "data": {"user_input": "hi"}})
    ts = TraceStore(s.trace_path)
    md = ts.export_md()
    assert "Trace export" in md
    assert "Cost" in md
    assert "hi" in md


def test_empty_trace(tmp_path):
    s = SessionStore(root=str(tmp_path))
    ts = TraceStore(s.trace_path)
    assert "no trace" in ts.timeline_view().lower()
    cost = ts.cost()
    assert cost == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

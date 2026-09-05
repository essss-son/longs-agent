"""Compaction 测试：配对边界 + 单策略 RollingSummary（滚动摘要 + 台账）+ Compactor。"""
from __future__ import annotations

from agent.archive import ArchiveStore
from agent.compaction import (
    LEDGER_HEADER,
    Compactor,
    RollingSummary,
    _split_old_summary,
    find_pair_units,
)
from agent.messages import Message, NormalizedResponse, ToolCall
from agent.provider import FakeProvider


def test_find_pair_units_assistant_with_tools():
    msgs = [
        Message("user", "q"),
        Message("assistant", "thinking", tool_calls=[ToolCall(id="1", name="Read", arguments={})]),
        Message("tool", "result1", tool_call_id="1", name="Read"),
        Message("assistant", "done"),
    ]
    units = find_pair_units(msgs)
    assert units == [(0, 0), (1, 2), (3, 3)]


def test_find_pair_units_parallel_tool_calls():
    """一次 assistant 两个并行 tool_calls → 一个 unit 含 assistant + 2 tool。"""
    msgs = [
        Message(
            "assistant",
            "t",
            tool_calls=[
                ToolCall(id="1", name="A", arguments={}),
                ToolCall(id="2", name="B", arguments={}),
            ],
        ),
        Message("tool", "r1", tool_call_id="1", name="A"),
        Message("tool", "r2", tool_call_id="2", name="B"),
    ]
    assert find_pair_units(msgs) == [(0, 2)]


async def test_rolling_summary_replaces_old_with_summary():
    summary_resp = NormalizedResponse(content="summary of past")
    provider = FakeProvider([summary_resp])
    summarizer = RollingSummary(provider, keep_recent_units=2)
    msgs = [
        Message("user", "old1"), Message("assistant", "old1 reply"),
        Message("user", "old2"), Message("assistant", "old2 reply"),
        Message("user", "recent1"), Message("assistant", "recent1 reply"),
        Message("user", "recent2"), Message("assistant", "recent2 reply"),
    ]
    out = await summarizer.compact(msgs)
    assert out[0].content.startswith("[summary")
    assert "summary of past" in out[0].content
    assert out[-1].content == "recent2 reply"
    assert len(out) < len(msgs)


async def test_rolling_summary_archives_tool_and_builds_ledger(tmp_path):
    """换出 unit 里的 tool 输出 → t_ 归档 + 台账行（name+args）。"""
    store = ArchiveStore(tmp_path)
    summary_resp = NormalizedResponse(content="summary")
    provider = FakeProvider([summary_resp])
    summarizer = RollingSummary(provider, keep_recent_units=1, archive=store)
    msgs = [
        Message("assistant", "run", tool_calls=[ToolCall(id="9", name="Bash", arguments={"command": "pytest -x"})]),
        Message("tool", "119 passed", tool_call_id="9", name="Bash"),
        Message("user", "recent"),
        Message("assistant", "ok"),
    ]
    out = await summarizer.compact(msgs)
    # tool 原文内联归档为一条 t_
    assert any(r["kind"] == "tool" and r["content"] == "119 passed" for r in store._records)
    # 台账含 mem_id + 命令
    assert "早期工具调用台账" in out[0].content
    assert "pytest -x" in out[0].content


async def test_rolling_summary_envelope_passthrough(tmp_path):
    """信封截断的 tool（content 带 mem_id=t_）不重复归档，台账透传 mem_id。"""
    store = ArchiveStore(tmp_path)
    summary_resp = NormalizedResponse(content="summary")
    provider = FakeProvider([summary_resp])
    summarizer = RollingSummary(provider, keep_recent_units=1, archive=store)
    truncated = "⚠️ 输出过大已截断 | mem_id=t_0001 | full_output_path=/tmp/tool.json\nline0\nline1"
    msgs = [
        Message("assistant", "run", tool_calls=[ToolCall(id="9", name="Bash", arguments={"command": "pytest"})]),
        Message("tool", truncated, tool_call_id="9", name="Bash"),
        Message("user", "recent"),
        Message("assistant", "ok"),
    ]
    out = await summarizer.compact(msgs)
    assert store._records == []  # 透传：不重复归档
    assert "t_0001" in out[0].content  # 台账引用信封 mem_id


async def test_rolling_summary_non_tool_content_not_archived(tmp_path):
    """非工具内容只进摘要，不发 t_。"""
    store = ArchiveStore(tmp_path)
    summary_resp = NormalizedResponse(content="summary")
    provider = FakeProvider([summary_resp])
    summarizer = RollingSummary(provider, keep_recent_units=2, archive=store)
    msgs = [
        Message("user", "old1"), Message("assistant", "old1 reply"),
        Message("user", "old2"), Message("assistant", "old2 reply"),
        Message("user", "recent1"), Message("assistant", "recent1 reply"),
        Message("user", "recent2"), Message("assistant", "recent2 reply"),
    ]
    await summarizer.compact(msgs)
    assert store._records == []  # 无工具输出 → 无归档


async def test_ledger_truncated_to_limit(tmp_path):
    """台账截断到 ledger_limit 条（最老的被丢弃）。"""
    store = ArchiveStore(tmp_path)
    summary_resp = NormalizedResponse(content="summary")
    provider = FakeProvider([summary_resp])
    summarizer = RollingSummary(provider, keep_recent_units=1, archive=store, ledger_limit=2)
    msgs = []
    for i in range(5):
        msgs.append(Message("assistant", "run", tool_calls=[ToolCall(id=str(i), name="Bash", arguments={"command": f"c{i}"})]))
        msgs.append(Message("tool", f"out{i}", tool_call_id=str(i), name="Bash"))
    msgs += [Message("user", "recent"), Message("assistant", "ok")]
    out = await summarizer.compact(msgs)
    assert len(store._records) == 5  # 5 条 tool 全归档
    ledger = out[0].content
    assert "c4" in ledger and "c3" in ledger  # 保留最近 2 条
    assert "c0" not in ledger  # 最老被截掉


async def test_ledger_carries_over_across_compactions(tmp_path):
    """两次压缩：第一次的台账条目在第二次压缩后续传保留（程序解析，不靠 LLM 抄写）。"""
    store = ArchiveStore(tmp_path)
    provider = FakeProvider([
        NormalizedResponse(content="summary1"),
        NormalizedResponse(content="summary2"),
    ])
    rs = RollingSummary(provider, keep_recent_units=1, archive=store)
    msgs = [
        Message("assistant", "run", tool_calls=[ToolCall(id="1", name="Bash", arguments={"command": "pytest"})]),
        Message("tool", "out1", tool_call_id="1", name="Bash"),
        Message("user", "q1"),
    ]
    out1 = await rs.compact(msgs)
    assert "t_0001" in out1[0].content
    # 对话继续（out1 已是「摘要+台账+q1」合并消息），触发第二次压缩
    msgs2 = out1 + [
        Message("assistant", "run2", tool_calls=[ToolCall(id="2", name="Read", arguments={"file_path": "a.py"})]),
        Message("tool", "out2", tool_call_id="2", name="Read"),
        Message("user", "q2"),
    ]
    out2 = await rs.compact(msgs2)
    body = out2[0].content
    assert "t_0001" in body  # 旧条目续传
    assert "t_0002" in body  # 新条目
    # 旧摘要文本仍喂给了 LLM（滚动合并，非重写）
    user_prompt = provider.calls[1][0][-1].content
    assert "summary1" in user_prompt
    assert LEDGER_HEADER not in user_prompt  # 台账不进摘要 prompt


def test_split_old_summary_keeps_merged_user_text():
    """拆旧摘要消息：台账块后的 user 合并文本保留在摘要文本里。"""
    content = (
        "[summary of earlier turns]\nsummary1 text\n\n"
        "## 早期工具调用台账\n"
        "- t_0001 | Bash  {\"command\": \"pytest\"}\n"
        "- t_0002 | Read  {\"file_path\": \"a.py\"}\n"
        "q1 用户问题原文"
    )
    body, entries = _split_old_summary(content)
    assert body.endswith("q1 用户问题原文")
    assert "summary1 text" in body
    assert [e[1] for e in entries] == ["t_0001", "t_0002"]
    assert entries[0][2] == "Bash"


def test_compactor_should_compact_threshold():
    compactor = Compactor(provider=None, context_window=10000)
    # effective = 10000 - 8192 = 1808, 80% = 1446.4
    compactor.token_fn = lambda msgs, tools=None: 5000  # > 1446
    assert compactor.should_compact([Message("user", "x")])
    compactor.token_fn = lambda msgs, tools=None: 100  # < 1446
    assert not compactor.should_compact([Message("user", "x")])


def test_compactor_hysteresis():
    compactor = Compactor(provider=None, context_window=10000)
    compactor.token_fn = lambda msgs, tools=None: 5000
    assert compactor.should_compact([Message("user", "x")])
    compactor._just_compacted = True
    assert not compactor.should_compact([Message("user", "x")])


async def test_compactor_compact_runs_strategies():
    summary_resp = NormalizedResponse(content="summary")
    provider = FakeProvider([summary_resp])
    compactor = Compactor(provider=provider, context_window=10000)
    compactor.token_fn = lambda msgs, tools=None: 5000  # 超 80%，触发
    msgs = [Message("tool", f"r{i}", tool_call_id=str(i), name="T") for i in range(20)]
    out = await compactor.compact(msgs)
    assert compactor._just_compacted
    assert len(out) <= len(msgs)


async def test_rolling_summary_uses_five_section_template():
    """摘要 prompt 注入五段结构化模板。"""

    class CapturingProvider:
        def __init__(self, resp):
            self.resp = resp
            self.messages = None

        async def chat(self, messages, tools=None, system=None, **kw):
            self.messages = messages
            return self.resp

    provider = CapturingProvider(NormalizedResponse(content="summary"))
    summarizer = RollingSummary(provider, keep_recent_units=2)
    msgs = [
        Message("user", "old1"), Message("assistant", "old1 reply"),
        Message("user", "old2"), Message("assistant", "old2 reply"),
        Message("user", "recent1"), Message("assistant", "recent1 reply"),
        Message("user", "recent2"), Message("assistant", "recent2 reply"),
    ]
    await summarizer.compact(msgs)
    system_text = provider.messages[0].content
    assert "Objectives & Status" in system_text
    assert "Technical Context" in system_text
    assert "Completed Milestones" in system_text
    assert "Key Insights & Decisions" in system_text
    assert "File System State" in system_text

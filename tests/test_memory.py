"""memory / system prompt 装配测试。"""
from __future__ import annotations

from agent.memory import build_system_prompt, load_agent_md
from agent.tools import ToolRegistry
from agent.builtin_tools import Read, Write, Bash


def test_build_system_prompt_has_identity():
    registry = ToolRegistry()
    registry.register(Read())
    sp = build_system_prompt(registry=registry)
    assert "longs-agent" in sp
    assert "code agent" in sp.lower()


def test_build_system_prompt_has_tools_block():
    registry = ToolRegistry()
    registry.register(Read())
    registry.register(Write())
    registry.register(Bash())
    sp = build_system_prompt(registry=registry)
    assert "Available tools" in sp
    assert "Read" in sp and "read-only" in sp
    assert "Write" in sp
    assert "Bash" in sp


def test_build_system_prompt_has_agent_md(tmp_path):
    (tmp_path / "AGENT.md").write_text("# My project\nproject notes here", encoding="utf-8")
    agent_md = load_agent_md(str(tmp_path))
    sp = build_system_prompt(agent_md=agent_md, registry=ToolRegistry())
    assert "My project" in sp
    assert "project notes" in sp


def test_build_system_prompt_has_todos():
    sp = build_system_prompt(
        registry=ToolRegistry(),
        todos=[{"content": "task1", "status": "in_progress", "active_form": "doing task1"}],
    )
    assert "Current todos" in sp
    assert "doing task1" in sp
    assert "[~]" in sp


def test_build_system_prompt_no_todos_when_empty():
    sp = build_system_prompt(registry=ToolRegistry(), todos=[])
    assert "Current todos" not in sp


def test_build_system_prompt_has_skills():
    sp = build_system_prompt(
        registry=ToolRegistry(),
        skills_block="- my-skill: does thing (path/x)",
    )
    assert "my-skill" in sp


def test_build_system_prompt_order():
    """身份 → AGENT.md → 工具 → todos → skills 顺序。"""
    sp = build_system_prompt(
        agent_md="AGENTMD_MARKER",
        registry=None,
        todos=[{"content": "TODO_MARKER", "status": "pending"}],
        skills_block="SKILL_MARKER",
    )
    # 各段都在
    assert "AGENTMD_MARKER" in sp
    assert "TODO_MARKER" in sp
    assert "SKILL_MARKER" in sp
    # 顺序：身份(含 longs-agent) 最前
    assert sp.index("longs-agent") < sp.index("AGENTMD_MARKER")
    assert sp.index("AGENTMD_MARKER") < sp.index("TODO_MARKER")

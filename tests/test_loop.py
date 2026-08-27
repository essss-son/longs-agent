"""AgentLoop 多轮工具调用剧本测试（FakeProvider，确定性端到端）+ 权限。"""
from __future__ import annotations

from agent.builtin_tools import Bash, Edit, Read, Write
from agent.loop import AgentLoop
from agent.messages import NormalizedResponse, ToolCall
from agent.permissions import Mode, PermissionConfig
from agent.provider import FakeProvider
from agent.session import SessionStore
from agent.tools import ToolRegistry


async def test_loop_multi_turn_tool_call(tmp_project):
    target = str(tmp_project / "src" / "a.py")
    script = [
        NormalizedResponse(
            tool_calls=[ToolCall(id="tc1", name="Read", arguments={"file_path": target})]
        ),
        NormalizedResponse(content="文件内容是 x = 1"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    registry.register(Read())
    registry.register(Write())
    registry.register(Edit())
    session = SessionStore(root=str(tmp_project / ".agent" / "sessions"))
    loop = AgentLoop(provider, registry, session)

    answer = await loop.run_turn("读 a.py")
    assert answer == "文件内容是 x = 1"

    # session 落盘 4 条：user / assistant(tool_calls) / tool / assistant
    msgs = session.read_messages()
    assert len(msgs) == 4
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert msgs[1].tool_calls is not None
    assert msgs[2].role == "tool"
    assert msgs[2].tool_call_id == "tc1"
    assert msgs[3].role == "assistant"
    assert msgs[3].content == "文件内容是 x = 1"


async def test_loop_dispatch_error_text_feedback(tmp_project):
    # Read 不存在的文件 → 抛 FileNotFoundError → dispatch 捕获转 [error: ...]
    target = str(tmp_project / "nope.py")
    script = [
        NormalizedResponse(
            tool_calls=[ToolCall(id="tc1", name="Read", arguments={"file_path": target})]
        ),
        NormalizedResponse(content="ok"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    registry.register(Read())
    session = SessionStore(root=str(tmp_project / ".agent" / "sessions"))
    loop = AgentLoop(provider, registry, session, mode=Mode.AUTO)

    await loop.run_turn("读 nope")
    msgs = session.read_messages()
    tool_msg = msgs[2]
    assert tool_msg.role == "tool"
    assert "error" in tool_msg.content.lower()


async def test_loop_unknown_tool_feedback(tmp_project):
    script = [
        NormalizedResponse(
            tool_calls=[ToolCall(id="tc1", name="NoSuchTool", arguments={})]
        ),
        NormalizedResponse(content="done"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    session = SessionStore(root=str(tmp_project / ".agent" / "sessions"))
    loop = AgentLoop(provider, registry, session)

    await loop.run_turn("go")
    msgs = session.read_messages()
    assert "error" in msgs[2].content.lower()
    assert "unknown tool" in msgs[2].content.lower()


async def test_loop_hard_deny_returns_denied_message(tmp_project):
    """rm -rf / 在任何模式 hard deny → tool 消息含 [denied: hard deny]。"""
    script = [
        NormalizedResponse(
            tool_calls=[ToolCall(id="tc1", name="Bash", arguments={"command": "rm -rf /"})]
        ),
        NormalizedResponse(content="ok"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    registry.register(Bash())
    session = SessionStore(root=str(tmp_project / ".agent" / "sessions"))
    loop = AgentLoop(provider, registry, session, mode=Mode.MANUAL)
    await loop.run_turn("rm")
    msgs = session.read_messages()
    assert msgs[2].role == "tool"
    assert "denied" in msgs[2].content.lower()
    assert "hard" in msgs[2].content.lower()


async def test_loop_ask_without_repl_default_deny(tmp_project):
    """无 repl 时 ASK 默认拒绝（测试环境无 REPL 交互）。"""
    script = [
        NormalizedResponse(
            tool_calls=[ToolCall(id="tc1", name="Bash", arguments={"command": "echo hi"})]
        ),
        NormalizedResponse(content="ok"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    registry.register(Bash())
    session = SessionStore(root=str(tmp_project / ".agent" / "sessions"))
    loop = AgentLoop(provider, registry, session, mode=Mode.MANUAL)
    await loop.run_turn("go")
    msgs = session.read_messages()
    assert "denied" in msgs[2].content.lower()


async def test_loop_auto_mode_allows_non_disaster(tmp_project):
    """AUTO 模式：echo hi 放行执行（hard deny 仍拦，见 D4 test_permissions）。"""
    script = [
        NormalizedResponse(
            tool_calls=[ToolCall(id="tc1", name="Bash", arguments={"command": "echo hi"})]
        ),
        NormalizedResponse(content="done"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    registry.register(Bash())
    session = SessionStore(root=str(tmp_project / ".agent" / "sessions"))
    loop = AgentLoop(provider, registry, session, mode=Mode.AUTO)
    await loop.run_turn("go")
    msgs = session.read_messages()
    # AUTO 放行：tool 消息含命令输出，非 denied
    assert "hi" in msgs[2].content.lower()
    assert "denied" not in msgs[2].content.lower()


# ---- 乐观锁（harness 自动校验，不依赖模型回传 sha） ----


def _make_auto_loop(tmp_project, *tools):
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    session = SessionStore(root=str(tmp_project / ".agent" / "sessions"))
    loop = AgentLoop(
        None, registry, session,
        mode=Mode.AUTO,
        permission_config=PermissionConfig(project_root=str(tmp_project)),
    )
    return loop


async def test_optimistic_lock_edit_rejected_after_external_change(tmp_project):
    """Read 后外部改文件 → Edit 被拒，文件保持外部改动。"""
    target = str(tmp_project / "src" / "a.py")  # "x = 1\n"
    loop = _make_auto_loop(tmp_project, Read(), Edit())
    # 1. Read 记录文件 hash
    await loop.dispatch(ToolCall(id="1", name="Read", arguments={"file_path": target}))
    # 2. 外部改动
    (tmp_project / "src" / "a.py").write_text("x = 999\n", encoding="utf-8")
    # 3. Edit 被乐观锁拒绝
    msg = await loop.dispatch(
        ToolCall(id="2", name="Edit",
                 arguments={"file_path": target, "old_string": "x = 1", "new_string": "x = 2"})
    )
    assert "file changed" in msg.content.lower()
    assert (tmp_project / "src" / "a.py").read_text() == "x = 999\n"


async def test_optimistic_lock_edit_succeeds_then_cache_updates(tmp_project):
    """Read 后无改动 → Edit 成功；成功后缓存更新，可连续 Edit 不被自己卡住。"""
    target = str(tmp_project / "src" / "a.py")
    loop = _make_auto_loop(tmp_project, Read(), Edit())
    await loop.dispatch(ToolCall(id="1", name="Read", arguments={"file_path": target}))
    msg1 = await loop.dispatch(
        ToolCall(id="2", name="Edit",
                 arguments={"file_path": target, "old_string": "x = 1", "new_string": "x = 2"})
    )
    assert "replaced" in msg1.content
    assert (tmp_project / "src" / "a.py").read_text() == "x = 2\n"
    # 缓存已更新为新 hash，第二次 Edit 不被自己卡住
    msg2 = await loop.dispatch(
        ToolCall(id="3", name="Edit",
                 arguments={"file_path": target, "old_string": "x = 2", "new_string": "x = 3"})
    )
    assert "replaced" in msg2.content
    assert (tmp_project / "src" / "a.py").read_text() == "x = 3\n"


async def test_optimistic_lock_no_cache_edit_allowed(tmp_project):
    """从没 Read 过的文件 → 无缓存不校验，Edit 正常（保持宽松）。"""
    target = str(tmp_project / "src" / "a.py")
    loop = _make_auto_loop(tmp_project, Edit())
    msg = await loop.dispatch(
        ToolCall(id="1", name="Edit",
                 arguments={"file_path": target, "old_string": "x = 1", "new_string": "x = 2"})
    )
    assert "replaced" in msg.content
    assert (tmp_project / "src" / "a.py").read_text() == "x = 2\n"


async def test_optimistic_lock_persists_across_resume(tmp_project):
    """乐观锁状态持久化：同 sid 重建（模拟 resume）后锁仍生效。"""
    target = str(tmp_project / "src" / "a.py")
    root = str(tmp_project / ".agent" / "sessions")

    def _loop(session):
        registry = ToolRegistry()
        registry.register(Read())
        registry.register(Edit())
        return AgentLoop(
            None, registry, session,
            mode=Mode.AUTO,
            permission_config=PermissionConfig(project_root=str(tmp_project)),
        )

    session1 = SessionStore(root=root)
    await _loop(session1).dispatch(
        ToolCall(id="1", name="Read", arguments={"file_path": target})
    )
    # 模拟退出后外部改动
    (tmp_project / "src" / "a.py").write_text("x = 999\n", encoding="utf-8")
    # 同 sid 重建（resume）：__init__ 从 file_hashes.json 读回锁状态
    session2 = SessionStore(root=root, sid=session1.sid)
    msg = await _loop(session2).dispatch(
        ToolCall(id="2", name="Edit",
                 arguments={"file_path": target, "old_string": "x = 1", "new_string": "x = 2"})
    )
    assert "file changed" in msg.content.lower()
    assert (tmp_project / "src" / "a.py").read_text() == "x = 999\n"

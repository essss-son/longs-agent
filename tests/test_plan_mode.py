"""Plan mode 双重安全 + ExitPlanMode 审批流程测试。"""
from __future__ import annotations

from agent.builtin_tools import Write
from agent.loop import AgentLoop
from agent.messages import NormalizedResponse, ToolCall
from agent.permissions import Mode
from agent.plan_mode import ExitPlanMode, parse_plan_to_todos
from agent.provider import FakeProvider
from agent.session import SessionStore
from agent.todo import TodoStore
from agent.tools import ToolRegistry


class _FakeRepl:
    """测试用：approve_plan 返回预设，ask_permission 默认拒绝。"""

    def __init__(self, approve: bool = True):
        self._approve = approve
        self.plans_seen: list[str] = []

    async def ask_permission(self, tc) -> str:
        return "n"

    async def approve_plan(self, plan: str) -> bool:
        self.plans_seen.append(plan)
        return self._approve


def test_exit_plan_mode_is_readonly():
    """ExitPlanMode 必须 read_only=True 才能通过 plan 模式注册表过滤。"""
    assert ExitPlanMode().read_only is True


def test_parse_plan_checklist():
    plan = "## Plan\n- [ ] step 1\n- [ ] step 2\n- [x] done step\n"
    todos = parse_plan_to_todos(plan)
    assert len(todos) == 3
    assert todos[0].content == "step 1"
    assert todos[2].content == "done step"
    from agent.todo import TodoStatus

    assert todos[2].status == TodoStatus.COMPLETED


def test_parse_plan_no_checklist_falls_back():
    """无 checklist 落单个 'Execute plan' todo。"""
    todos = parse_plan_to_todos("just a plan narrative, no checklist")
    assert len(todos) == 1
    assert todos[0].content == "Execute plan"


def test_plan_mode_filters_write_tools(tmp_path):
    """plan 模式 _active_tools 不含 Write/Bash，含 Read + ExitPlanMode。"""
    from agent.builtin_tools import Bash, Read

    registry = ToolRegistry()
    registry.register(Read())
    registry.register(Write())
    registry.register(Bash())
    registry.register(ExitPlanMode())
    session = SessionStore(root=str(tmp_path / ".agent" / "sessions"))
    loop = AgentLoop(FakeProvider([]), registry, session, mode=Mode.PLAN)

    names = [t["function"]["name"] for t in loop._active_tools()]
    assert "Read" in names
    assert "ExitPlanMode" in names
    assert "Write" not in names  # 写工具被过滤
    assert "Bash" not in names


async def test_plan_mode_dispatch_write_denied(tmp_path):
    """plan 模式 dispatch 写工具 → 兜底 deny（双重安全第二层）。"""
    script = [
        NormalizedResponse(
            tool_calls=[
                ToolCall(
                    id="tc1", name="Write",
                    arguments={"file_path": str(tmp_path / "x"), "content": "y"},
                )
            ]
        ),
        NormalizedResponse(content="done"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    registry.register(Write())
    session = SessionStore(root=str(tmp_path / ".agent" / "sessions"))
    loop = AgentLoop(provider, registry, session, mode=Mode.PLAN)
    await loop.run_turn("write")
    msgs = session.read_messages()
    assert "denied" in msgs[2].content.lower()
    assert "plan" in msgs[2].content.lower()
    # 文件未被写
    assert not (tmp_path / "x").exists()


async def test_exit_plan_mode_approve_exits_and_sets_todos(tmp_path):
    plan_text = "## Plan\n- [ ] step 1\n- [ ] step 2\n"
    script = [
        NormalizedResponse(
            tool_calls=[ToolCall(id="tc1", name="ExitPlanMode", arguments={"plan": plan_text})]
        ),
        NormalizedResponse(content="ok"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    registry.register(ExitPlanMode())
    session = SessionStore(root=str(tmp_path / ".agent" / "sessions"))
    todo_store = TodoStore(path=session.todo_path)
    loop = AgentLoop(
        provider, registry, session, mode=Mode.PLAN, todo_store=todo_store
    )
    loop.repl = _FakeRepl(approve=True)
    await loop.run_turn("plan")
    assert loop.mode == Mode.MANUAL  # 退出 plan
    todos = todo_store.all()
    assert len(todos) == 2
    assert todos[0].content == "step 1"


async def test_exit_plan_mode_reject_stays_in_plan(tmp_path):
    plan_text = "## Plan\n- [ ] step 1\n"
    script = [
        NormalizedResponse(
            tool_calls=[ToolCall(id="tc1", name="ExitPlanMode", arguments={"plan": plan_text})]
        ),
        NormalizedResponse(content="ok"),
    ]
    provider = FakeProvider(script)
    registry = ToolRegistry()
    registry.register(ExitPlanMode())
    session = SessionStore(root=str(tmp_path / ".agent" / "sessions"))
    todo_store = TodoStore(path=session.todo_path)
    loop = AgentLoop(
        provider, registry, session, mode=Mode.PLAN, todo_store=todo_store
    )
    loop.repl = _FakeRepl(approve=False)
    await loop.run_turn("plan")
    assert loop.mode == Mode.PLAN  # 拒绝后仍 plan
    assert len(todo_store.all()) == 0  # 未落 todos

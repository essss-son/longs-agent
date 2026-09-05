"""Todo 三态不变量 + TodoStore + TodoWrite 测试。"""
from __future__ import annotations

from agent.todo import Todo, TodoStatus, TodoStore, TodoWrite


def test_todo_roundtrip():
    t = Todo("task", TodoStatus.IN_PROGRESS, "doing task")
    d = t.to_dict()
    assert d["status"] == "in_progress"
    assert d["active_form"] == "doing task"
    t2 = Todo.from_dict(d)
    assert t2.status == TodoStatus.IN_PROGRESS
    assert t2.active_form == "doing task"


def test_todo_default_active_form_is_content():
    t = Todo("my task")
    assert t.active_form == ""  # dataclass 默认空
    assert t.to_dict()["active_form"] == "my task"  # 序列化时回退到 content


def test_set_only_one_in_progress(tmp_path):
    s = TodoStore(path=str(tmp_path / "todo.json"))
    s.set([
        Todo("a", TodoStatus.IN_PROGRESS),
        Todo("b", TodoStatus.IN_PROGRESS),  # 应降为 pending
        Todo("c", TodoStatus.PENDING),
    ])
    in_progress = [t for t in s.all() if t.status == TodoStatus.IN_PROGRESS]
    assert len(in_progress) == 1
    assert in_progress[0].content == "a"
    # b 被降级
    b = [t for t in s.all() if t.content == "b"][0]
    assert b.status == TodoStatus.PENDING


def test_update_to_in_progress_demotes_others(tmp_path):
    s = TodoStore(path=str(tmp_path / "todo.json"))
    s.set([Todo("a", TodoStatus.IN_PROGRESS), Todo("b")])
    s.update(1, TodoStatus.IN_PROGRESS)
    todos = s.all()
    assert todos[0].status == TodoStatus.PENDING  # a 被降级
    assert todos[1].status == TodoStatus.IN_PROGRESS


def test_persist_and_load(tmp_path):
    p = str(tmp_path / "todo.json")
    s1 = TodoStore(path=p)
    s1.set([Todo("a"), Todo("b", TodoStatus.COMPLETED)])
    s2 = TodoStore(path=p)  # 重新加载
    assert len(s2.all()) == 2
    assert s2.all()[1].status == TodoStatus.COMPLETED


def test_update_out_of_range_raises(tmp_path):
    import pytest
    s = TodoStore(path=str(tmp_path / "todo.json"))
    s.set([Todo("a")])
    with pytest.raises(IndexError):
        s.update(5, TodoStatus.IN_PROGRESS)


async def test_todowrite_tool_sets_todos(tmp_path):
    s = TodoStore(path=str(tmp_path / "todo.json"))
    tw = TodoWrite(s)
    result = await tw.execute(todos=[
        {"content": "task1", "status": "pending"},
        {"content": "task2", "status": "in_progress", "active_form": "doing task2"},
    ])
    assert "set 2 todos" in result
    todos = s.all()
    assert len(todos) == 2
    assert todos[1].status == TodoStatus.IN_PROGRESS
    assert todos[0].status == TodoStatus.PENDING  # 同时只一个 in_progress
    assert todos[1].active_form == "doing task2"

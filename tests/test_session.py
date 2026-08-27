"""SessionStore 持久化 + 容错 + resume 测试。"""
from __future__ import annotations

import json

from agent.messages import Message
from agent.session import SessionStore


def test_append_and_read_messages(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.append_message(Message("user", "hi"))
    s.append_message(Message("assistant", "hello"))
    msgs = s.read_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "hi"
    assert msgs[1].content == "hello"


def test_read_tolerates_truncated_last_line(tmp_path):
    """末行半截（无换行符，Ctrl+C 中断）静默跳过。"""
    s = SessionStore(root=str(tmp_path))
    p = s.messages_path
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "user", "content": "a"}) + "\n")
        f.write(json.dumps({"role": "user", "content": "b"}) + "\n")
        f.write('{"role": "user", "content": "half')  # 半截无换行
    assert len(s.read_messages()) == 2  # 半截跳过


def test_read_tolerates_corrupt_middle_line(tmp_path):
    """中间坏行跳过。"""
    s = SessionStore(root=str(tmp_path))
    p = s.messages_path
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "user", "content": "a"}) + "\n")
        f.write("this is not json\n")  # 坏行
        f.write(json.dumps({"role": "user", "content": "b"}) + "\n")
    assert len(s.read_messages()) == 2  # 坏行跳过


def test_append_and_read_trace(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.append_trace({"type": "turn_start", "data": {"user_input": "hi"}})
    s.append_trace({"type": "llm_response", "data": {"model": "m"}})
    trace = s.read_trace()
    assert len(trace) == 2
    assert trace[0]["type"] == "turn_start"
    assert trace[0]["seq"] == 1
    assert trace[1]["seq"] == 2
    assert "ts" in trace[0]


def test_write_and_read_meta(tmp_path):
    s = SessionStore(root=str(tmp_path))
    meta = {"sid": s.sid, "model_alias": "demo", "mode": "NORMAL"}
    s.write_meta(meta)
    assert s.read_meta() == meta


def test_read_meta_missing_returns_empty(tmp_path):
    s = SessionStore(root=str(tmp_path))
    assert s.read_meta() == {}


def test_meta_atomic_surives_partial_tmp(tmp_path):
    """write_meta 原子：写半截 tmp 但未 replace 时，旧 meta.json 仍可读。"""
    s = SessionStore(root=str(tmp_path))
    s.write_meta({"sid": s.sid, "mode": "NORMAL"})
    # 模拟中断：留一个半截 tmp，但不调 os.replace
    tmp = s.meta_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write('{"partial":')  # 半截 tmp
    meta = s.read_meta()
    assert meta.get("mode") == "NORMAL"  # 旧 meta 未被腐蚀


def test_write_and_read_todos(tmp_path):
    s = SessionStore(root=str(tmp_path))
    todos = [{"content": "task1", "status": "pending"}]
    s.write_todos(todos)
    assert s.read_todos() == todos


def test_write_and_read_file_hashes(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.save_file_hashes({"/a.py": "hash1", "/b.py": "hash2"})
    assert s.read_file_hashes() == {"/a.py": "hash1", "/b.py": "hash2"}


def test_read_file_hashes_missing_returns_empty(tmp_path):
    s = SessionStore(root=str(tmp_path))
    assert s.read_file_hashes() == {}


def test_read_file_hashes_tolerates_corrupt(tmp_path):
    s = SessionStore(root=str(tmp_path))
    s.file_hashes_path.write_text("{not json", encoding="utf-8")
    assert s.read_file_hashes() == {}


def test_resume_loads_history(tmp_path):
    """写 session A，新建 SessionStore(sid=A) 读回历史。"""
    s1 = SessionStore(root=str(tmp_path))
    s1.append_message(Message("user", "first"))
    s1.append_message(Message("assistant", "reply"))
    sid = s1.sid

    s2 = SessionStore(root=str(tmp_path), sid=sid)
    msgs = s2.read_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "first"
    assert msgs[1].content == "reply"


def test_list_sessions(tmp_path):
    SessionStore(root=str(tmp_path), sid="aaa11111")
    SessionStore(root=str(tmp_path), sid="bbb22222")
    sids = SessionStore.list_sessions(root=str(tmp_path))
    assert set(sids) == {"aaa11111", "bbb22222"}


# ---- 文件级 Checkpoint（三线回滚） ----


def test_undo_restores_file_content(tmp_path):
    s = SessionStore(root=str(tmp_path))
    f = tmp_path / "f.txt"
    f.write_text("v0\n", encoding="utf-8")
    s.save_file_snapshot(seq=10, file_path=str(f), assistant_index=1)
    f.write_text("v1\n", encoding="utf-8")  # 模拟写操作
    msg = s.undo_last_write()
    assert f.read_text() == "v0\n"
    assert "回滚" in msg


def test_undo_deletes_newly_created_file(tmp_path):
    s = SessionStore(root=str(tmp_path))
    f = tmp_path / "new.txt"
    # 原本不存在 → 快照记 was_absent
    s.save_file_snapshot(seq=10, file_path=str(f), assistant_index=1)
    f.write_text("created\n", encoding="utf-8")  # 模拟新建写操作
    s.undo_last_write()
    assert not f.exists()  # 回滚后文件被删除


def test_rewind_to_user_restores_earlier_turn(tmp_path):
    s = SessionStore(root=str(tmp_path))
    f = tmp_path / "f.txt"
    # 三条用户消息（user_index 0/4/8），各触发一次写
    f.write_text("v0\n", encoding="utf-8")
    s.save_file_snapshot(seq=10, file_path=str(f), assistant_index=1, user_index=0)  # 消息1 写前 v0
    f.write_text("v1\n", encoding="utf-8")
    s.save_file_snapshot(seq=20, file_path=str(f), assistant_index=5, user_index=4)  # 消息2 写前 v1
    f.write_text("v2\n", encoding="utf-8")
    s.save_file_snapshot(seq=30, file_path=str(f), assistant_index=9, user_index=8)  # 消息3 写前 v2
    f.write_text("v3\n", encoding="utf-8")
    # 回退到最后一条（消息3）→ 无变化
    assert "无需回退" in s.rewind_to_user(8)
    assert f.read_text() == "v3\n"
    # 回退到消息2 → 撤销消息3 的写，回到 v2
    s.rewind_to_user(4)
    assert f.read_text() == "v2\n"
    # 回退到消息1 → 撤销消息2、3 的写，回到 v1
    s.rewind_to_user(0)
    assert f.read_text() == "v1\n"


def test_rewind_to_user_truncates_messages(tmp_path):
    s = SessionStore(root=str(tmp_path))
    f = tmp_path / "f.txt"
    for m in [
        Message("user", "消息1"),
        Message("assistant", "写1"),
        Message("tool", "r1"),
        Message("assistant", "回复1"),
        Message("user", "消息2"),
        Message("assistant", "写2"),
        Message("tool", "r2"),
        Message("assistant", "回复2"),
    ]:
        s.append_message(m)
    f.write_text("v0\n", encoding="utf-8")
    s.save_file_snapshot(seq=10, file_path=str(f), assistant_index=1, user_index=0)
    f.write_text("v1\n", encoding="utf-8")
    s.save_file_snapshot(seq=20, file_path=str(f), assistant_index=5, user_index=4)
    f.write_text("v2\n", encoding="utf-8")
    # 回退到消息1 → messages 截断到消息2（idx 4）之前，剩 4 条
    s.rewind_to_user(0)
    assert len(s.read_messages()) == 4
    assert s.read_messages()[-1].content == "回复1"


def test_undo_truncates_messages_to_assistant_index(tmp_path):
    s = SessionStore(root=str(tmp_path))
    f = tmp_path / "f.txt"
    f.write_text("v0\n", encoding="utf-8")
    s.append_message(Message("user", "hi"))
    s.append_message(Message("assistant", "calling tools"))  # 索引 1
    # 写操作前快照：assistant_index=1 → 回滚保留前 1 条（去掉 assistant 及之后）
    s.save_file_snapshot(seq=10, file_path=str(f), assistant_index=1)
    s.append_message(Message("tool", "result"))
    assert len(s.read_messages()) == 3
    s.undo_last_write()
    msgs = s.read_messages()
    assert len(msgs) == 1
    assert msgs[0].content == "hi"

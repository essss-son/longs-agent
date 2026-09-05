"""L2 档案层测试：ArchiveStore 读写回环 / mem_id 续号 / 容错读 / MemoryRead 取回。"""
from __future__ import annotations

import json

from agent.archive import ArchiveStore, MemoryRead


def test_archive_store_write_read_roundtrip(tmp_path):
    store = ArchiveStore(tmp_path)
    mid = store.archive("hello world", tool_name="Read", tool_call_id="1")
    assert mid == "t_0001"
    r = store.read(mid)
    assert r is not None
    assert r["content"] == "hello world"
    assert r["tool_name"] == "Read"
    assert r["char_count"] == 11
    assert "hello world" in r["preview"]


def test_archive_store_mem_id_sequential(tmp_path):
    store = ArchiveStore(tmp_path)
    ids = [store.archive(f"content{i}") for i in range(3)]
    assert ids == ["t_0001", "t_0002", "t_0003"]


def test_archive_store_resume_continues_numbering(tmp_path):
    """resume 场景：读已有 archive.jsonl，mem_id 从已有数量续号。"""
    store1 = ArchiveStore(tmp_path)
    store1.archive("old")
    store1.archive("older")
    # 新会话读同一档案目录
    store2 = ArchiveStore(tmp_path)
    assert len(store2._records) == 2
    mid = store2.archive("new")
    assert mid == "t_0003"


def test_archive_store_tolerant_read_half_line(tmp_path):
    """半截行（Ctrl+C 中断 write）静默跳过。"""
    p = tmp_path / "archive.jsonl"
    full1 = json.dumps({"mem_id": "t_0001", "content": "ok", "kind": "tool"})
    half = '{"mem_id": "t_0002", "content": "half'  # 无换行 → 半截
    full3 = json.dumps({"mem_id": "t_0003", "content": "ok3", "kind": "tool"})
    p.write_text(full1 + "\n" + half + "\n" + full3 + "\n", encoding="utf-8")
    store = ArchiveStore(tmp_path)
    assert len(store._records) == 2  # t_0002 半截被跳过
    assert store.read("t_0003")["content"] == "ok3"


def test_has_tool_call_id_dedups(tmp_path):
    store = ArchiveStore(tmp_path)
    assert not store.has_tool_call_id("dup1")
    store.archive("full", tool_name="Read", tool_call_id="dup1")
    assert store.has_tool_call_id("dup1")


def test_archive_with_file_path_pointer(tmp_path):
    """信封大输出：只存 file_path 指针，char_count 覆盖 len(content)。"""
    store = ArchiveStore(tmp_path)
    f = tmp_path / "tool_1.json"
    f.write_text("x" * 3000, encoding="utf-8")
    mid = store.archive("", tool_name="Bash", file_path=str(f), char_count=3000)
    r = store.read(mid)
    assert r["file_path"] == str(f)
    assert r["char_count"] == 3000


async def test_memory_read_returns_content_with_header(tmp_path):
    store = ArchiveStore(tmp_path)
    mid = store.archive("full original content here", tool_name="Read", tool_call_id="1")
    tool = MemoryRead(store)
    out = await tool.execute(mem_id=mid)
    assert "[archive t_0001" in out
    assert "full original content here" in out
    assert "历史快照" in out


async def test_memory_read_truncates_large_content(tmp_path):
    store = ArchiveStore(tmp_path)
    big = "x" * 50000
    mid = store.archive(big)
    tool = MemoryRead(store)
    out = await tool.execute(mem_id=mid, max_chars=1000)
    assert "truncated at 1000 chars" in out
    assert len(out) < 50000


async def test_memory_read_missing_mem_id(tmp_path):
    store = ArchiveStore(tmp_path)
    tool = MemoryRead(store)
    out = await tool.execute(mem_id="t_9999")
    assert "error" in out


async def test_memory_read_reads_file_when_file_path(tmp_path):
    """带 file_path 的记录：MemoryRead 读文件并解包信封 JSON（返回干净原文）。"""
    store = ArchiveStore(tmp_path)
    f = tmp_path / "tool_1.json"
    f.write_text(json.dumps({"tool": "Bash", "content": "full dumped output"}, ensure_ascii=False), encoding="utf-8")
    mid = store.archive("", tool_name="Bash", file_path=str(f), char_count=17)
    tool = MemoryRead(store)
    out = await tool.execute(mem_id=mid)
    assert "full dumped output" in out
    assert '{"tool":' not in out  # 不带 JSON 包装


async def test_memory_read_raw_file_fallback(tmp_path):
    """file_path 指向非 JSON 文件：原样返回，不报错。"""
    store = ArchiveStore(tmp_path)
    f = tmp_path / "note.txt"
    f.write_text("plain text output", encoding="utf-8")
    mid = store.archive("", tool_name="Bash", file_path=str(f), char_count=17)
    tool = MemoryRead(store)
    out = await tool.execute(mem_id=mid)
    assert "plain text output" in out


async def test_memory_read_returns_oversized_content_verbatim(tmp_path):
    """超长内容取回：原样返回（自带 max_chars 护栏），无需外部信封再截断。

    dispatch 层已对 MemoryRead 豁免统一信封——否则同一段原文会落盘出第二个
    mem_id，破坏"一级寻址"。本测试锁住 execute 自身不做信封处理。
    """
    store = ArchiveStore(tmp_path)
    big = "\n".join(f"line{i}" for i in range(500))  # 500 行，远超信封 100 行阈值
    mid = store.archive(big, tool_name="Bash")
    tool = MemoryRead(store)
    out = await tool.execute(mem_id=mid)
    assert "line0" in out and "line499" in out  # 首尾都在，原样返回
    assert "已截断" not in out  # 没有信封头

"""工具返回信封测试：截断 + 落盘 + 指针。"""
from __future__ import annotations

import json
from pathlib import Path

from agent.envelope import wrap


def test_small_content_passthrough():
    assert wrap("hello", tool_name="Bash") == "hello"


def test_long_content_truncates_and_dumps(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    content = "\n".join(f"line{i}" for i in range(600))  # 600 行 > 500
    out = wrap(content, tool_name="Bash")
    assert "已截断" in out
    assert "original_lines=600" in out
    # 落盘文件存在且内容完整
    path = Path(out.split("full_output_path=")[1].split("\n")[0])
    assert path.exists()
    dumped = json.loads(path.read_text(encoding="utf-8"))
    assert dumped["content"] == content


def test_large_bytes_triggers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    content = "x" * 20000  # 1 行但 20k 字节 > 12800
    out = wrap(content, tool_name="Read")
    assert "已截断" in out


def test_head_tail_direction(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    content = "\n".join(f"line{i}" for i in range(600))
    out = wrap(content, tool_name="Grep", direction="head_tail")
    assert "line0" in out
    assert "line599" in out
    assert "中间省略" in out
    assert "line50" not in out  # 中间被省略


def test_tail_direction(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    content = "\n".join(f"line{i}" for i in range(600))
    out = wrap(content, tool_name="Grep", direction="tail")
    assert "line599" in out
    assert "line0" not in out  # 后 100 行不含 line0


def test_no_dump_truncates_without_file(monkeypatch, tmp_path):
    """dump=False（子代理）：截断生效，但不落盘、无指针。"""
    monkeypatch.chdir(tmp_path)
    content = "\n".join(f"line{i}" for i in range(600))
    out = wrap(content, tool_name="Read", dump=False)
    assert "已截断" in out
    assert "未落盘" in out
    assert "full_output_path" not in out
    assert "line0" in out  # head 截断保留前 100 行
    assert "line500" not in out
    assert not (tmp_path / "tool-output").exists()  # 目录都没建
    assert list(tmp_path.rglob("tool_*.json")) == []  # 任何位置都没落盘文件

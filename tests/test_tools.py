"""内置工具测试（tmp fixture 真实执行）。"""
from __future__ import annotations

from agent.builtin_tools import Bash, Edit, Glob, Grep, Read, Write


async def test_read_returns_numbered(tmp_project):
    r = Read()
    out = await r.execute(file_path=str(tmp_project / "src" / "a.py"))
    assert "1" in out
    assert "x = 1" in out


async def test_write_creates_file(tmp_project):
    w = Write()
    p = tmp_project / "new.txt"
    result = await w.execute(file_path=str(p), content="hello")
    assert p.read_text() == "hello"
    assert "wrote" in result


async def test_edit_precise_replace(tmp_project):
    p = tmp_project / "src" / "a.py"
    e = Edit()
    result = await e.execute(file_path=str(p), old_string="x = 1", new_string="x = 2")
    assert "replaced" in result
    assert p.read_text() == "x = 2\n"


async def test_edit_non_unique_without_replace_all_errors(tmp_project):
    p = tmp_project / "dup.txt"
    p.write_text("a\na\n", encoding="utf-8")
    e = Edit()
    result = await e.execute(file_path=str(p), old_string="a", new_string="b")
    assert "error" in result.lower()
    assert "not unique" in result.lower() or "2 times" in result
    # 文件未改
    assert p.read_text() == "a\na\n"


async def test_edit_replace_all(tmp_project):
    p = tmp_project / "dup.txt"
    p.write_text("a\na\n", encoding="utf-8")
    e = Edit()
    result = await e.execute(file_path=str(p), old_string="a", new_string="b", replace_all=True)
    assert "replaced" in result
    assert p.read_text() == "b\nb\n"


async def test_edit_old_string_not_found(tmp_project):
    p = tmp_project / "src" / "a.py"
    e = Edit()
    result = await e.execute(file_path=str(p), old_string="nope", new_string="x")
    assert "error" in result.lower()
    assert "not found" in result.lower()
    assert p.read_text() == "x = 1\n"


async def test_bash_runs_command(tmp_project):
    b = Bash()
    result = await b.execute(command="echo hello")
    assert "hello" in result


async def test_bash_timeout(tmp_project):
    b = Bash()
    result = await b.execute(command="sleep 5", timeout=1)
    assert "timed out" in result.lower()


async def test_glob_finds_files(tmp_project):
    g = Glob()
    result = await g.execute(pattern="**/*.py", path=str(tmp_project))
    assert "src/a.py" in result


async def test_glob_ignores_git(tmp_project):
    (tmp_project / ".git").mkdir()
    (tmp_project / ".git" / "config").write_text("x", encoding="utf-8")
    g = Glob()
    result = await g.execute(pattern="**/*", path=str(tmp_project))
    assert ".git" not in result


async def test_grep_finds_pattern(tmp_project):
    g = Grep()
    result = await g.execute(pattern="x = 1", path=str(tmp_project))
    assert "a.py" in result


async def test_grep_no_match(tmp_project):
    g = Grep()
    result = await g.execute(pattern="zzznotexist", path=str(tmp_project))
    assert "no matches" in result.lower()


async def test_read_offset_and_limit(tmp_project):
    r = Read()
    p = tmp_project / "multi.txt"
    p.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
    out = await r.execute(file_path=str(p), offset=2, limit=2)
    # 行号是绝对行号：从 2 开始，只 2 行
    assert out.startswith("2\tline2\n3\tline3\n")


async def test_read_offset_beyond_eof(tmp_project):
    r = Read()
    p = tmp_project / "multi.txt"
    p.write_text("a\nb\n", encoding="utf-8")
    out = await r.execute(file_path=str(p), offset=10)
    assert out == ""  # 越界：无内容行

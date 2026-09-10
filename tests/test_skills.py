"""Skills 测试。"""
from __future__ import annotations

import asyncio

import agent.skills as skills_mod
from agent.skills import Skill, SkillSpec, _parse_frontmatter, scan_skills, skills_prompt_block


def test_parse_frontmatter(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text(
        "---\nname: test\ndescription: a test skill\n---\nbody content",
        encoding="utf-8",
    )
    fm, body = _parse_frontmatter(p.read_text())
    assert fm["name"] == "test"
    assert fm["description"] == "a test skill"
    assert "body content" in body


def test_parse_frontmatter_none(tmp_path):
    fm, body = _parse_frontmatter("just content no frontmatter")
    assert fm == {}
    assert "just content" in body


def test_scan_skills(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: does thing\n---\n",
        encoding="utf-8",
    )
    skills = scan_skills(root=str(tmp_path))
    assert len(skills) == 1
    assert skills[0].name == "my-skill"
    assert skills[0].description == "does thing"


def test_scan_skills_empty(tmp_path):
    assert scan_skills(root=str(tmp_path)) == []


def test_skills_prompt_block():
    skills = [SkillSpec("a", "desc a", "path/a"), SkillSpec("b", "desc b", "path/b")]
    block = skills_prompt_block(skills)
    assert "a" in block and "desc a" in block
    assert "b" in block


def test_skills_prompt_block_empty():
    assert skills_prompt_block([]) == ""


def test_skill_tool_loads_body(tmp_path, monkeypatch):
    d = tmp_path / "guide"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: guide\ndescription: g\n---\nGUIDE BODY", encoding="utf-8"
    )
    monkeypatch.setattr(skills_mod, "scan_skills", lambda: scan_skills(str(tmp_path)))
    out = asyncio.run(Skill().execute("guide"))
    assert "GUIDE BODY" in out


def test_skill_tool_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_mod, "scan_skills", lambda: scan_skills(str(tmp_path)))
    out = asyncio.run(Skill().execute("nope"))
    assert "not found" in out
    assert "Available" in out

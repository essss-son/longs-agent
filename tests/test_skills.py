"""Skills 测试。"""
from __future__ import annotations

from agent.skills import Skill, _parse_frontmatter, scan_skills, skills_prompt_block


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
    skills = [Skill("a", "desc a", "path/a"), Skill("b", "desc b", "path/b")]
    block = skills_prompt_block(skills)
    assert "a" in block and "desc a" in block
    assert "b" in block


def test_skills_prompt_block_empty():
    assert skills_prompt_block([]) == ""

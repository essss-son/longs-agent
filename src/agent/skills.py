"""Skills（D13）。

scan .agent/skills/*/SKILL.md frontmatter，注入 name+description+path（渐进式披露，
省 token）。模型判断相关时用 Read 加载完整内容执行。
"""
from __future__ import annotations

from pathlib import Path


class Skill:
    def __init__(self, name: str, description: str, path: str, frontmatter: dict | None = None):
        self.name = name
        self.description = description
        self.path = path
        self.frontmatter = frontmatter or {}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 --- 之间 frontmatter（简化 YAML，不依赖 pyyaml，手解析 key:value）。"""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            block = text[3:end].strip()
            fm: dict = {}
            for line in block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip()
            body = text[end + 3 :].strip()
            return fm, body
    return {}, text


def scan_skills(root: str = ".agent/skills") -> list[Skill]:
    r = Path(root)
    if not r.exists():
        return []
    skills: list[Skill] = []
    for skill_md in r.rglob("SKILL.md"):
        try:
            text = skill_md.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, _ = _parse_frontmatter(text)
        name = fm.get("name", skill_md.parent.name)
        desc = fm.get("description", "")
        skills.append(Skill(name=name, description=desc, path=str(skill_md), frontmatter=fm))
    return skills


def skills_prompt_block(skills: list[Skill]) -> str:
    """只注入 name+description+path（渐进式披露，省 token）。"""
    if not skills:
        return ""
    lines = ["# Available skills (use Read to load full content when relevant):"]
    for s in skills:
        lines.append(f"- {s.name}: {s.description} ({s.path})")
    return "\n".join(lines)

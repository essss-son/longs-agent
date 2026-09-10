"""Skills（D13）。

scan .agent/skills/*/SKILL.md frontmatter，注入 name+description（渐进式披露，
省 token）。模型判断相关时用 Skill 工具加载完整内容执行。
"""
from __future__ import annotations

from pathlib import Path

from .tools import Tool


class SkillSpec:
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


def scan_skills(root: str = ".agent/skills") -> list[SkillSpec]:
    r = Path(root)
    if not r.exists():
        return []
    skills: list[SkillSpec] = []
    for skill_md in r.rglob("SKILL.md"):
        try:
            text = skill_md.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, _ = _parse_frontmatter(text)
        name = fm.get("name", skill_md.parent.name)
        desc = fm.get("description", "")
        skills.append(SkillSpec(name=name, description=desc, path=str(skill_md), frontmatter=fm))
    return skills


def skills_prompt_block(skills: list[SkillSpec]) -> str:
    """只注入 name+description（渐进式披露，省 token）。"""
    if not skills:
        return ""
    lines = ["# Available skills (use the Skill tool with the name to load full content):"]
    for s in skills:
        lines.append(f"- {s.name}: {s.description}")
    return "\n".join(lines)


class Skill(Tool):
    """加载 skill 完整正文的工具：传 skill 名，返回 SKILL.md body（剥 frontmatter）。"""

    name = "Skill"
    description = "按名字加载一个 skill 的完整正文。传入 skill 名，返回 SKILL.md 正文（不含 frontmatter）。"
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "skill 名（frontmatter 的 name）"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    read_only = True

    async def execute(self, name: str, **_: object) -> str:
        skills = scan_skills()
        for s in skills:
            if s.name == name:
                try:
                    text = Path(s.path).read_text(encoding="utf-8")
                except OSError:
                    return f"[error: cannot read skill '{name}']"
                _, body = _parse_frontmatter(text)
                return body
        avail = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        return f"[error: skill '{name}' not found]\nAvailable skills:\n{avail}"

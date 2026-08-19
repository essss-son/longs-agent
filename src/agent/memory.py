"""记忆 + system prompt 装配（D13 补全）。

system prompt 结构（顺序）：
1. 身份 + 工作模式（始终置顶，模型知道自己是 longs-agent、何时进 plan、何时用 TodoWrite）
2. AGENT.md 全文（项目级记忆）
3. 工具使用指导（何时用 Read vs Grep、Edit 失败回喂策略、Bash 默认 ask）
4. 当前 todos（任务进度，loop 每轮刷新）
5. skills 清单（渐进式披露）
"""
from __future__ import annotations

from pathlib import Path


def load_agent_md(root: str = ".") -> str | None:
    """加载项目根 AGENT.md 全文，注入 system prompt。"""
    p = Path(root) / "AGENT.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def memory_save(content: str, root: str = ".agent/notes.md") -> str:
    """追加记忆到 notes.md（简易）。"""
    p = Path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    return f"saved to {p}"


IDENTITY_BLOCK = """\
You are longs-agent, an async code agent CLI (self-built, Claude Code-style).

## Working mode
- Default NORMAL mode: read-only tools auto-allow; Write/Edit/Bash require user approval (ask). Use TodoWrite for multi-step tasks.
- Call EnterPlanMode to switch into plan mode when: the task is complex or spans multiple files; requirements are ambiguous with multiple approaches; changes carry destructive risk; or you must explore the code before deciding how to proceed.
- In plan mode, explore read-only (Read/Glob/Grep), then submit a plan via ExitPlanMode for approval.
- In AUTO mode all non-disaster commands run without asking (hard deny still blocks rm -rf /).

## Tool discipline
- Read files before editing (avoid stale cache).
- Edit: exact string replace. On [error: not unique] → set replace_all=true or use larger unique context. On [error: not found] → re-Read the file (it may have changed). Never generate diffs.
- Bash output is truncated; if you need more, narrow the command.
- Prefer Grep/Glob to locate, Read to inspect, Edit to change.

## Robustness
- Tool failures are fed back as text ([error: ...] / [denied: ...]) — read them and self-correct.
- tool_call/tool_result pairs must stay paired; compaction preserves this.

## Memory archive
- Compacted content is swapped out, not lost: old tool outputs show a marker like `[elided N chars | mem_id=t_0007 | ...]`. Call MemoryRead("t_0007") to restore the full original when you need exact details.
- Use MemorySearch to find earlier swapped-out content (paths, errors, numbers a user mentioned "before"/"刚才"); then MemoryRead the mem_id it returns.
"""


def _tools_block(registry) -> str:
    """工具清单 + 使用指导。registry 为 ToolRegistry。"""
    lines = ["## Available tools"]
    for t in registry.all():
        lines.append(f"- {t.name}{' (read-only)' if t.read_only else ''}: {t.description}")
    return "\n".join(lines)


def _todos_block(todos: list[dict] | None) -> str:
    """当前 todos 注入（模型知任务进度）。"""
    if not todos:
        return ""
    lines = ["## Current todos"]
    marks = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
    for t in todos:
        status = t.get("status", "pending")
        mark = marks.get(status, "[ ]")
        label = t.get("active_form") or t.get("content", "")
        lines.append(f"{mark} {label}")
    return "\n".join(lines)


def build_system_prompt(
    agent_md: str | None = None,
    skills_block: str = "",
    registry=None,
    todos: list[dict] | None = None,
) -> str:
    """装配 system prompt：身份+工作模式 → AGENT.md → 工具 → todos → skills。"""
    parts: list[str] = [IDENTITY_BLOCK]
    if agent_md:
        parts.append(f"## Project memory (AGENT.md)\n{agent_md}")
    if registry is not None:
        parts.append(_tools_block(registry))
    todos_b = _todos_block(todos)
    if todos_b:
        parts.append(todos_b)
    if skills_block:
        parts.append(skills_block)
    return "\n\n".join(parts)

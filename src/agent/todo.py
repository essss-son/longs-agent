"""Todo 三态 + TodoStore + TodoWrite 工具（D6）。

三态：pending / in_progress / completed。
约束：同时只允许一个 in_progress，多余自动降为 pending（TodoStore.set/update 保证）。
TodoStore 落 todo.json（原子写，resume 恢复，compaction 不动它）。
TodoWrite 工具供模型管理 todo（plan 批准后落地，D4/D7 plan→todos）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .tools import Tool


class TodoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Todo:
    content: str
    status: TodoStatus = TodoStatus.PENDING
    active_form: str = ""  # 进行时标签，渲染用

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "status": self.status.value,
            "active_form": self.active_form or self.content,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Todo":
        return cls(
            content=d["content"],
            status=TodoStatus(d.get("status", "pending")),
            active_form=d.get("active_form", d["content"]),
        )


class TodoStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.todos: list[Todo] = []
        self.load()

    def set(self, todos: list[Todo]) -> None:
        """整体设置。约束：同时只一个 in_progress，多余降 pending。"""
        seen = False
        for t in todos:
            if t.status == TodoStatus.IN_PROGRESS:
                if seen:
                    t.status = TodoStatus.PENDING
                seen = True
        self.todos = todos
        self.save()

    def update(self, idx: int, status: TodoStatus) -> None:
        if not (0 <= idx < len(self.todos)):
            raise IndexError(f"todo index {idx} out of range")
        # 设为 in_progress 时，其余 in_progress 降 pending
        if status == TodoStatus.IN_PROGRESS:
            for t in self.todos:
                if t.status == TodoStatus.IN_PROGRESS:
                    t.status = TodoStatus.PENDING
        self.todos[idx].status = status
        self.save()

    def all(self) -> list[Todo]:
        return list(self.todos)

    def save(self) -> None:
        """原子写 todo.json（tmp + os.replace）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in self.todos], f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def load(self) -> None:
        if not self.path.exists():
            self.todos = []
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.todos = [Todo.from_dict(d) for d in data]
        except (json.JSONDecodeError, OSError):
            self.todos = []


class TodoWrite(Tool):
    name = "TodoWrite"
    description = (
        "管理任务清单。todos 为数组，每项含 content/active_form/status"
        "(pending/in_progress/completed)。同时只允许一个 in_progress。"
        "用于多步任务跟踪。"
    )
    schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "active_form": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        "required": ["todos"],
        "additionalProperties": False,
    }
    read_only = False

    def __init__(self, store: TodoStore):
        self.store = store

    async def execute(self, todos: list[dict], **_: Any) -> str:
        parsed = [
            Todo(
                content=d["content"],
                status=TodoStatus(d.get("status", "pending")),
                active_form=d.get("active_form", d["content"]),
            )
            for d in todos
        ]
        self.store.set(parsed)
        return f"set {len(parsed)} todos"

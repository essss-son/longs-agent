"""会话持久化（D5：trace + meta + todo 原子读写）。

目录：.agent/sessions/<id>/{messages.jsonl, trace.jsonl, todo.json, meta.json}
容错读 JSONL：末行半截（Ctrl+C 中断）静默跳过；中间坏行跳过。
meta.json / todo.json 原子写（tmp + os.replace），防 Ctrl+C 腐蚀。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from .messages import Message


def new_session_id() -> str:
    """uuid4 hex 前 8 位。created_at 在 meta.json，sid 不含时间。"""
    return uuid.uuid4().hex[:8]


def read_jsonl_tolerant(path: Path) -> list[dict]:
    """容错读 JSONL：末行无换行符（半截，Ctrl+C 中断 write）静默跳过；中间坏行跳过。"""
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 半截行 / 坏行：静默跳过
    return out


class SessionStore:
    def __init__(self, root: str | Path = ".agent/sessions", sid: str | None = None):
        self.sid = sid or new_session_id()
        self.root = Path(root)
        self.dir = self.root / self.sid
        self.dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0  # trace 事件序号（实例内自增）

    @property
    def messages_path(self) -> Path:
        return self.dir / "messages.jsonl"

    @property
    def trace_path(self) -> Path:
        return self.dir / "trace.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def todo_path(self) -> Path:
        return self.dir / "todo.json"

    def append_message(self, m: Message) -> None:
        # 一次性生成完整 JSON 字符串再单次 write + flush，降低 Ctrl+C 半截概率
        line = json.dumps(m.to_dict(), ensure_ascii=False)
        with open(self.messages_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def read_messages(self) -> list[Message]:
        return [Message.from_dict(d) for d in read_jsonl_tolerant(self.messages_path)]

    def append_trace(self, event: dict) -> None:
        """写 trace.jsonl 一行。自动加 seq + ts。"""
        self._seq += 1
        line_obj = {"seq": self._seq, "ts": datetime.now().isoformat(), **event}
        line = json.dumps(line_obj, ensure_ascii=False)
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def read_trace(self) -> list[dict]:
        return read_jsonl_tolerant(self.trace_path)

    def write_meta(self, meta: dict) -> None:
        """原子写 meta.json：写 .tmp → os.replace。防 Ctrl+C 腐蚀整个 meta。"""
        tmp = self.meta_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.flush()
        os.replace(tmp, self.meta_path)

    def read_meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def write_todos(self, todos: list[dict]) -> None:
        """原子写 todo.json。D6 接 TodoStore 三态。"""
        tmp = self.todo_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(todos, f, ensure_ascii=False, indent=2)
            f.flush()
        os.replace(tmp, self.todo_path)

    def read_todos(self) -> list[dict]:
        if not self.todo_path.exists():
            return []
        try:
            with open(self.todo_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def list_sessions(root: str | Path = ".agent/sessions") -> list[str]:
        """列所有会话 sid，最近修改在前（按 mtime）。"""
        r = Path(root)
        if not r.exists():
            return []
        sids = [(d.name, d.stat().st_mtime) for d in r.iterdir() if d.is_dir()]
        sids.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sids]

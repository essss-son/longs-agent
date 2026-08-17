"""会话持久化（最小版）。

D1 只实现 messages.jsonl 的 append + 容错 read。trace/meta/todo 留 D5。
目录：.agent/sessions/<id>/messages.jsonl
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from .messages import Message


def new_session_id() -> str:
    """uuid4 hex 前 8 位。created_at 在 meta.json（D5），sid 不含时间。"""
    return uuid.uuid4().hex[:8]


def _read_jsonl_tolerant(path: Path) -> list[dict]:
    """容错读 JSONL：末行无换行符（半截，Ctrl+C 中断 write）静默跳过；中间坏行跳过。

    D5 会在此加 trace 警告区分半截行与中间坏行。
    """
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
        self.dir = Path(root) / self.sid
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def messages_path(self) -> Path:
        return self.dir / "messages.jsonl"

    def append_message(self, m: Message) -> None:
        # 一次性生成完整 JSON 字符串再单次 write + flush，降低 Ctrl+C 半截概率
        line = json.dumps(m.to_dict(), ensure_ascii=False)
        with open(self.messages_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def read_messages(self) -> list[Message]:
        return [Message.from_dict(d) for d in _read_jsonl_tolerant(self.messages_path)]

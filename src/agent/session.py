"""会话持久化（D5：trace + meta + todo 原子读写）。

目录：.agent/sessions/<id>/{messages.jsonl, trace.jsonl, todo.json, meta.json}
容错读 JSONL：末行半截（Ctrl+C 中断）静默跳过；中间坏行跳过。
meta.json / todo.json 原子写（tmp + os.replace），防 Ctrl+C 腐蚀。
"""
from __future__ import annotations

import json
import os
import shutil
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
        self._seq = self._restore_seq()  # 恢复 trace 最大 seq（resume 续号，防与旧快照冲突）

    def _restore_seq(self) -> int:
        """恢复 trace 事件序号：取 trace.jsonl 已有最大 seq。

        _seq 实例内自增，但 resume 新建 SessionStore 会归零，导致新事件 seq 与磁盘
        历史 seq 冲突（快照文件撞名、undo/rewind 按 max(seq) 判断错乱）。此处从
        trace.jsonl 读回最大 seq 续号。
        """
        max_seq = 0
        for e in read_jsonl_tolerant(self.trace_path):
            if isinstance(e.get("seq"), int) and e["seq"] > max_seq:
                max_seq = e["seq"]
        return max_seq

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

    @property
    def file_hashes_path(self) -> Path:
        return self.dir / "file_hashes.json"

    @property
    def snapshot_dir(self) -> Path:
        return self.dir / "snapshots"

    def append_message(self, m: Message) -> None:
        # 一次性生成完整 JSON 字符串再单次 write + flush，降低 Ctrl+C 半截概率
        line = json.dumps(m.to_dict(), ensure_ascii=False)
        with open(self.messages_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def read_messages(self) -> list[Message]:
        return [Message.from_dict(d) for d in read_jsonl_tolerant(self.messages_path)]

    def append_trace(self, event: dict) -> int:
        """写 trace.jsonl 一行。自动加 seq + ts。返回本次 seq。"""
        self._seq += 1
        line_obj = {"seq": self._seq, "ts": datetime.now().isoformat(), **event}
        line = json.dumps(line_obj, ensure_ascii=False)
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
        return self._seq

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

    def get_name(self) -> str:
        """读会话显示名（meta.json 的 name）。sid 目录名是物理 ID，不变。"""
        return self.read_meta().get("name", "")

    def set_name(self, name: str) -> None:
        """设置会话显示名（meta.json 的 name，原子写）。"""
        meta = self.read_meta()
        meta["name"] = name
        self.write_meta(meta)

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

    def read_file_hashes(self) -> dict[str, str]:
        """读乐观锁状态（文件路径 → 内容 hash）。resume 时恢复锁。"""
        if not self.file_hashes_path.exists():
            return {}
        try:
            with open(self.file_hashes_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def save_file_hashes(self, d: dict[str, str]) -> None:
        """原子写 file_hashes.json（乐观锁状态持久化）。"""
        tmp = self.file_hashes_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
            f.flush()
        os.replace(tmp, self.file_hashes_path)

    @staticmethod
    def list_sessions(root: str | Path = ".agent/sessions") -> list[str]:
        """列所有会话 sid，最近修改在前（按 mtime）。"""
        r = Path(root)
        if not r.exists():
            return []
        sids = [(d.name, d.stat().st_mtime) for d in r.iterdir() if d.is_dir()]
        sids.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sids]

    # ---- 文件级 Checkpoint（D：三线回滚 = 文件 + todo + messages 按 seq 对齐） ----

    @staticmethod
    def _safe_name(file_path: str) -> str:
        """快照文件名编码：/ 换 _，配合 seq 前缀保证唯一。恢复时用 index 里的原路径重构。"""
        return file_path.replace("/", "_")

    def _read_snapshot_index(self) -> list[dict]:
        p = self.snapshot_dir / "index.json"
        if not p.exists():
            return []
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _write_snapshot_index(self, idx: list[dict]) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        p = self.snapshot_dir / "index.json"
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    def save_file_snapshot(self, seq: int, file_path: str, assistant_index: int, user_index: int = 0) -> None:
        """Write/Edit 执行前快照目标文件。assistant_index 是发起本次写操作的
        assistant 消息在 messages.jsonl 的索引（回滚时截断边界）；
        user_index 是触发本次写的那条 user 消息索引（rewind 按用户消息粒度回退）。"""
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        dst = self.snapshot_dir / f"{seq}_{self._safe_name(file_path)}"
        was_absent = not os.path.exists(file_path)
        if was_absent:
            Path(f"{dst}.absent").touch()  # 新文件：标记"原本不存在"，恢复时删除
        else:
            shutil.copyfile(file_path, dst)
        idx = self._read_snapshot_index()
        idx.append(
            {
                "seq": seq,
                "file_path": file_path,
                "was_absent": was_absent,
                "assistant_index": assistant_index,
                "user_index": user_index,
            }
        )
        self._write_snapshot_index(idx)

    def save_todo_snapshot(self, seq: int) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if self.todo_path.exists():
            shutil.copyfile(self.todo_path, self.snapshot_dir / f"todo.{seq}.json")

    def list_file_snapshots(self) -> list[dict]:
        return self._read_snapshot_index()

    def restore_file_snapshot(self, snap: dict) -> None:
        dst = self.snapshot_dir / f'{snap["seq"]}_{self._safe_name(snap["file_path"])}'
        if snap["was_absent"]:
            if os.path.exists(snap["file_path"]):
                os.remove(snap["file_path"])
        elif dst.exists():
            os.makedirs(os.path.dirname(snap["file_path"]) or ".", exist_ok=True)
            shutil.copyfile(dst, snap["file_path"])

    def restore_todo_snapshot(self, seq: int) -> bool:
        src = self.snapshot_dir / f"todo.{seq}.json"
        if not src.exists():
            return False
        shutil.copyfile(src, self.todo_path)
        return True

    def _delete_snapshot_files(self, snaps: list[dict]) -> None:
        """删除已消费的快照：文件副本（含 .absent 标记）+ todo 快照，并从 index 移除记录。"""
        seqs = {s["seq"] for s in snaps}
        for s in snaps:
            dst = self.snapshot_dir / f'{s["seq"]}_{self._safe_name(s["file_path"])}'
            for p in (dst, Path(f"{dst}.absent")):
                if p.exists():
                    p.unlink()
        for seq in seqs:
            todo_snap = self.snapshot_dir / f"todo.{seq}.json"
            if todo_snap.exists():
                todo_snap.unlink()
        idx = [s for s in self._read_snapshot_index() if s["seq"] not in seqs]
        self._write_snapshot_index(idx)

    def truncate_messages(self, keep_lines: int) -> None:
        """把 messages.jsonl 截断到前 keep_lines 行（回滚三线之一）。"""
        lines = read_jsonl_tolerant(self.messages_path)
        keep = lines[:keep_lines]
        tmp = self.messages_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for d in keep:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        os.replace(tmp, self.messages_path)

    def undo_last_write(self) -> str:
        """回滚最近一次写操作：文件 + todo + messages 三线回到写之前。"""
        idx = self.list_file_snapshots()
        if not idx:
            return "(无可回滚的快照)"
        last_seq = max(s["seq"] for s in idx)
        snaps = [s for s in idx if s["seq"] == last_seq]
        for s in snaps:
            self.restore_file_snapshot(s)
        self.restore_todo_snapshot(last_seq)
        self.truncate_messages(snaps[0]["assistant_index"])
        self._delete_snapshot_files(snaps)  # 回滚后删除已消费快照
        files = ", ".join(s["file_path"] for s in snaps)
        return f"回滚最近一次写操作（seq={last_seq}）：{files}"

    def rewind_to_user(self, user_idx: int) -> str:
        """回退到第 user_idx 条用户消息处理完成后的状态：撤销 user_index > user_idx 的写。

        文件恢复到最早被撤销写（seq 最小）的写前快照；messages 截断到 user_idx 之后
        第一条用户消息之前（即丢掉该消息及其后所有内容，与是否有写无关）。回退后
        删除已消费的快照。
        """
        idx = self.list_file_snapshots()
        undone = [s for s in idx if s.get("user_index", -1) > user_idx]
        # 文件：每个文件恢复到最早被撤销写的写前快照
        if undone:
            earliest: dict[str, dict] = {}
            for s in undone:
                prev = earliest.get(s["file_path"])
                if prev is None or s["seq"] < prev["seq"]:
                    earliest[s["file_path"]] = s
            for s in earliest.values():
                self.restore_file_snapshot(s)
            self.restore_todo_snapshot(min(s["seq"] for s in undone))
        # messages：截断到 user_idx 之后第一条用户消息之前（与是否有写无关）
        next_user = None
        for i, m in enumerate(self.read_messages()):
            if m.role == "user" and i > user_idx:
                next_user = i
                break
        if next_user is not None:
            self.truncate_messages(next_user)
        # 清理已消费的快照
        if undone:
            self._delete_snapshot_files(undone)
            return f"撤销 {len(undone)} 次写操作"
        if next_user is not None:
            return "已回退（无写操作撤销）"
        return "无需回退"

    def list_rewind_targets(self) -> list[dict]:
        """列出所有用户消息（/rewind 交互选择用）。返回 [{idx, preview}]。"""
        out: list[dict] = []
        for i, m in enumerate(self.read_messages()):
            if m.role == "user":
                out.append({"idx": i, "preview": (m.content or "")[:60]})
        return out

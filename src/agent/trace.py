"""结构化追踪（D11+D12）。

D5 已在 loop 埋事件到 trace.jsonl（turn_start/llm_response/tool_call/tool_result/
permission/compaction/error）。TraceStore 提供查询导出。
CLI: agent trace view <id> [--failed] | export <id> -o <path>
REPL: /trace 时间线 / /cost 累计 token。
"""
from __future__ import annotations

from pathlib import Path


class TraceStore:
    def __init__(self, trace_path):
        self.path = Path(trace_path)

    def all(self) -> list[dict]:
        from .session import read_jsonl_tolerant

        return read_jsonl_tolerant(self.path)

    def timeline_view(self) -> str:
        events = self.all()
        if not events:
            return "(no trace events)"
        lines = [f"trace: {self.path.parent.name}，{len(events)} events"]
        for e in events:
            seq = e.get("seq", "?")
            ts = e.get("ts", "")[:19]
            typ = e.get("type", "?")
            summary = self._summarize(typ, e.get("data", {}))
            lines.append(f"  {seq:>3} {ts} {typ:14} {summary}")
        return "\n".join(lines)

    def failed_view(self) -> str:
        """定位最后 error/permission DENY + 前后各 3 步。"""
        events = self.all()
        fail_types = {"error"}
        fails = [
            i
            for i, e in enumerate(events)
            if e.get("type") in fail_types
            or (e.get("type") == "permission" and e.get("data", {}).get("verdict") == "DENY")
        ]
        if not fails:
            return "(no failures)"
        last = fails[-1]
        start = max(0, last - 3)
        end = min(len(events), last + 4)
        lines = [f"last failure context (events {start}-{end - 1}):"]
        for i in range(start, end):
            e = events[i]
            mark = ">>" if i == last else "  "
            lines.append(
                f"{mark} {e.get('seq', '?'):>3} {e.get('ts', '')[:19]} "
                f"{e.get('type', '?'):14} {self._summarize(e.get('type', '?'), e.get('data', {}))}"
            )
        return "\n".join(lines)

    def cost(self) -> dict:
        """累计 prompt/completion/total tokens（从 llm_response 事件）。"""
        prompt = completion = total = 0
        for e in self.all():
            if e.get("type") == "llm_response":
                usage = e.get("data", {}).get("usage", {})
                prompt += usage.get("prompt_tokens", 0)
                completion += usage.get("completion_tokens", 0)
                total += usage.get("total_tokens", 0)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    def export_md(self) -> str:
        lines = [f"# Trace export: {self.path.parent.name}", ""]
        lines.append(self.timeline_view())
        lines.append("")
        lines.append("## Failures")
        lines.append(self.failed_view())
        lines.append("")
        cost = self.cost()
        lines.append(f"## Cost: prompt={cost['prompt_tokens']} completion={cost['completion_tokens']} total={cost['total_tokens']}")
        return "\n".join(lines)

    def _summarize(self, typ: str, data: dict) -> str:
        if typ == "turn_start":
            return f"input: {str(data.get('user_input', ''))[:40]}"
        if typ == "llm_response":
            return f"model={data.get('model', '')} tokens={data.get('usage', {}).get('total_tokens', 0)} finish={data.get('finish', '')}"
        if typ == "tool_call":
            return f"{data.get('name', '')} args={str(data.get('arguments', {}))[:40]}"
        if typ == "tool_result":
            return f"preview: {str(data.get('content_preview', ''))[:40]}"
        if typ == "permission":
            return f"{data.get('verdict', '')} {data.get('reason', '')}"
        if typ == "compaction":
            return f"remaining={data.get('remaining', '')}"
        if typ == "error":
            return f"tool={data.get('tool', '')} {str(data.get('error', ''))[:40]}"
        return str(data)[:60]

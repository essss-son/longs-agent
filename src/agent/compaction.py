"""Compaction（D9+D10：两策略 + 配对边界 + token 检查 + 滞回）。

触发：used > 0.8 * (context_window - output_reserve)（effective，预留输出 token；工具 schema
token 必须算入，最常被遗忘 → 阈值偏小、触发滞后）。
配对边界不拆：find_pair_units 把 assistant(tool_calls)+其后匹配的 tool 消息当一个 unit，
整体保留或整体处理——避免拆配对导致 tool_call_id 悬空。
工具结果的超长截断已由 envelope（dispatch 阶段）统一承担，这里只做历史 unit 级压缩。
两策略（换出而非丢弃：原始内容先进 archive.jsonl，context 留 mem_id 指针，MemoryRead 可取回）：
- LLMSummary：旧 unit 调 provider 摘要，替换为单条 user "[summary: ...]"（边界对齐 unit）
- SlidingWindow：system + 最近 N units（边界对齐）
滞回：compact 后 _just_compacted=True，跳过下一次 should_compact 检查（让消息增长一轮），
之后自动解除恢复正常判断——避免"压缩一次后永久失效"（否则长对话 context 会撑爆）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .messages import Message
from .utils import estimate_messages_tokens


def find_pair_units(messages: list[Message]) -> list[tuple[int, int]]:
    """返回 (start, end) 索引列表，每单元整体保留或整体处理。

    unit = 一条 assistant(带 tool_calls) + 其后所有匹配 tool_call_id 的 tool 消息。
    无 tool_calls 的消息各自独立成 unit。
    """
    units: list[tuple[int, int]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "assistant" and m.tool_calls:
            start = i
            ids = {tc.id for tc in m.tool_calls}
            j = i + 1
            while j < len(messages) and messages[j].role == "tool" and messages[j].tool_call_id in ids:
                j += 1
            units.append((start, j - 1))
            i = j
        else:
            units.append((i, i))
            i += 1
    return units


def _serialize_for_summary(messages: list[Message]) -> str:
    parts: list[str] = []
    for m in messages:
        content = m.content or ""
        if m.role == "assistant" and m.tool_calls:
            tcs = ", ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls)
            parts.append(f"assistant: {content} [called: {tcs}]")
        elif m.role == "tool":
            parts.append(f"tool({m.name}): {content[:200]}")
        else:
            parts.append(f"{m.role}: {content}")
    return "\n".join(parts)


def _ensure_tool_archived(archive, unit_messages: list[Message]) -> None:
    """换出 unit 前补归档：对 unit 里每条 tool 消息，若全文未归档则补一份 kind=elision。

    堵「错位带」丢失——elision 保留窗（按 tool 条数）与 summary/window 保留窗（按 unit 数）
    不重合时，某条长 tool 可能落在 elision 保留区（未归档全文）却落在 summary 换出区，
    被序列化时 content[:200] 截断、超出部分无全文副本。本函数在换出前兜底补全文。
    用 tool_call_id 防重复：已归档的跳过。
    """
    for m in unit_messages:
        if m.role != "tool":
            continue
        if archive.has_tool_call_id(m.tool_call_id or ""):
            continue
        archive.archive(
            m.content or "",
            kind="elision",
            tool_name=m.name or "",
            tool_call_id=m.tool_call_id or "",
        )


class CompactionStrategy(ABC):
    @abstractmethod
    async def compact(self, messages: list[Message]) -> list[Message]:
        ...

class LLMSummary(CompactionStrategy):
    """旧 unit 归档后调 provider 摘要，替换为单条 user "[summary]"。边界对齐 unit（不拆配对）。

    摘要是"目录"不是"内容"：原始 units 先进 archive，摘要只保留关键事实 + 标识符锚点
    （文件路径/函数名/错误码/ID 原样保留——这些是后续检索的锚，最易被摘要吞掉）。
    """

    def __init__(self, provider, keep_recent_units: int = 4, archive=None, usage_sink=None):
        self.provider = provider
        self.keep_recent_units = keep_recent_units
        self.archive = archive
        self.usage_sink = usage_sink  # 回调：摘要请求结束后上报 usage，补 /cost 缺口

    async def compact(self, messages: list[Message]) -> list[Message]:
        units = find_pair_units(messages)
        if len(units) <= self.keep_recent_units:
            return messages
        recent_start = units[-self.keep_recent_units][0]
        old_msgs = messages[:recent_start]
        recent_msgs = messages[recent_start:]
        if not old_msgs:
            return messages
        # 原始 units 先归档（档案是 source of truth，摘要是目录）
        if self.archive is not None:
            for s, e in units[: -self.keep_recent_units]:
                unit_msgs = messages[s : e + 1]
                _ensure_tool_archived(self.archive, unit_msgs)  # 换出前补 tool 全文
                self.archive.archive(_serialize_for_summary(unit_msgs), kind="summary")
        summary_prompt = [
            Message(
                "system",
                "Summarize the conversation into the following FIVE sections, keeping this EXACT structure:\n"
                "\n"
                "## 📌 Archived Session Summary\n"
                "*(Contains context from [Start Time] to [Cutoff Time])*\n"
                "\n"
                "### 🎯 Objectives & Status\n"
                "* **Original Goal**: [what the user originally wanted to do]\n"
                "\n"
                "### 🏗️ Technical Context (Static)\n"
                "* **Stack**: [language, framework, versions]\n"
                "* **Environment**: [OS, shell, key env vars]\n"
                "\n"
                "### ✅ Completed Milestones (The \"Done\" Pile)\n"
                "* [✓] [completed task] - [brief result]\n"
                "\n"
                "### 🧠 Key Insights & Decisions (Persistent Memory)\n"
                "* **Decisions**: [key technical choices or abandoned approaches]\n"
                "* **Learnings**: [special configs, API formats, gotchas]\n"
                "* **User Preferences**: [habits the user emphasized]\n"
                "\n"
                "### 📂 File System State (Snapshot)\n"
                "*(Modified files in this archive segment)*\n"
                "* `path/to/file`: what changed.\n"
                "\n"
                "CRITICAL — preserve these identifiers VERBATIM (do not translate, paraphrase, "
                "or abbreviate): file paths (e.g. src/agent/loop.py), function/class names "
                "(find_pair_units), error codes/messages, IDs and numbers (t_0007, 110 passed), URLs.\n"
                "These are retrieval anchors: rewording 'src/agent/loop.py' as 'a file' makes it "
                "unsearchable later. Prefer verbose-but-exact over concise-but-vague.",
            ),
            Message("user", _serialize_for_summary(old_msgs)),
        ]
        try:
            resp = await self.provider.chat(summary_prompt, tools=None)
            summary = resp.content or "(empty summary)"
            if self.usage_sink is not None:
                self.usage_sink(resp.usage)  # 摘要请求的 token 也要进 /cost
        except Exception as e:
            summary = f"(summary failed: {e})"
        marker = "[summary of earlier turns]"
        if self.archive is not None:
            marker += "（原始内容已归档：MemorySearch 检索 / MemoryRead 取回）"
        summary_msg = Message("user", f"{marker}\n{summary}")
        # 若第一条 recent 也是 user，合并避免连续同角色（Claude API 拒绝）
        if recent_msgs and recent_msgs[0].role == "user":
            recent_msgs[0].content = summary_msg.content + "\n" + (recent_msgs[0].content or "")
            return recent_msgs
        return [summary_msg] + recent_msgs


class SlidingWindow(CompactionStrategy):
    """system + 最近 N units（边界对齐 unit）。换出的旧 unit 按 unit 归档，留指针消息。"""

    def __init__(self, keep_recent_units: int = 6, archive=None):
        self.keep_recent_units = keep_recent_units
        self.archive = archive

    async def compact(self, messages: list[Message]) -> list[Message]:
        units = find_pair_units(messages)
        if len(units) <= self.keep_recent_units:
            return messages
        old_units = units[: -self.keep_recent_units]
        # 换出的旧 unit（非 system）按 unit 归档：检索回来必是完整 call+result 配对
        if self.archive is not None:
            swapped = 0
            for s, e in old_units:
                if messages[s].role != "system":
                    unit_msgs = messages[s : e + 1]
                    _ensure_tool_archived(self.archive, unit_msgs)  # 换出前补 tool 全文
                    self.archive.archive(
                        _serialize_for_summary(unit_msgs), kind="window"
                    )
                    swapped += e - s + 1
        out = [m for m in messages if m.role == "system"]
        if self.archive is not None and swapped:
            out.append(
                Message(
                    "user",
                    f"[{swapped} 条早期消息已换出到会话档案 | 用 MemorySearch 检索 / MemoryRead 取回]",
                )
            )
        recent_start = units[-self.keep_recent_units][0]
        recent = messages[recent_start:]
        # 若指针消息与第一条 recent 都是 user，合并避免连续同角色（Claude API 拒绝）
        if recent and recent[0].role == "user" and out and out[-1].role == "user":
            recent[0].content = (out[-1].content or "") + "\n" + (recent[0].content or "")
            out.pop()
        out.extend(recent)
        return out


class Compactor:
    def __init__(
        self,
        provider,
        token_fn=estimate_messages_tokens,
        context_window: int = 32768,
        output_reserve: int = 8192,
        strategies=None,
        archive=None,
    ):
        self.provider = provider
        self.token_fn = token_fn
        self.context_window = context_window
        self.output_reserve = output_reserve
        self.archive = archive  # ArchiveStore | None：压缩换出的内容归档处
        self._summary_prompt = 0  # 压缩期 LLMSummary 摘要请求累计的 token（补 /cost 缺口）
        self._summary_completion = 0
        self.strategies = strategies or [
            LLMSummary(provider, archive=archive, usage_sink=self._add_summary_usage),
            SlidingWindow(archive=archive),
        ]
        self._just_compacted = False  # 滞回

    def _add_summary_usage(self, usage) -> None:
        """LLMSummary 摘要请求结束后回调：累加 token。"""
        self._summary_prompt += usage.prompt_tokens
        self._summary_completion += usage.completion_tokens

    def consume_summary_usage(self) -> dict:
        """返回并清零压缩期累计的摘要 token（loop 写 trace 用）。"""
        d = {
            "prompt_tokens": self._summary_prompt,
            "completion_tokens": self._summary_completion,
            "total_tokens": self._summary_prompt + self._summary_completion,
        }
        self._summary_prompt = 0
        self._summary_completion = 0
        return d

    def reset(self) -> None:
        """重置滞回标记：新一轮对话开始时重新允许压缩。"""
        self._just_compacted = False

    def effective_window(self) -> int:
        return max(1024, self.context_window - self.output_reserve)  # 预留输出 token

    def should_compact(self, messages, tools=None) -> bool:
        if self._just_compacted:
            self._just_compacted = False  # 滞回：压缩后跳过本次检查，下次恢复正常判断
            return False
        used = self.token_fn(messages, tools)
        return used > 0.8 * self.effective_window()

    async def compact(self, messages, tools=None) -> list[Message]:
        target = 0.1 * self.effective_window()
        result = messages
        for strategy in self.strategies:
            if self.token_fn(result, tools) <= target:
                break
            try:
                result = await strategy.compact(result)
            except Exception:
                continue  # 策略失败跳过，用下一个
        self._just_compacted = True
        return result

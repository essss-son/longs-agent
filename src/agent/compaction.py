"""Compaction（D9+D10：单策略滚动摘要 + 工具调用台账）。

触发：used > 0.8 * (context_window - output_reserve)（effective）。
配对边界不拆：find_pair_units 把 assistant(tool_calls)+其后匹配的 tool 消息当一个 unit。
换出而非丢弃：每个换出的 tool 输出发一个 t_XXXX（信封大输出已有指针则透传，其余内联
全文），context 里留「滚动摘要 + 早期工具调用台账」——台账是程序生成的 name+args→mem_id
映射，模型据此用 MemoryRead 按需取回原文。非工具内容只进滚动摘要，不单独归档。
台账跨压缩续传：旧摘要消息里的台账块由程序解析回条目滚动保留（不让 LLM 抄写结构化映射）。
滞回：compact 后 _just_compacted=True，跳过下一次 should_compact，让消息增长一轮。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .messages import Message
from .utils import estimate_messages_tokens

MEM_ID_RE = re.compile(r"mem_id=(t_\w+)")
SUMMARY_MARKER = "[summary of earlier turns]"
LEDGER_HEADER = "## 早期工具调用台账"
LEDGER_LINE_RE = re.compile(r"^- (t_\d+) \| (\S+)\s+(.*)$")

SUMMARY_TEMPLATE = (
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
    "unsearchable later. Prefer verbose-but-exact over concise-but-vague."
)


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


def _compact_args(args: dict) -> str:
    """args 序列化为紧凑字符串，统一截断 120 字符（超长加 …）。"""
    if not args:
        return ""
    s = " ".join(json.dumps(args, ensure_ascii=False).split())
    return s[:120] + ("…" if len(s) > 120 else "")


def _extract_mem_id(content: str) -> str | None:
    """从信封头部抽出 t_XXXX（信封消息才有；普通 tool 消息无）。"""
    m = MEM_ID_RE.search(content or "")
    return m.group(1) if m else None


def _ledger_block(entries: list[tuple[str, str, str, str]], limit: int) -> str:
    """台账块：程序生成的 name+args → mem_id 映射，截断到最近 limit 条。"""
    if not entries:
        return ""
    lines = [LEDGER_HEADER]
    for _, mem_id, name, args in entries[-limit:]:
        lines.append(f"- {mem_id} | {name}  {args}")
    return "\n".join(lines)


def _split_old_summary(content: str) -> tuple[str, list[tuple[str, str, str, str]]]:
    """拆旧摘要消息：返回 (摘要文本, 台账条目)。

    台账块由 _ledger_block 程序生成、逐行格式确定，可解析回条目续传——LLM 只见摘要
    文本，永远不接触台账（不让模型抄写 mem_id 映射）。台账块之后若还有文本（上一轮
    合并进来的 user 内容），保留在摘要文本里继续喂 LLM。
    """
    idx = content.find(LEDGER_HEADER)
    if idx < 0:
        return content, []
    head = content[:idx].rstrip()
    lines = content[idx + len(LEDGER_HEADER):].splitlines()
    entries: list[tuple[str, str, str, str]] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue  # header 自身的换行 / 台账行间空行
        m = LEDGER_LINE_RE.match(s)
        if m is None:
            break  # 台账行结束，后面是合并进来的 user 内容
        entries.append(("", m.group(1), m.group(2), m.group(3)))
        i += 1
    tail = "\n".join(lines[i:]).strip()
    if tail:
        head = f"{head}\n{tail}" if head else tail
    return head, entries


def _serialize_unit(unit_msgs: list[Message], mem_by_tc: dict[str, str]) -> str:
    """序列化一个 unit 喂摘要：tool 输出不贴正文，占位 [输出见 t_XXXX]。"""
    parts: list[str] = []
    for m in unit_msgs:
        content = m.content or ""
        if m.role == "assistant" and m.tool_calls:
            tcs = ", ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls)
            parts.append(f"assistant: {content} [called: {tcs}]")
        elif m.role == "tool":
            mem_id = mem_by_tc.get(m.tool_call_id) or _extract_mem_id(content)
            parts.append(f"tool({m.name}): [输出见 {mem_id}]" if mem_id else f"tool({m.name}): {content[:200]}")
        else:
            parts.append(f"{m.role}: {content}")
    return "\n".join(parts)


class CompactionStrategy(ABC):
    @abstractmethod
    async def compact(self, messages: list[Message]) -> list[Message]:
        ...


class RollingSummary(CompactionStrategy):
    """旧 unit 归档为 t_ + 滚动摘要 + 台账（单策略，原 LLMSummary/SlidingWindow 已合并）。

    每个换出的 tool 输出发一个 t_（信封大输出透传已有 mem_id，其余内联全文）；非工具
    内容只喂摘要，不归档。摘要输入是旧 unit 序列化文本（tool 占位），输出替换旧的单条
    summary。台账由程序生成拼在摘要后。
    """

    def __init__(self, provider, keep_recent_units: int = 4, archive=None, usage_sink=None, ledger_limit: int = 40):
        self.provider = provider
        self.keep_recent_units = keep_recent_units
        self.archive = archive
        self.usage_sink = usage_sink  # 回调：摘要请求结束后上报 usage，补 /cost 缺口
        self.ledger_limit = ledger_limit

    async def compact(self, messages: list[Message]) -> list[Message]:
        units = find_pair_units(messages)
        if len(units) <= self.keep_recent_units:
            return messages
        recent_start = units[-self.keep_recent_units][0]
        recent_msgs = messages[recent_start:]
        if not messages[:recent_start]:
            return messages

        ledger: list[tuple[str, str, str, str]] = []
        mem_by_tc: dict[str, str] = {}
        serialized_parts: list[str] = []
        for s, e in units[: -self.keep_recent_units]:
            unit_msgs = messages[s : e + 1]
            # 旧摘要消息：台账条目解析出来续传，摘要文本（去台账块）照常喂 LLM 滚动合并
            if unit_msgs[0].role == "user" and SUMMARY_MARKER in (unit_msgs[0].content or ""):
                body, old_entries = _split_old_summary(unit_msgs[0].content or "")
                ledger.extend(old_entries)
                serialized_parts.append(f"user: {body}")
                continue
            entries = self._elide_unit(unit_msgs)
            ledger.extend(entries)
            for tc_id, mem_id, _, _ in entries:
                mem_by_tc[tc_id] = mem_id
            serialized_parts.append(_serialize_unit(unit_msgs, mem_by_tc))
        # 按 mem_id 去重（resume 双重压缩时，信封透传可能与旧台账撞同一 mem_id）
        seen: set[str] = set()
        unique: list[tuple[str, str, str, str]] = []
        for en in ledger:
            if en[1] not in seen:
                seen.add(en[1])
                unique.append(en)
        ledger = unique

        summary = await self._summarize("\n".join(serialized_parts))

        marker = SUMMARY_MARKER
        if self.archive is not None:
            marker += "（原始内容已归档：用 MemoryRead 按 mem_id 取回）"
        body = f"{marker}\n{summary}"
        ledger_text = _ledger_block(ledger, self.ledger_limit)
        if ledger_text:
            body += "\n\n" + ledger_text
        summary_msg = Message("user", body)
        # 若第一条 recent 也是 user，合并避免连续同角色（Claude API 拒绝）
        if recent_msgs and recent_msgs[0].role == "user":
            recent_msgs[0].content = summary_msg.content + "\n" + (recent_msgs[0].content or "")
            return recent_msgs
        return [summary_msg] + recent_msgs

    def _elide_unit(self, unit_msgs: list[Message]) -> list[tuple[str, str, str, str]]:
        """为 unit 里每个 tool 输出发 t_，返回台账条目 [(tool_call_id, mem_id, name, args)]。

        信封透传：content 带 mem_id=t_ 的不重复归档，直接登记；其余内联全文归档。
        """
        entries: list[tuple[str, str, str, str]] = []
        assistant = next((m for m in unit_msgs if m.role == "assistant" and m.tool_calls), None)
        if assistant is None:
            return entries
        tool_by_id = {m.tool_call_id: m for m in unit_msgs if m.role == "tool"}
        for tc in assistant.tool_calls:
            tmsg = tool_by_id.get(tc.id)
            if tmsg is None:
                continue
            mem_id = _extract_mem_id(tmsg.content or "")
            if mem_id is None and self.archive is not None and not self.archive.has_tool_call_id(tc.id):
                mem_id = self.archive.archive(
                    tmsg.content or "",
                    kind="tool",
                    tool_name=tmsg.name or "",
                    tool_call_id=tc.id,
                )
            if mem_id:
                entries.append((tc.id, mem_id, tc.name, _compact_args(tc.arguments)))
        return entries

    async def _summarize(self, serialized_old: str) -> str:
        summary_prompt = [
            Message("system", SUMMARY_TEMPLATE),
            Message("user", serialized_old),
        ]
        try:
            resp = await self.provider.chat(summary_prompt, tools=None)
            summary = resp.content or "(empty summary)"
            if self.usage_sink is not None:
                self.usage_sink(resp.usage)  # 摘要请求的 token 也要进 /cost
        except Exception as e:
            summary = f"(summary failed: {e})"
        return summary


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
        self._summary_prompt = 0  # 压缩期 RollingSummary 摘要请求累计的 token（补 /cost 缺口）
        self._summary_completion = 0
        self.strategies = strategies or [
            RollingSummary(provider, archive=archive, usage_sink=self._add_summary_usage),
        ]
        self._just_compacted = False  # 滞回

    def _add_summary_usage(self, usage) -> None:
        """RollingSummary 摘要请求结束后回调：累加 token。"""
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

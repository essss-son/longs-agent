"""权限引擎：hard deny 不可覆盖 + 规则链 + 三模式。

hard deny 清单短而致命（代码内置正则，任何模式/配置不可覆盖）。
MANUAL: hard deny → deny → always → 默认 ASK（所有工具都问，读工具也问）。
AUTO:   hard deny → Write/Edit 项目目录外 ASK → 其余放行。
PLAN:   注册表过滤（D7），dispatch 兜底校验 read_only。
always 按 tool_name 粒度，会话内有效（存 meta.json，resume 同会话延续，新会话不继承）。

诚实声明：shell 技巧（eval / base64 解码 / ${VAR:-/}）拦不住，ask + 人类审批才是真正安全网。
简历话术："默认最小权限 + 代码层兜底灾难性命令"，不吹全沙箱。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto


class Mode(Enum):
    MANUAL = auto()  # 原 NORMAL：所有工具调用都需批准
    AUTO = auto()
    PLAN = auto()


class Verdict(Enum):
    ALLOW = auto()
    DENY = auto()
    ASK = auto()


@dataclass
class PermissionConfig:
    # 只读工具默认 allow（不烦扰用户）；写工具默认 ASK
    allow: list[str] = field(default_factory=lambda: ["Read", "Glob", "Grep"])
    deny: list[str] = field(default_factory=list)
    always_grants: set[str] = field(default_factory=set)  # tool_name，会话内有效
    project_root: str = "."  # auto 模式项目目录边界（目录外写操作需确认）


class PermissionEngine:
    # 短而致命的灾难命令（case-insensitive，容忍多空白）
    HARD_DENY_BASH = [
        re.compile(r"\brm\s+-[rfRF]+\s+/(?:\s|$)", re.I),   # rm -rf /（根递归删）
        re.compile(r"\bdd\b.*\bof\s*=\s*/dev/", re.I),      # dd of=/dev/
        re.compile(r"\bmkfs\b", re.I),                      # 格式化
        re.compile(r":\s*\(\s*\)\s*\{", re.I),              # fork bomb :(){
        re.compile(r">\s*/dev/s[dv]", re.I),               # 写磁盘设备
        re.compile(r"\bchmod\s+-R\s+0+\s+/", re.I),         # chmod -R 000 /
    ]
    # 系统敏感写路径
    HARD_DENY_PATH = [
        re.compile(r"^/etc/"),
        re.compile(r"^/boot/"),
        re.compile(r"^/sys/"),
        re.compile(r"^/proc/"),
        re.compile(r"^/dev/"),
        re.compile(r"^/etc/sudoers"),
        re.compile(r"/\.ssh/(?:authorized_keys|id_rsa|id_ed25519)"),
    ]

    def is_hard_denied(self, tc) -> tuple[bool, str]:
        if tc.name == "Bash":
            cmd = tc.arguments.get("command", "")
            for pat in self.HARD_DENY_BASH:
                if pat.search(cmd):
                    return True, f"hard deny: bash matched {pat.pattern!r}"
        if tc.name in ("Write", "Edit"):
            fp = tc.arguments.get("file_path", "")
            for pat in self.HARD_DENY_PATH:
                if pat.search(fp):
                    return True, f"hard deny: path matched {pat.pattern!r}"
        return False, ""

    def check(self, tc, config: PermissionConfig, mode: Mode) -> tuple[Verdict, str]:
        denied, reason = self.is_hard_denied(tc)
        if denied:
            return Verdict.DENY, reason  # 任何模式不可覆盖
        if mode == Mode.AUTO:
            # 项目目录外写操作 → ASK（危险）；其余放行
            if tc.name in ("Write", "Edit"):
                fp = tc.arguments.get("file_path", "")
                if not self._inside_project(fp, config.project_root):
                    return Verdict.ASK, "auto: write outside project"
            return Verdict.ALLOW, "auto mode"
        if mode == Mode.PLAN:
            # 只读 ALLOW，写工具 DENY（registry 已过滤，这里双保险）
            if tc.name in ("Write", "Edit", "Bash"):
                return Verdict.DENY, "plan mode readonly"
            return Verdict.ALLOW, "plan mode readonly"
        # MANUAL：所有工具都 ASK；always_grants 仍生效
        if tc.name in config.deny:
            return Verdict.DENY, "deny rule"
        if tc.name in config.always_grants:
            return Verdict.ALLOW, "always granted"
        return Verdict.ASK, "manual mode ask"

    @staticmethod
    def _inside_project(fp: str, root: str) -> bool:
        from pathlib import Path

        try:
            return Path(fp).resolve().is_relative_to(Path(root).resolve())
        except Exception:
            return False

"""权限引擎 table-driven 测试。"""
from __future__ import annotations

from agent.messages import ToolCall
from agent.permissions import Mode, PermissionConfig, PermissionEngine, Verdict


def _bash(cmd: str) -> ToolCall:
    return ToolCall(id="x", name="Bash", arguments={"command": cmd})


def _write(fp: str) -> ToolCall:
    return ToolCall(id="x", name="Write", arguments={"file_path": fp, "content": "y"})


def test_hard_deny_bash_root_all_modes():
    """rm -rf / 在任何模式（MANUAL/AUTO/PLAN）都必须 hard deny。"""
    eng = PermissionEngine()
    cfg = PermissionConfig()
    tc = _bash("rm -rf /")
    for mode in (Mode.MANUAL, Mode.AUTO, Mode.PLAN):
        v, _ = eng.check(tc, cfg, mode)
        assert v == Verdict.DENY, f"{mode}: rm -rf / must hard-deny"
    # 下面用枚举值避免 NameError（上面 Verdict 未 import 时）


# 重新引入 Verdict（上面 test 用了 Verdict，补 import）
from agent.permissions import Verdict  # noqa: E402


def test_hard_deny_bash_disaster_commands():
    eng = PermissionEngine()
    cfg = PermissionConfig()
    for cmd in [
        "dd if=x of=/dev/sda",
        "mkfs.ext4 /dev/sda",
        ":(){ :|:& };:",
        "> /dev/sda",
        "chmod -R 000 /",
    ]:
        v, _ = eng.check(_bash(cmd), cfg, Mode.AUTO)
        assert v == Verdict.DENY, f"{cmd!r} must hard-deny"


def test_hard_deny_not_triggered_for_safe_rm():
    """rm -rf /tmp/x 不是根，不 hard deny（MANUAL 默认 ASK）。"""
    eng = PermissionEngine()
    cfg = PermissionConfig()
    v, _ = eng.check(_bash("rm -rf /tmp/x"), cfg, Mode.MANUAL)
    assert v == Verdict.ASK


def test_hard_deny_path_system_dirs():
    eng = PermissionEngine()
    cfg = PermissionConfig()
    for fp in [
        "/etc/passwd",
        "/boot/vmlinuz",
        "/etc/sudoers",
        "/Users/x/.ssh/authorized_keys",
    ]:
        v, _ = eng.check(_write(fp), cfg, Mode.AUTO)
        assert v == Verdict.DENY, f"{fp} must hard-deny"


def test_auto_mode_allows_non_disaster():
    eng = PermissionEngine()
    cfg = PermissionConfig()
    v, _ = eng.check(_bash("echo hi"), cfg, Mode.AUTO)
    assert v == Verdict.ALLOW


def test_normal_default_ask_for_write_tool():
    eng = PermissionEngine()
    cfg = PermissionConfig()
    v, _ = eng.check(_write("/tmp/x.txt"), cfg, Mode.MANUAL)
    assert v == Verdict.ASK


def test_manual_asks_read_tool():
    """manual 模式下读工具也要确认。"""
    eng = PermissionEngine()
    cfg = PermissionConfig()
    tc = ToolCall(id="x", name="Read", arguments={"file_path": "/tmp/x"})
    v, _ = eng.check(tc, cfg, Mode.MANUAL)
    assert v == Verdict.ASK


def test_always_grants_overrides_default_ask():
    eng = PermissionEngine()
    cfg = PermissionConfig(always_grants={"Bash"})
    v, _ = eng.check(_bash("echo hi"), cfg, Mode.MANUAL)
    assert v == Verdict.ALLOW


def test_deny_rule_overrides_allow():
    eng = PermissionEngine()
    cfg = PermissionConfig(allow=["Bash"], deny=["Bash"])
    v, _ = eng.check(_bash("echo hi"), cfg, Mode.MANUAL)
    assert v == Verdict.DENY


def test_hard_deny_overrides_auto():
    """AUTO 放行普通命令，但 rm -rf / 仍 hard deny。"""
    eng = PermissionEngine()
    cfg = PermissionConfig()
    v, _ = eng.check(_bash("rm -rf /"), cfg, Mode.AUTO)
    assert v == Verdict.DENY


def test_is_hard_denied_returns_reason():
    eng = PermissionEngine()
    denied, reason = eng.is_hard_denied(_bash("rm -rf /"))
    assert denied
    assert "hard deny" in reason


def test_auto_write_outside_project_asks():
    """auto 模式 Write 项目目录外 → ASK（危险确认）。"""
    eng = PermissionEngine()
    cfg = PermissionConfig(project_root="/tmp/proj")
    v, _ = eng.check(_write("/tmp/outside/x.txt"), cfg, Mode.AUTO)
    assert v == Verdict.ASK


def test_auto_write_inside_project_allows():
    """auto 模式 Write 项目目录内 → ALLOW。"""
    eng = PermissionEngine()
    cfg = PermissionConfig(project_root="/tmp/proj")
    v, _ = eng.check(_write("/tmp/proj/x.txt"), cfg, Mode.AUTO)
    assert v == Verdict.ALLOW

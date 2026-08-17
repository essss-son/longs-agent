"""pytest 共享 fixtures。"""
from __future__ import annotations

import pytest

from agent.provider import FakeProvider


@pytest.fixture
def tmp_project(tmp_path):
    """建一个临时小 repo 供工具测试（D2 起的 Read/Write/Edit 用）。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def make_fake():
    """返回构造器，便于测试时指定剧本。"""
    def _make(script, **kw):
        return FakeProvider(script, **kw)
    return _make

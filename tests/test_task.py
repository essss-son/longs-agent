"""Task 子代理测试：简化子循环 + 白名单 + 模型路由。"""
from __future__ import annotations

from agent.messages import NormalizedResponse, ToolCall
from agent.provider import FakeProvider
from agent.task import Task


async def test_task_returns_subagent_report(tmp_path):
    script = [
        NormalizedResponse(
            tool_calls=[
                ToolCall(id="c1", name="Glob", arguments={"pattern": "*.py", "path": str(tmp_path)})
            ]
        ),
        NormalizedResponse(content="报告：分析完成"),
    ]
    provider = FakeProvider(script)
    task = Task(provider)
    out = await task.execute(description="d", prompt="列出 py 文件")
    assert "报告：分析完成" in out
    assert len(provider.calls) == 2  # 两次 chat（工具调用 + 收尾）


async def test_task_blocks_write_tool():
    seen: dict = {}

    def second(messages, tools):
        joined = "\n".join(m.content or "" for m in messages)
        seen["blocked"] = "not allowed" in joined
        return NormalizedResponse(content="done")

    script = [
        NormalizedResponse(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="Write",
                    arguments={"file_path": "x", "content": "y"},
                )
            ]
        ),
        second,
    ]
    provider = FakeProvider(script)
    task = Task(provider)
    out = await task.execute(description="d", prompt="写文件")
    assert out == "done"
    assert seen["blocked"] is True  # Write 不在白名单，被拒并回喂


async def test_task_light_model_routing():
    light = FakeProvider([NormalizedResponse(content="light done")])
    main = FakeProvider([NormalizedResponse(content="main done")])
    task = Task(main, light_provider=light)
    out = await task.execute(description="d", prompt="p", model="light")
    assert out == "light done"
    assert len(light.calls) == 1
    assert len(main.calls) == 0


async def test_task_light_falls_back_to_main():
    main = FakeProvider([NormalizedResponse(content="main done")])
    task = Task(main)  # 无 light_provider
    out = await task.execute(description="d", prompt="p", model="light")
    assert out == "main done"

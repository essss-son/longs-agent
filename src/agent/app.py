"""CLI bootstrap：装配 Config/Provider/Session，跑单轮对话。

D1 最小版：有 config + api_key 则真实 OpenAICompatibleProvider，否则 FakeProvider demo
（证明装配链路通，不调真实 API）。D2 接 AgentLoop，D3 换 prompt_toolkit REPL（async input）。
"""
from __future__ import annotations

from .config import Config
from .messages import Message, NormalizedResponse
from .provider import FakeProvider, OpenAICompatibleProvider
from .session import SessionStore


def _demo_script() -> list[NormalizedResponse]:
    """无 API key 时的演示剧本，证明 provider→message 装配链路通。"""
    return [
        NormalizedResponse(
            content="（demo）你好！我是 longs-agent。配置 .agent/config.toml 后可用真实模型。"
        )
    ]


async def main() -> None:
    cfg = Config.load()
    model = cfg.get()
    api_key = cfg.api_key()

    if model and api_key:
        provider = OpenAICompatibleProvider(
            base_url=model.base_url,
            api_key=api_key,
            model=model.model,
            context_window=model.context_window,
        )
        print(f"[longs-agent] 模型: {model.model} @ {model.base_url}")
    else:
        provider = FakeProvider(_demo_script())
        print("[longs-agent] 未配置 .agent/config.toml 或缺少 api_key，用 demo 模式。")

    session = SessionStore()
    try:
        user_input = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not user_input:
        return

    messages = [Message("user", user_input)]
    session.append_message(messages[0])

    resp = await provider.chat(messages)
    print(f"assistant> {resp.content}")
    session.append_message(Message("assistant", resp.content))

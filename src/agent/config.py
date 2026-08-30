"""配置加载：模型别名表 + Provider 参数。

读 .agent/config.toml（用 Python 3.12 内置 tomllib，无额外依赖）。
api_key 从环境变量读（api_key_env 指定变量名），不入库。
无 config 时返回空 Config，由 app 决定 fallback（D1 用 FakeProvider demo）。
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    alias: str
    provider: str               # "openai_compatible"（P1 可加 "anthropic"）
    base_url: str
    model: str
    context_window: int
    api_key: str = ""           # 直接在 config.toml 写明文（优先）
    api_key_env: str = ""       # 或从此环境变量读 api_key（fallback）
    max_tokens: int = 8192      # Anthropic 必传；OpenAI-compatible 忽略


@dataclass
class McpServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    url: str = ""               # 非空走 streamable HTTP（云端）；空走 stdio（本地）


@dataclass
class Config:
    models: dict[str, ModelConfig] = field(default_factory=dict)
    mcp_servers: dict[str, McpServerConfig] = field(default_factory=dict)
    default_alias: str = ""

    def get(self, alias: str | None = None) -> ModelConfig | None:
        if not self.models:
            return None
        key = alias or self.default_alias
        return self.models.get(key)

    def api_key(self, alias: str | None = None) -> str | None:
        m = self.get(alias)
        if not m:
            return None
        if m.api_key:            # toml 明文优先
            return m.api_key
        if m.api_key_env:        # 否则 fallback 环境变量
            return os.environ.get(m.api_key_env)
        return None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else Path(".agent/config.toml")
        if not p.exists():
            return cls()
        with open(p, "rb") as f:
            data = tomllib.load(f)
        models: dict[str, ModelConfig] = {}
        mcp_servers: dict[str, McpServerConfig] = {}
        default_alias = ""
        for table_name, tbl in data.items():
            # mcp server 段：[mcp.<name>] 解析成嵌套 data["mcp"]["<name>"]
            if table_name == "mcp" and isinstance(tbl, dict):
                for mname, sub in tbl.items():
                    if isinstance(sub, dict):
                        mcp_servers[mname] = McpServerConfig(
                            name=mname,
                            command=sub.get("command", ""),
                            args=list(sub.get("args", [])),
                            url=sub.get("url", ""),
                        )
                continue
            alias = tbl.get("alias", table_name)
            models[alias] = ModelConfig(
                alias=alias,
                provider=tbl.get("provider", "openai_compatible"),
                base_url=tbl.get("base_url", ""),
                model=tbl.get("model", ""),
                context_window=int(tbl.get("context_window", 32768)),
                api_key=tbl.get("api_key", ""),
                api_key_env=tbl.get("api_key_env", ""),
                max_tokens=int(tbl.get("max_tokens", 8192)),
            )
            if not default_alias:
                default_alias = alias
        return cls(models=models, mcp_servers=mcp_servers, default_alias=default_alias)

# longs-agent

一个 Claude Code 风格的 async code agent CLI，自研实现核心能力并做有意识取舍裁剪。

## 特性

- **多供应商接入**：OpenAI 兼容（DeepSeek / Qwen / Ollama / 本地 vLLM），`/model` 热切换
- **工具调用循环**：Read / Write / Edit / Bash / Glob / Grep + 只读子代理
- **权限三级模式**：NORMAL / AUTO / PLAN，hard deny 不可覆盖
- **上下文压缩**：滚动摘要 + 工具输出归档，换出内容可按 mem_id 取回原文
- **写前 checkpoint**：`/undo` 撤销最近一次写、`/rewind <n>` 按消息粒度回退
- **会话持久化 + resume**、**Todo 任务跟踪**、**Skills + AGENT.md 记忆**、**MCP 工具接入**、**结构化追踪**

## 安装

要求 Python ≥ 3.12。

```bash
conda create -n longs-agent python=3.12 -y
conda activate longs-agent
pip install -e ".[dev,tui,mcp]"
```

可选依赖组：`dev`（pytest）、`tui`（rich 全屏终端界面）、`mcp`（MCP 工具生态）。只装核心：`pip install -e .`。

## 配置

复制模板并填入供应商信息：

```bash
mkdir -p .agent
cp .agent/config.example.toml .agent/config.toml
```

`api_key` 从环境变量读取（不写入文件）：

```bash
export DEEPSEEK_API_KEY=sk-xxx
```

可选配置 `light` 别名作为子代理的轻量模型路由。未找到 `config.toml` 时进入 demo 模式（无需 API key 即可体验交互）。

## 运行

```bash
agent                                  # 启动（有 rich 时进全屏 TUI，否则 REPL）
agent trace view <sid> [--failed]      # 查看会话追踪时间线
agent trace export <sid> -o trace.md   # 导出追踪
```

会话内主要命令：

| 命令 | 作用 |
|---|---|
| `/help` | 命令列表 |
| `/plan` | 进入计划模式（只读探索，提交计划审批） |
| `/mode` | 切换 NORMAL / AUTO / PLAN |
| `/model [alias]` | 查看或热切换供应商 |
| `/resume [sid]` | 恢复历史会话 |
| `/compact` | 手动压缩上下文 |
| `/context` | token 用量估算 |
| `/cost` | 成本汇总 |
| `/undo` | 撤销最近一次写操作 |
| `/rewind <n>` | 回退到第 n 条用户消息 |
| `/exit` | 退出 |

## 测试

```bash
pytest
```

160 个用例，FakeProvider 剧本式假 LLM，确定性执行，不花 API 费。

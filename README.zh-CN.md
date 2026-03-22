# MiniAgent

[English](README.md) · 简体中文

MiniAgent 是一个轻量且逻辑透明的 ReAct Agent 运行时。其控制循环完全
显式：模型输出 `Thought` 与 `Action`，工具执行产生 `Observation`，直到
返回 `Finish`。各环节均记录状态并支持离线测试，便于排查、二开或直接作为
自动化脚本底座。

## 特性

- 显式 ReAct 循环：步数预算、重复动作保护、观察结果截断。
- OpenAI 兼容客户端：指数退避 + 抖动重试、错误分类、token 用量统计。
- 类型化工具注册表：参数 schema、校验、类型转换、输出截断、装饰器注册。
- 内置工具：AST 安全计算器、当前时间、多源联网搜索（Tavily、SerpApi 或 ddgs 降级）。
- 确定性离线模型：没有 API Key 也能跑通完整循环。
- JSONL trace：每次运行写入带 `run_id` 的结构化事件，便于事后排查。
- CLI + 可选 Streamlit 网页界面。
- 97 个测试、91%+ 覆盖率、ruff、mypy strict、Python 3.10-3.13 CI。

## 快速开始

要求 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[web]"

cp .env.example .env
# 编辑 .env：MINI_AGENT_MODEL / MINI_AGENT_API_KEY / MINI_AGENT_BASE_URL

mini-agent run -q "计算 (25+15)*3-8"
```

没有 API Key 时使用离线模型：

```bash
mini-agent run -q "计算 (25+15)*3-8" --mock
# Answer: 112
# [2 steps, 1 tool calls, 3 ms, finish]
```

网页界面：

```bash
streamlit run app.py
```

## 常用命令

```bash
mini-agent                              # 交互式会话
mini-agent run -q "现在几点了？"
mini-agent run -q "..." --mock          # 离线模型
mini-agent run -q "..." --trace run.jsonl
mini-agent run -q "..." --json          # 机器可读输出
mini-agent tools                        # 查看工具及参数
mini-agent config                       # 查看生效配置（Key 已脱敏）
```

交互命令：`/help`、`/tools`、`/config`、`/history`、`/clear`、`/exit`。

## 配置

`MINI_AGENT_*` 优先，同时兼容 `LLM_*`、`OPENAI_*`、`MODELSCOPE_API_KEY`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MINI_AGENT_MODEL` | `gpt-4o-mini` | 模型名称 |
| `MINI_AGENT_API_KEY` | 空 | API Key（本地服务可留空） |
| `MINI_AGENT_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容地址 |
| `MINI_AGENT_TIMEOUT` | `60` | 请求超时（秒） |
| `MINI_AGENT_TEMPERATURE` | `0` | 采样温度 |
| `MINI_AGENT_MAX_STEPS` | `5` | ReAct 最大步数 |
| `MINI_AGENT_MAX_RETRIES` | `3` | 瞬时错误最大尝试次数 |
| `MINI_AGENT_MAX_OBSERVATION_CHARS` | `4000` | 单次观察截断长度 |
| `MINI_AGENT_MOCK` | 关闭 | `1` 启用离线模型 |
| `MINI_AGENT_TRACE` | 空 | JSONL trace 路径（追加写） |

本地 Ollama 示例：

```bash
export MINI_AGENT_BASE_URL=http://localhost:11434/v1
export MINI_AGENT_MODEL=qwen2.5:7b
mini-agent run -q "计算 1+1"
```

## 自定义工具

```python
from mini_agent import ToolParameter, ToolRegistry

registry = ToolRegistry()

@registry.tool(
    "word_count",
    "统计字符数与单词数。",
    (ToolParameter("text", "string", "待统计文本"),),
)
def word_count(text: str) -> str:
    return f"characters={len(text)} words={len(text.split())}"

print(registry.execute("word_count", "hello mini agent").content)
```

多参数工具支持 JSON 对象或 `key=value` 形式：

```text
repeat[prefix=hi, count=3, upper=true]
repeat[{"prefix": "hi", "count": 3, "upper": true}]
```

## 质量门禁

```bash
make dev        # 安装开发依赖
make test       # pytest + 覆盖率（fail-under 85）
make lint       # ruff check + format --check
make typecheck  # mypy strict
```

测试全部使用离线模型与注入的假客户端，CI 不需要 API Key 或网络。

## 目录结构

```text
src/mini_agent/
├── agent.py      # ReAct 循环、事件、RunResult
├── cli.py        # 命令行入口
├── config.py     # 环境变量加载与校验
├── errors.py     # 异常层级
├── llm.py        # OpenAI 兼容客户端、重试策略、usage
├── mock.py       # 确定性离线模型
├── parser.py     # Thought / Action 解析
├── tools.py      # 工具注册表、校验、内置工具
└── trace.py      # JSONL 记录器
tests/            # 97 个离线测试
app.py            # 可选 Streamlit 界面
docs/             # 架构与设计说明
examples/         # 独立示例
```

## 设计取舍与限制

- 文本协议依赖模型的指令遵循能力；原生 function calling 是后续适配方向。
- 运行时是同步的，每步执行一个工具调用。
- 对话历史仅保存在内存中。
- 计算器通过 AST 白名单限定求值范围，并非通用执行沙箱；工具输出与网页内容需按不可信输入处理。
- 联网搜索的质量取决于配置的服务商。

更多细节见 [docs/architecture.md](docs/architecture.md) 与
[docs/design-notes.md](docs/design-notes.md)。

## 贡献与许可

欢迎提交问题反馈与代码改进。安全问题请按
[SECURITY.md](SECURITY.md) 私下报告。项目使用 [MIT](LICENSE) 许可。

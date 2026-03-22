# MiniAgent

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-mypy%20strict-blue)](pyproject.toml)

**A small, auditable ReAct agent runtime.** MiniAgent keeps the control loop
explicit, validates every tool call, retries transient model failures, and
writes inspectable JSONL traces. It is intentionally not a framework: you can
read the whole core in one sitting, then extend it without fighting layers of
abstraction.

## Why another agent project?

Most agent libraries hide the loop behind chains, graphs or planners. That is
useful in production, but it makes debugging and review hard. MiniAgent takes
the opposite approach:

- **The loop is the product.** Thought → Action → Observation → Finish is a
  small state machine you can step through in a debugger.
- **Failures are first-class.** Transient LLM errors are retried with backoff;
  malformed model output is fed back as an observation; repeated actions stop
  the run instead of burning tokens.
- **Everything is testable.** The model is a protocol. Tests inject a scripted
  model and never touch the network.
- **Runs are reproducible.** `--trace` writes one JSONL record per event with a
  stable `run_id`, so a failed run can be inspected after the fact.

## Features

- Explicit ReAct loop with step budget, repeated-action guard and observation
  truncation.
- OpenAI-compatible client with exponential backoff, jitter, typed errors and
  token-usage accounting.
- Typed tool registry: parameter schemas, validation, type coercion, output
  truncation and a decorator API.
- Built-in tools: safe AST calculator, current time, and multi-provider web
  search (Tavily → SerpApi → ddgs fallback).
- Deterministic offline model: the full loop runs without an API key.
- Append-only JSONL traces for every run.
- CLI with `run`, `tools` and `config` commands, JSON output, and trace support.
- Optional Streamlit UI with live reasoning events and run statistics.
- 97 tests, 91%+ coverage, ruff, mypy strict, and a CI matrix for Python
  3.10–3.13.

## Architecture

```mermaid
flowchart LR
    Q[Question] --> A[MiniAgent]
    A -->|prompt| M[ChatModel]
    M -->|Thought + Action| A
    A -->|tool[argument]| R[ToolRegistry]
    R --> C[calculator]
    R --> T[current_time]
    R --> S[web_search]
    R -->|Observation| A
    A -->|Finish[answer]| Answer[RunResult]
    A -.events.-> Trace[(JSONL trace)]
```

The core loop (`src/mini_agent/agent.py`):

```text
for step in 1..max_steps:
    prompt = render(tools, dialogue, question, history)
    response = model.chat(prompt)
    thought, action = parse(response)

    if action is Finish: return answer
    if action is a tool call:
        observation = registry.execute(tool, argument)
        history += (action, observation)
        continue

    history += format_error_feedback
```

Design details, failure modes and trade-offs are documented in
[docs/architecture.md](docs/architecture.md) and
[docs/design-notes.md](docs/design-notes.md).

## Quickstart

Requires Python 3.10+.

```bash
git clone <repository-url> mini-agent
cd mini-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[web]"

cp .env.example .env
# edit .env: MINI_AGENT_MODEL / MINI_AGENT_API_KEY / MINI_AGENT_BASE_URL

mini-agent run -q "Calculate (25+15)*3-8"
```

No API key? Run the deterministic offline model:

```bash
mini-agent run -q "Calculate (25+15)*3-8" --mock
```

```text
-- Step 1 --
Thought: I need the calculator for this expression.
Action: calculator[(25+15)*3-8]
Observation: 112

-- Step 2 --
Thought: The tool returned a result, so I can answer.

Answer: 112
[2 steps, 1 tool calls, 3 ms, finish]
```

## Usage

### CLI

```bash
mini-agent                              # interactive session
mini-agent run -q "What time is it?"    # one question
mini-agent run -q "..." --mock          # offline model
mini-agent run -q "..." --trace run.jsonl
mini-agent run -q "..." --json          # machine-readable result
mini-agent tools                        # list tools and parameters
mini-agent config                       # effective config (key redacted)
```

Interactive commands: `/help`, `/tools`, `/config`, `/history`, `/clear`,
`/exit`.

Exit codes: `0` success, `1` runtime/LLM error, `2` configuration error,
`130` interrupted.

### Python API

```python
from mini_agent import Config, MiniAgent, build_default_registry, build_model

config = Config.from_env()
agent = MiniAgent(build_model(config), build_default_registry())
result = agent.run("Calculate (25+15)*3-8")

print(result.answer)          # 112
print(result.steps)           # 2
print(result.tool_calls)      # 1
print(result.stopped_reason)  # finish
```

### Streamlit UI

```bash
streamlit run app.py
```

The sidebar exposes the mock toggle, step budget, temperature, tool list and
example prompts. Each assistant message stores its reasoning events so you can
re-open the trace after the run.

## Configuration

`MINI_AGENT_*` variables take precedence. Common `LLM_*`, `OPENAI_*` and
`MODELSCOPE_API_KEY` variables are accepted as fallbacks.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MINI_AGENT_MODEL` | `gpt-4o-mini` | Model name sent to the provider |
| `MINI_AGENT_API_KEY` | empty | API key (not required for local servers) |
| `MINI_AGENT_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `MINI_AGENT_TIMEOUT` | `60` | Request timeout in seconds |
| `MINI_AGENT_TEMPERATURE` | `0` | Sampling temperature |
| `MINI_AGENT_MAX_STEPS` | `5` | ReAct step budget |
| `MINI_AGENT_MAX_RETRIES` | `3` | Attempts for transient LLM errors |
| `MINI_AGENT_MAX_OBSERVATION_CHARS` | `4000` | Per-observation truncation |
| `MINI_AGENT_MOCK` | off | `1` enables the offline model |
| `MINI_AGENT_TRACE` | empty | JSONL trace path (append mode) |

Local servers on `localhost` / `127.0.0.1` receive the placeholder key `local`
when no key is configured, because SDKs require a non-empty value.

## Tools

Tools are functions with a schema. The registry validates the argument, coerces
its type and truncates the output before it reaches the model.

```python
from mini_agent import ToolParameter, ToolRegistry

registry = ToolRegistry()

@registry.tool(
    "word_count",
    "Count characters and whitespace-separated words.",
    (ToolParameter("text", "string", "Text to analyse."),),
)
def word_count(text: str) -> str:
    return f"characters={len(text)} words={len(text.split())}"

print(registry.execute("word_count", "hello mini agent").content)
```

Multi-argument tools accept JSON objects or `key=value` pairs:

```text
repeat[prefix=hi, count=3, upper=true]
repeat[{"prefix": "hi", "count": 3, "upper": true}]
```

See [examples/custom_tool.py](examples/custom_tool.py) and
[examples/minimal_react.py](examples/minimal_react.py) for standalone examples.
The built-in calculator evaluates expressions through an AST whitelist and
rejects imports, attribute access, comprehensions and oversized exponents.

## Tracing

```bash
mini-agent run -q "Calculate 1+1" --mock --trace traces/run.jsonl
```

Each line is a standalone JSON object:

```json
{"ts":"2026-11-17T14:45:00.123+00:00","run_id":"5f2c...","event":"run_start","question":"Calculate 1+1","model":"mock","max_steps":5}
{"ts":"2026-11-17T14:45:00.124+00:00","run_id":"5f2c...","event":"step_start","step":1}
{"ts":"2026-11-17T14:45:00.125+00:00","run_id":"5f2c...","event":"tool_result","step":1,"tool":"calculator","ok":true,"duration_ms":0.04,"output_chars":1}
{"ts":"2026-11-17T14:45:00.126+00:00","run_id":"5f2c...","event":"run_end","answer":"2","steps":2,"tool_calls":1,"duration_ms":0.3,"stopped_reason":"finish","total_tokens":123}
```

Traces are opt-in because prompts and observations may contain user data. API
keys are never written.

## Quality gates

```bash
make dev        # pip install -e ".[dev,web]"
make test       # pytest + coverage, fail-under 85
make lint       # ruff check + ruff format --check
make typecheck  # mypy strict
```

The GitHub Actions workflow runs lint, type checking and tests on Python 3.10,
3.11, 3.12 and 3.13, plus a wheel build. The test suite uses only the offline
model and injected fakes, so CI never needs an API key or network access.

## Project layout

```text
src/mini_agent/
├── agent.py      # ReAct loop, events, RunResult
├── cli.py        # argparse entry point
├── config.py     # environment loading and validation
├── errors.py     # exception hierarchy
├── llm.py        # OpenAI-compatible client, retry policy, usage
├── mock.py       # deterministic offline model
├── parser.py     # Thought/Action parsing
├── tools.py      # registry, validation, built-in tools
└── trace.py      # JSONL recorder
tests/            # 97 offline tests
app.py            # optional Streamlit UI
docs/             # architecture and design notes
examples/         # standalone examples
```

## Limitations

- The text protocol depends on the model's instruction-following ability.
  Native function calling is a planned adapter, not the current default.
- The runtime is synchronous and executes one tool call per step.
- Conversation history is in-memory only; persistence is left to the caller.
- The calculator is a safe evaluator, not a general-purpose sandbox. Treat tool
  output and web content as untrusted input.
- Web search quality and availability depend on the configured provider.

## Contributing

Bug reports, design feedback and pull requests are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR. Security issues should
follow [SECURITY.md](SECURITY.md), not the public issue tracker.

## License

[MIT](LICENSE) © 2026 MiniAgent contributors.

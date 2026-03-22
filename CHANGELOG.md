# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-17

### Added

- Explicit ReAct loop with configurable step budget, repeated-action guard and
  observation truncation.
- OpenAI-compatible `LLMClient` with exponential backoff, jitter, typed errors
  and token-usage accounting.
- Typed `ToolRegistry` with parameter schemas, validation, coercion, output
  truncation and a decorator registration API.
- Built-in tools: safe AST calculator, current time and multi-provider web
  search (Tavily, SerpApi, ddgs).
- Append-only JSONL tracing with one `run_id` per agent run.
- Deterministic offline model (`--mock`) for demos and CI.
- CLI (`run`, `tools`, `config`) with JSON output and trace support.
- Optional Streamlit UI with live reasoning events and run statistics.
- pytest suite (97 tests, 91%+ coverage), strict mypy configuration, ruff
  lint/format and a GitHub Actions matrix for Python 3.10-3.13.

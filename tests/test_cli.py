"""CLI tests (offline mock model only)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from mini_agent.cli import main

ENV_VARS = [
    "MINI_AGENT_MODEL",
    "MINI_AGENT_API_KEY",
    "MINI_AGENT_BASE_URL",
    "MINI_AGENT_TIMEOUT",
    "MINI_AGENT_TEMPERATURE",
    "MINI_AGENT_MAX_STEPS",
    "MINI_AGENT_MAX_RETRIES",
    "MINI_AGENT_MAX_OBSERVATION_CHARS",
    "MINI_AGENT_MOCK",
    "MINI_AGENT_TRACE",
    "LLM_MODEL_ID",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_TIMEOUT",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "MODELSCOPE_API_KEY",
    "AGENT_MOCK",
    "TAVILY_API_KEY",
    "SERPAPI_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mini_agent.cli.load_env", lambda: None)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_run_once_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "-q", "Calculate 1+1", "--mock", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["answer"] == "2"
    assert payload["stopped_reason"] == "finish"
    assert payload["steps"] == 2
    assert payload["tool_calls"] == 1
    assert payload["usage"]["total_tokens"] > 0


def test_run_once_with_trace(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    trace_path = tmp_path / "run.jsonl"
    exit_code = main(
        [
            "run",
            "-q",
            "Calculate 1+1",
            "--mock",
            "--json",
            "--trace",
            str(trace_path),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert records[0]["event"] == "run_start"
    assert records[-1]["event"] == "run_end"
    assert records[-1]["answer"] == "2"


def test_run_without_explicit_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--mock", "-q", "Calculate 2+2", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["answer"] == "4"


def test_tools_listing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools"]) == 0
    output = capsys.readouterr().out
    assert "calculator(expression: string)" in output
    assert "current_time(" in output
    assert "web_search(query: string)" in output


def test_tools_json_listing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools", "--json"]) == 0
    tools = json.loads(capsys.readouterr().out)
    names = {tool["name"] for tool in tools}
    assert names == {"calculator", "current_time", "web_search"}
    calculator = next(tool for tool in tools if tool["name"] == "calculator")
    assert calculator["parameters"][0]["name"] == "expression"


def test_config_redacts_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MINI_AGENT_API_KEY", "super-secret")
    assert main(["config"]) == 0
    output = capsys.readouterr().out
    assert "***" in output
    assert "super-secret" not in output


def test_missing_api_key_returns_config_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "-q", "hello"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error" in captured.err
    assert "No API key" in captured.err


def test_interactive_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    commands = "\n".join(
        [
            "Calculate 1+1",
            "/history",
            "/tools",
            "/config",
            "/help",
            "/clear",
            "/exit",
            "",
        ]
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(commands))

    assert main(["run", "--mock"]) == 0
    output = capsys.readouterr().out
    assert "MiniAgent" in output
    assert "Assistant: 2" in output
    assert "Conversation history cleared." in output


def test_run_once_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-q", "Calculate 1+1", "--mock"]) == 0
    output = capsys.readouterr().out
    assert "Thought:" in output
    assert "Observation: 2" in output
    assert "Answer: 2" in output
    assert "finish]" in output


def test_quiet_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-q", "Calculate 1+1", "--mock", "--quiet"]) == 0
    output = capsys.readouterr().out
    assert "Thought:" not in output
    assert "Answer: 2" in output


def test_default_command_is_interactive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MINI_AGENT_MOCK", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("/unknown\n/exit\n"))

    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Unknown command" in output
    assert "Bye." in output


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert "mini-agent" in capsys.readouterr().out


def test_invalid_trace_path_returns_config_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    exit_code = main(["run", "-q", "hi", "--mock", "--trace", str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error" in captured.err

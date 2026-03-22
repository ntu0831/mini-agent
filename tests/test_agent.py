"""ReAct loop behaviour tests using a scripted model."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from mini_agent import (
    AgentEvent,
    LLMResponse,
    MiniAgent,
    ToolRegistry,
    TraceRecorder,
    Usage,
    build_default_registry,
)
from mini_agent.llm import ChatMessage


class ScriptedModel:
    name = "scripted"

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        text = self.texts.pop(0) if self.texts else "Thought: done\nAction: Finish[fallback]"
        return LLMResponse(
            text=text,
            model="scripted",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def _agent(texts: list[str], **kwargs: object) -> MiniAgent:
    return MiniAgent(ScriptedModel(texts), build_default_registry(), **kwargs)  # type: ignore[arg-type]


def test_immediate_finish() -> None:
    events: list[AgentEvent] = []
    agent = _agent(["Thought: easy\nAction: Finish[42]"])
    result = agent.run("question", on_event=events.append)

    assert result.answer == "42"
    assert result.steps == 1
    assert result.tool_calls == 0
    assert result.stopped_reason == "finish"
    assert result.usage.total_tokens == 2
    assert [event.type for event in events] == ["step", "thought", "final"]
    assert str(result) == "42"


def test_tool_call_then_finish() -> None:
    model = ScriptedModel(
        [
            "Thought: need the calculator\nAction: calculator[1+2]",
            "Thought: done\nAction: Finish[3]",
        ]
    )
    agent = MiniAgent(model, build_default_registry())
    events: list[AgentEvent] = []
    result = agent.run("what is 1+2?", on_event=events.append)

    assert result.answer == "3"
    assert result.steps == 2
    assert result.tool_calls == 1
    observations = [event.content for event in events if event.type == "observation"]
    assert observations == ["3"]
    assert "Observation: 3" in model.calls[1][1]["content"]


def test_malformed_response_is_fed_back() -> None:
    agent = _agent(
        [
            "I will not follow the format.",
            "Thought: corrected\nAction: Finish[recovered]",
        ]
    )
    events: list[AgentEvent] = []
    result = agent.run("question", on_event=events.append)

    assert result.answer == "recovered"
    assert result.steps == 2
    assert any(event.type == "error" for event in events)


def test_max_steps_is_enforced() -> None:
    agent = _agent(
        [
            "Thought: one\nAction: missing[1]",
            "Thought: two\nAction: missing[2]",
        ],
        max_steps=2,
    )
    result = agent.run("question")

    assert result.stopped_reason == "max_steps"
    assert result.steps == 2
    assert result.tool_calls == 2
    assert "2 steps" in result.answer


def test_repeated_action_guard() -> None:
    agent = _agent(
        ["Thought: retry\nAction: calculator[1+1]"] * 5,
        max_steps=5,
        loop_threshold=2,
    )
    events: list[AgentEvent] = []
    result = agent.run("question", on_event=events.append)

    assert result.stopped_reason == "loop"
    assert result.tool_calls == 2
    assert "repeated 3 times" in result.answer
    observations = [event.content for event in events if event.type == "observation"]
    assert len(observations) == 2
    assert observations[1].startswith("Note: this Action has already been executed")


def test_observation_truncation() -> None:
    registry = ToolRegistry()
    registry.tool("long", "Long output.")(lambda: "x" * 100)
    agent = MiniAgent(
        ScriptedModel(
            [
                "Thought: long\nAction: long[]",
                "Thought: done\nAction: Finish[done]",
            ]
        ),
        registry,
        max_observation_chars=20,
    )
    events: list[AgentEvent] = []
    agent.run("question", on_event=events.append)

    observation = next(event.content for event in events if event.type == "observation")
    assert observation.startswith("x" * 20)
    assert "truncated 80 chars" in observation


def test_multi_turn_dialogue() -> None:
    model = ScriptedModel(
        [
            "Thought: calc\nAction: calculator[1+1]",
            "Thought: done\nAction: Finish[2]",
            "Thought: calc\nAction: calculator[2+2]",
            "Thought: done\nAction: Finish[4]",
        ]
    )
    agent = MiniAgent(model, build_default_registry())

    assert agent.run("first").answer == "2"
    assert agent.run("second").answer == "4"
    assert len(agent.dialogue) == 2
    prompt = model.calls[2][1]["content"]
    assert "Previous conversation" in prompt
    assert "User: first" in prompt
    assert "Assistant: 2" in prompt

    agent.clear_history()
    assert agent.dialogue == []


def test_event_callback_errors_are_ignored() -> None:
    agent = _agent(["Thought: done\nAction: Finish[ok]"])

    def broken(event: AgentEvent) -> None:
        raise RuntimeError("UI is broken")

    assert agent.run("question", on_event=broken).answer == "ok"


def test_empty_question() -> None:
    with pytest.raises(ValueError, match="question"):
        _agent([]).run("   ")


def test_invalid_agent_configuration() -> None:
    with pytest.raises(ValueError):
        _agent([], max_steps=0)
    with pytest.raises(ValueError):
        _agent([], max_observation_chars=0)
    with pytest.raises(ValueError):
        _agent([], loop_threshold=0)


def test_trace_file_contains_structured_events(tmp_path: object) -> None:
    path = tmp_path / "trace.jsonl"  # type: ignore[operator]
    with TraceRecorder(path) as trace:
        agent = _agent(
            [
                "Thought: calc\nAction: calculator[1+2]",
                "Thought: done\nAction: Finish[3]",
            ],
            trace=trace,
        )
        agent.run("question")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    events = [record["event"] for record in records]
    assert events[0] == "run_start"
    assert "tool_result" in events
    assert events[-1] == "run_end"
    assert records[-1]["answer"] == "3"
    assert records[-1]["stopped_reason"] == "finish"
    assert len({record["run_id"] for record in records}) == 1
    assert all("api_key" not in json.dumps(record) for record in records)

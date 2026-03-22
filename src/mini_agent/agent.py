"""The ReAct agent loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .llm import ChatMessage, ChatModel, Usage, add_usage
from .parser import is_finish, parse_action, parse_thought_action
from .tools import ToolRegistry
from .trace import TraceRecorder

EventType = Literal["step", "thought", "action", "observation", "error", "final"]
StoppedReason = Literal["finish", "max_steps", "loop"]

SYSTEM_PROMPT = (
    "You are a precise ReAct agent. Follow the requested Thought/Action "
    "format exactly and never invent tool observations."
)

REACT_PROMPT = """You are a precise ReAct agent. You can call tools to gather information and then answer.

## Available tools
{tools}

## Previous conversation
{dialogue}

## Response format (exactly two lines per step)
Thought: your reasoning
Action: tool_name[argument]    or    Finish[final answer]

## Rules
1. Execute one Action per step and wait for the Observation.
2. Use only the tool names listed above; put the argument inside square brackets.
3. Use Finish[...] only when you can answer. The final answer must be complete and direct.
4. Never invent observations. Do not output anything outside the two lines.

## Current task
Question: {question}

## Execution history
{history}"""


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A progress event emitted while the agent runs."""

    type: EventType
    content: str
    step: int


@dataclass(frozen=True, slots=True)
class Turn:
    """One completed question/answer pair."""

    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class RunResult:
    """Structured result of one agent run."""

    answer: str
    steps: int
    tool_calls: int
    duration_ms: float
    usage: Usage
    stopped_reason: StoppedReason

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.answer


class MiniAgent:
    """A small ReAct agent with a step budget and a repeated-action guard."""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        *,
        max_steps: int = 5,
        system_prompt: str = SYSTEM_PROMPT,
        max_observation_chars: int = 4000,
        loop_threshold: int = 2,
        trace: TraceRecorder | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_observation_chars < 1:
            raise ValueError("max_observation_chars must be at least 1")
        if loop_threshold < 1:
            raise ValueError("loop_threshold must be at least 1")

        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.max_observation_chars = max_observation_chars
        self.loop_threshold = loop_threshold
        self.dialogue: list[Turn] = []

        self._trace = trace or TraceRecorder(None)
        self._logger = logger or logging.getLogger(__name__)

    def run(
        self,
        question: str,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> RunResult:
        """Run the ReAct loop until ``Finish``, the step budget, or a loop."""

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        started = time.perf_counter()
        usage = Usage()
        tool_calls = 0
        steps = 0
        history: list[str] = []
        action_counts: dict[str, int] = {}
        model_name = getattr(self.model, "name", type(self.model).__name__)

        self._trace.emit(
            "run_start",
            question=question,
            model=model_name,
            max_steps=self.max_steps,
        )

        try:
            for step in range(1, self.max_steps + 1):
                steps = step
                self._emit(on_event, "step", f"Step {step}", step)
                self._trace.emit("step_start", step=step)

                prompt = REACT_PROMPT.format(
                    tools=self.tools.describe(),
                    dialogue=self._format_dialogue(),
                    question=question,
                    history="\n".join(history) if history else "(none)",
                )
                messages: Sequence[ChatMessage] = (
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                )
                response = self.model.chat(messages)
                usage = add_usage(usage, response.usage)
                self._trace.emit(
                    "llm_response",
                    step=step,
                    model=response.model,
                    latency_ms=round(response.latency_ms, 2),
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

                thought, action = parse_thought_action(response.text)
                if thought:
                    self._emit(on_event, "thought", thought, step)
                    self._trace.emit("thought", step=step, content=thought)

                if is_finish(action):
                    _, answer = parse_action(action)
                    answer = (answer or "").strip() or (
                        "(the model returned an empty final answer)"
                    )
                    return self._complete(
                        question=question,
                        answer=answer,
                        steps=steps,
                        tool_calls=tool_calls,
                        started=started,
                        usage=usage,
                        stopped_reason="finish",
                        on_event=on_event,
                    )

                if action:
                    self._emit(on_event, "action", action, step)
                    self._trace.emit("action", step=step, content=action)
                    tool_name, tool_input = parse_action(action)
                    if tool_name is None:
                        observation = (
                            "Could not parse the Action. Use tool_name[argument] or Finish[answer]."
                        )
                        self._trace.emit("format_error", step=step, content=observation)
                    else:
                        canonical = f"{tool_name.lower()}[{(tool_input or '').strip()}]"
                        count = action_counts.get(canonical, 0) + 1
                        action_counts[canonical] = count
                        if count > self.loop_threshold:
                            answer = (
                                "Stopped: the same action was repeated "
                                f"{count} times without progress."
                            )
                            self._trace.emit(
                                "loop_detected",
                                step=step,
                                action=action,
                                count=count,
                            )
                            self._emit(on_event, "error", answer, step)
                            return self._complete(
                                question=question,
                                answer=answer,
                                steps=steps,
                                tool_calls=tool_calls,
                                started=started,
                                usage=usage,
                                stopped_reason="loop",
                                on_event=on_event,
                            )

                        result = self.tools.execute(tool_name, tool_input or "")
                        tool_calls += 1
                        observation = self._truncate(result.content)
                        if count == self.loop_threshold:
                            observation = (
                                "Note: this Action has already been executed. "
                                "Try a different Action or use Finish[...].\n" + observation
                            )
                        self._trace.emit(
                            "tool_result",
                            step=step,
                            tool=result.tool,
                            ok=result.ok,
                            duration_ms=round(result.duration_ms, 2),
                            output_chars=len(result.content),
                        )

                    history.append(f"Action: {action}")
                    history.append(f"Observation: {observation}")
                    self._emit(on_event, "observation", observation, step)
                    continue

                feedback = (
                    "Could not parse your response. Reply with exactly "
                    "`Thought: ...` and `Action: tool_name[argument]` "
                    "(or `Finish[answer]`)."
                )
                history.append(f"Assistant: {response.text}")
                history.append(f"Observation: {feedback}")
                self._trace.emit("format_error", step=step, content=feedback)
                self._emit(on_event, "error", feedback, step)

            answer = f"Stopped: the task was not completed within {self.max_steps} steps."
            return self._complete(
                question=question,
                answer=answer,
                steps=steps,
                tool_calls=tool_calls,
                started=started,
                usage=usage,
                stopped_reason="max_steps",
                on_event=on_event,
            )
        except Exception as exc:
            self._trace.emit("run_error", error=f"{type(exc).__name__}: {exc}")
            raise

    def clear_history(self) -> None:
        self.dialogue.clear()

    def _complete(
        self,
        *,
        question: str,
        answer: str,
        steps: int,
        tool_calls: int,
        started: float,
        usage: Usage,
        stopped_reason: StoppedReason,
        on_event: Callable[[AgentEvent], None] | None,
    ) -> RunResult:
        duration_ms = (time.perf_counter() - started) * 1000.0
        self.dialogue.append(Turn(question=question, answer=answer))
        self._emit(on_event, "final", answer, steps)
        self._trace.emit(
            "run_end",
            answer=answer,
            steps=steps,
            tool_calls=tool_calls,
            duration_ms=round(duration_ms, 2),
            stopped_reason=stopped_reason,
            total_tokens=usage.total_tokens,
        )
        return RunResult(
            answer=answer,
            steps=steps,
            tool_calls=tool_calls,
            duration_ms=duration_ms,
            usage=usage,
            stopped_reason=stopped_reason,
        )

    def _format_dialogue(self) -> str:
        if not self.dialogue:
            return "(none)"
        parts = []
        for turn in self.dialogue[-5:]:
            question = turn.question
            answer = turn.answer
            if len(question) > 500:
                question = question[:500] + "..."
            if len(answer) > 300:
                answer = answer[:300] + "..."
            parts.append(f"User: {question}\nAssistant: {answer}")
        return "\n\n".join(parts)

    def _truncate(self, content: str) -> str:
        if len(content) <= self.max_observation_chars:
            return content
        omitted = len(content) - self.max_observation_chars
        return f"{content[: self.max_observation_chars]}\n...[truncated {omitted} chars]"

    def _emit(
        self,
        callback: Callable[[AgentEvent], None] | None,
        event_type: EventType,
        content: str,
        step: int,
    ) -> None:
        if callback is None:
            return
        try:
            callback(AgentEvent(type=event_type, content=content, step=step))
        except Exception:  # noqa: BLE001 - presentation errors must not break the run
            self._logger.debug("agent event callback failed", exc_info=True)

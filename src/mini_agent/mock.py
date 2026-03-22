"""Deterministic offline model used for demos, tests and CI.

It follows the same ReAct contract as a real model: pick a tool on the first
step, then finish with the observation. It is intentionally rule-based and does
not pretend to be intelligent.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

from .llm import ChatMessage, LLMResponse, Usage


class MockLLM:
    """Rule-based model implementing the :class:`ChatModel` protocol."""

    name = "mock"

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
    ) -> LLMResponse:
        prompt = messages[-1]["content"] if messages else ""
        observations = re.findall(r"^Observation:\s*(.*)$", prompt, re.MULTILINE)
        if observations:
            text = (
                "Thought: The tool returned a result, so I can answer.\n"
                f"Action: Finish[{observations[-1].strip()}]"
            )
            return self._response(prompt, text)

        question_match = re.search(r"Question:\s*(.*?)(?=\n## |\Z)", prompt, re.DOTALL)
        question = question_match.group(1).strip() if question_match else prompt

        if re.search(r"\b(hi|hello|hey)\b", question, re.IGNORECASE) or re.search(
            r"(你好|您好)", question
        ):
            text = (
                "Thought: This is a greeting.\n"
                "Action: Finish[Hello! I am the offline mock model. "
                "Configure an API key to use a real model.]"
            )
            return self._response(prompt, text)

        if re.search(r"\b(time|date|clock)\b", question, re.IGNORECASE) or re.search(
            r"(时间|几点|日期|今天)", question
        ):
            text = "Thought: I need the current time.\nAction: current_time[local]"
            return self._response(prompt, text)

        expression = _extract_math_expression(question)
        if expression:
            text = (
                "Thought: I need the calculator for this expression.\n"
                f"Action: calculator[{expression}]"
            )
            return self._response(prompt, text)

        query = re.sub(
            r"^(please\s+)?(search(?:\s+for)?|look up|find)\s+",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip(" ,.?!")
        text = (
            f"Thought: This question needs a web search.\nAction: web_search[{query or question}]"
        )
        return self._response(prompt, text)

    def _response(self, prompt: str, text: str) -> LLMResponse:
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(text) // 4)
        return LLMResponse(
            text=text,
            model=self.model,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=0.0,
        )


def _extract_math_expression(text: str) -> str | None:
    """Extract a parseable arithmetic expression from natural language."""

    normalized = text.replace("×", "*").replace("÷", "/").replace("^", "**")
    candidates = sorted(re.findall(r"[0-9.+\-*/()%\s]+", normalized), key=len, reverse=True)
    for candidate in candidates:
        candidate = candidate.strip()
        if not re.search(r"\d", candidate) or not re.search(r"[+\-*/%]", candidate):
            continue
        for expression in (candidate, candidate.strip("+*/%- ")):
            text_expression = str(expression)
            try:
                ast.parse(text_expression, mode="eval")
            except SyntaxError:
                continue
            return text_expression
    return None

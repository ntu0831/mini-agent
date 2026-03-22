"""Offline mock model behaviour."""

from __future__ import annotations

from mini_agent.mock import MockLLM


def _prompt(question: str, history: str = "(none)") -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"Question: {question}\n## Execution history\n{history}",
        }
    ]


def test_observation_leads_to_finish() -> None:
    response = MockLLM().chat(_prompt("what is 1+1?", "Action: calculator[1+1]\nObservation: 2"))
    assert "Action: Finish[2]" in response.text
    assert response.usage.total_tokens > 0


def test_greeting_in_english_and_chinese() -> None:
    assert "Finish[" in MockLLM().chat(_prompt("hello there")).text
    assert "Finish[" in MockLLM().chat(_prompt("你好")).text


def test_time_branch() -> None:
    english = MockLLM().chat(_prompt("What time is it?")).text
    chinese = MockLLM().chat(_prompt("现在几点了？")).text
    assert "current_time[local]" in english
    assert "current_time[local]" in chinese


def test_math_branch() -> None:
    response = MockLLM().chat(_prompt("Calculate (25+15)*3-8")).text
    assert "calculator[(25+15)*3-8]" in response


def test_search_fallback_branch() -> None:
    response = MockLLM().chat(_prompt("Search for Python")).text
    assert "web_search[Python]" in response


def test_empty_messages() -> None:
    response = MockLLM().chat([])
    assert "web_search[" in response.text

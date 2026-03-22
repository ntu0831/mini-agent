"""Tests for the text ReAct protocol parser."""

from __future__ import annotations

from mini_agent.parser import is_finish, parse_action, parse_thought_action


def test_plain_response() -> None:
    thought, action = parse_thought_action(
        "Thought: I need the calculator\nAction: calculator[(1+2)*3]"
    )
    assert thought == "I need the calculator"
    assert action == "calculator[(1+2)*3]"


def test_fenced_response() -> None:
    thought, action = parse_thought_action(
        "```text\nThought: check the time\nAction: current_time[local]\n```"
    )
    assert thought == "check the time"
    assert action == "current_time[local]"


def test_chinese_colons_and_finish_case() -> None:
    thought, action = parse_thought_action("Thought：done\nAction：Finish[42]")
    assert thought == "done"
    assert action == "Finish[42]"
    assert is_finish(action)


def test_action_with_trailing_prose() -> None:
    _, action = parse_thought_action(
        "Thought: go\nAction: calculator[1+2]\n\nThat should be enough."
    )
    assert parse_action(action) == ("calculator", "1+2")


def test_multiline_thought() -> None:
    thought, action = parse_thought_action("Thought: line one\nline two\nAction: Finish[ok]")
    assert thought == "line one\nline two"
    assert action == "Finish[ok]"


def test_missing_action() -> None:
    thought, action = parse_thought_action("Thought: only thinking")
    assert thought == "only thinking"
    assert action is None
    assert is_finish(None) is False


def test_action_with_brackets_in_argument() -> None:
    assert parse_action("web_search[query with [brackets]]") == (
        "web_search",
        "query with [brackets]",
    )


def test_invalid_action() -> None:
    assert parse_action("not a tool call") == (None, None)
    assert parse_action(None) == (None, None)
    assert is_finish("Finish[ok]")
    assert not is_finish("calculator[1+1]")

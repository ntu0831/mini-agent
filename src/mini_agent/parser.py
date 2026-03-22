"""Parsing helpers for the text-based ReAct protocol."""

from __future__ import annotations

import re


def strip_code_fences(text: str) -> str:
    """Remove a fence that wraps the whole response."""

    text = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*\n(.*?)\n?```$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def parse_thought_action(text: str) -> tuple[str | None, str | None]:
    """Return ``(thought, action)`` from a model response.

    The parser tolerates prose before/after the two required lines and uses the
    last ``Action:`` marker, which is the one the model typically means.
    """

    text = strip_code_fences(text)
    action_markers = list(re.finditer(r"Action\s*[:：]\s*", text, re.IGNORECASE))
    action = text[action_markers[-1].end() :].strip() if action_markers else None

    thought_match = re.search(
        r"Thought\s*[:：]\s*(.*?)(?=\n\s*Action\s*[:：]|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    thought = thought_match.group(1).strip() if thought_match else None
    return thought, action


def parse_action(action_text: str | None) -> tuple[str | None, str | None]:
    """Parse ``tool[argument]`` into ``(tool, argument)``.

    A greedy match is attempted first so arguments may contain brackets. If the
    response has trailing prose, a non-greedy fallback recovers the first call.
    """

    if not action_text:
        return None, None
    action_text = strip_code_fences(action_text).strip()

    match = re.match(
        r"^([A-Za-z_][A-Za-z0-9_-]*)\s*\[(.*)\]\s*$",
        action_text,
        re.DOTALL,
    )
    if match is None:
        match = re.search(
            r"([A-Za-z_][A-Za-z0-9_-]*)\s*\[(.*?)\]",
            action_text,
            re.DOTALL,
        )
    if match is None:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def is_finish(action_text: str | None) -> bool:
    """Return whether an Action text is a ``Finish[...]`` instruction."""

    if action_text is None:
        return False
    return action_text.strip().lower().startswith("finish")

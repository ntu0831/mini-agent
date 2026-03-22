"""Tool registry, validation and built-in tool tests."""

from __future__ import annotations

import json

import pytest

from mini_agent import (
    ToolError,
    ToolParameter,
    ToolRegistry,
    ToolSpec,
    build_default_registry,
    current_time,
    web_search,
)


def _repeat(prefix: str, count: int = 1, upper: bool = False) -> str:
    text = " ".join([prefix] * int(count))
    return text.upper() if upper else text


def _multi_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="repeat",
            description="Repeat a prefix.",
            func=_repeat,
            parameters=(
                ToolParameter("prefix", "string", "Text to repeat."),
                ToolParameter("count", "integer", "Repetitions.", required=False, default=1),
                ToolParameter(
                    "upper", "boolean", "Uppercase output.", required=False, default=False
                ),
            ),
        )
    )
    return registry


def test_default_registry_description() -> None:
    registry = build_default_registry()
    assert set(registry.names()) == {"calculator", "current_time", "web_search"}
    description = registry.describe()
    assert "calculator(expression: string)" in description
    assert "current_time(timezone: string = 'local')" in description
    assert "web_search(query: string)" in description
    assert len(registry.specs()) == 3


def test_calculator_tool() -> None:
    registry = build_default_registry()
    result = registry.execute("calculator", "(25+15)*3-8")
    assert result.ok is True
    assert result.content == "112"
    assert result.tool == "calculator"
    assert result.duration_ms >= 0


def test_tool_lookup_is_case_insensitive() -> None:
    registry = build_default_registry()
    assert registry.execute("CALCULATOR", "1+1").content == "2"


def test_unknown_tool() -> None:
    registry = build_default_registry()
    result = registry.execute("missing", "x")
    assert result.ok is False
    assert "Unknown tool" in result.content
    assert "calculator" in result.content


def test_single_argument_json() -> None:
    registry = build_default_registry()
    result = registry.execute("calculator", json.dumps({"expression": "2*3"}))
    assert result.ok is True
    assert result.content == "6"


def test_optional_parameter_default_and_value() -> None:
    registry = build_default_registry()
    local = registry.execute("current_time", "")
    assert local.ok is True
    assert "local" in local.content

    shanghai = registry.execute("current_time", "Asia/Shanghai")
    assert shanghai.ok is True
    assert "Asia/Shanghai" in shanghai.content

    invalid = registry.execute("current_time", "Mars/Olympus")
    assert invalid.ok is True
    assert "unknown timezone" in invalid.content


def test_multi_argument_json() -> None:
    registry = _multi_registry()
    result = registry.execute("repeat", json.dumps({"prefix": "hi", "count": 2, "upper": True}))
    assert result.ok is True
    assert result.content == "HI HI"


def test_multi_argument_key_value() -> None:
    registry = _multi_registry()
    result = registry.execute("repeat", "prefix=hi, count=3, upper=true")
    assert result.ok is True
    assert result.content == "HI HI HI"


def test_missing_required_argument() -> None:
    registry = _multi_registry()
    result = registry.execute("repeat", "count=2")
    assert result.ok is False
    assert "Missing required argument 'prefix'" in result.content


def test_unknown_argument() -> None:
    registry = _multi_registry()
    result = registry.execute("repeat", "prefix=hi, nope=1")
    assert result.ok is False
    assert "Unknown argument" in result.content


def test_bad_type_coercion() -> None:
    registry = _multi_registry()
    assert "must be an integer" in registry.execute("repeat", "prefix=hi, count=many").content
    assert "must be a boolean" in registry.execute("repeat", "prefix=hi, upper=maybe").content


def test_any_parameter_passes_json_values_through() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "echo_any",
            "Return the Python representation of the value.",
            lambda value: repr(value),
            (ToolParameter("value", "any"),),
        )
    )
    result = registry.execute("echo_any", json.dumps({"value": 3}))
    assert result.ok is True
    assert result.content == "3"


def test_no_argument_tool() -> None:
    registry = ToolRegistry()
    registry.tool("ping", "Return pong.")(lambda: "pong")
    assert registry.execute("ping", "").content == "pong"
    result = registry.execute("ping", "unexpected")
    assert result.ok is False
    assert "takes no arguments" in result.content


def test_tool_exception_is_returned() -> None:
    def boom() -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(ToolSpec("boom", "Always fails.", boom))
    result = registry.execute("boom", "")
    assert result.ok is False
    assert "RuntimeError: boom" in result.content


def test_output_truncation() -> None:
    registry = ToolRegistry(max_output_chars=5)
    registry.tool("long", "Long output.")(lambda: "abcdefgh")
    result = registry.execute("long", "")
    assert result.ok is True
    assert result.content.startswith("abcde")
    assert "truncated 3 chars" in result.content


def test_duplicate_registration() -> None:
    registry = ToolRegistry()
    registry.tool("ping", "Return pong.")(lambda: "pong")
    with pytest.raises(ToolError, match="already registered"):
        registry.tool("ping", "Duplicate.")(lambda: "pong")


def test_decorator_registers_and_returns_function() -> None:
    registry = ToolRegistry()

    @registry.tool("shout", "Uppercase text.", (ToolParameter("text"),))
    def shout(text: str) -> str:
        return text.upper()

    assert shout("hi") == "HI"
    assert registry.execute("shout", "hi").content == "HI"


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        ToolSpec("bad name", "x", lambda: "x")
    with pytest.raises(ValueError):
        ToolParameter("bad-name")
    with pytest.raises(ValueError):
        ToolParameter("x", "date")
    with pytest.raises(ValueError):
        ToolSpec(
            "dup",
            "x",
            lambda: "x",
            (ToolParameter("value"), ToolParameter("value")),
        )
    with pytest.raises(ValueError):
        ToolRegistry(max_output_chars=0)


def test_web_search_empty_query() -> None:
    assert "must not be empty" in web_search("  ")


def test_current_time_direct() -> None:
    assert "local" in current_time("")
    assert "Asia/Shanghai" in current_time("Asia/Shanghai")
    assert "unknown timezone" in current_time("Nowhere/Here")

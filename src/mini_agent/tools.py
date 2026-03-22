"""Typed tool registry and built-in tools."""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .errors import ToolError, ToolValidationError

_PARAM_TYPES = {"string", "integer", "number", "boolean", "any"}
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """A single documented tool parameter."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None

    def __post_init__(self) -> None:
        if not _PARAM_NAME_RE.fullmatch(self.name):
            raise ValueError(f"Invalid tool parameter name: {self.name!r}")
        if self.type not in _PARAM_TYPES:
            raise ValueError(
                f"Unsupported parameter type {self.type!r}; expected one of {sorted(_PARAM_TYPES)}"
            )

    def signature(self) -> str:
        if self.required:
            return f"{self.name}: {self.type}"
        return f"{self.name}: {self.type} = {self.default!r}"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool's metadata and implementation."""

    name: str
    description: str
    func: Callable[..., str]
    parameters: tuple[ToolParameter, ...] = ()

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.fullmatch(self.name):
            raise ValueError(f"Invalid tool name: {self.name!r}")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate parameter names in tool {self.name!r}")

    def signature(self) -> str:
        return ", ".join(parameter.signature() for parameter in self.parameters)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalised tool execution result."""

    tool: str
    ok: bool
    content: str
    duration_ms: float = 0.0


class ToolRegistry:
    """Register, describe and execute tools with validation and truncation."""

    def __init__(self, *, max_output_chars: int = 4000) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1")
        self.max_output_chars = max_output_chars
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        existing = self.get(spec.name)
        if existing is not None:
            raise ToolError(f"Tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

    def tool(
        self,
        name: str,
        description: str,
        parameters: tuple[ToolParameter, ...] = (),
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """Decorator that registers a function as a tool."""

        def decorator(func: Callable[..., str]) -> Callable[..., str]:
            self.register(ToolSpec(name, description, func, parameters))
            return func

        return decorator

    def get(self, name: str) -> ToolSpec | None:
        if name in self._tools:
            return self._tools[name]
        lowered = name.lower()
        for tool_name, spec in self._tools.items():
            if tool_name.lower() == lowered:
                return spec
        return None

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def describe(self) -> str:
        if not self._tools:
            return "No tools available."
        lines = []
        for spec in self._tools.values():
            lines.append(f"- {spec.name}({spec.signature()}): {spec.description}")
        return "\n".join(lines)

    def execute(self, name: str, raw_arguments: str = "") -> ToolResult:
        """Parse, validate and execute a tool call.

        Validation failures are returned as ``ok=False`` results instead of
        raising, because the ReAct loop feeds them back to the model.
        """

        started = time.perf_counter()
        spec = self.get(name)
        if spec is None:
            available = ", ".join(self.names()) or "none"
            return ToolResult(
                tool=name,
                ok=False,
                content=f"Unknown tool {name!r}. Available tools: {available}",
            )

        try:
            arguments = self._parse_arguments(spec, raw_arguments)
            output = spec.func(**arguments)
        except ToolValidationError as exc:
            return self._failure(spec.name, f"Invalid arguments: {exc}", started)
        except Exception as exc:  # noqa: BLE001 - tool failures are data for the model
            return self._failure(
                spec.name,
                f"Tool {spec.name!r} failed: {type(exc).__name__}: {exc}",
                started,
            )

        content = self._truncate(str(output))
        return ToolResult(
            tool=spec.name,
            ok=True,
            content=content,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _failure(self, name: str, message: str, started: float) -> ToolResult:
        return ToolResult(
            tool=name,
            ok=False,
            content=message,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _parse_arguments(self, spec: ToolSpec, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        parameters = spec.parameters

        if not parameters:
            if raw:
                raise ToolValidationError(f"Tool {spec.name!r} takes no arguments")
            return {}

        if len(parameters) == 1:
            parameter = parameters[0]
            if not raw:
                arguments: dict[str, Any] = {}
            else:
                parsed = _try_json_object(raw)
                if parsed is not None and parameter.name in parsed:
                    arguments = parsed
                else:
                    arguments = {parameter.name: raw}
        elif raw.startswith("{"):
            parsed = _try_json_object(raw)
            if parsed is None:
                raise ToolValidationError(
                    'Expected a JSON object, for example {"expression": "1+2", "precision": 2}'
                )
            arguments = parsed
        else:
            arguments = {}
            for chunk in re.split(r"[,;\n]", raw):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "=" not in chunk:
                    raise ToolValidationError(
                        "Expected key=value arguments, for example timezone=Asia/Shanghai"
                    )
                key, value = chunk.split("=", 1)
                arguments[key.strip()] = value.strip().strip("\"'")

        return self._validate_arguments(spec, arguments)

    def _validate_arguments(self, spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        known = {parameter.name: parameter for parameter in spec.parameters}
        unknown = set(arguments) - set(known)
        if unknown:
            raise ToolValidationError(
                f"Unknown argument(s) for {spec.name!r}: {', '.join(sorted(unknown))}"
            )

        resolved: dict[str, Any] = {}
        for parameter in spec.parameters:
            if parameter.name in arguments and arguments[parameter.name] != "":
                resolved[parameter.name] = _coerce(parameter, arguments[parameter.name])
            elif parameter.required:
                raise ToolValidationError(
                    f"Missing required argument {parameter.name!r} for {spec.name!r}"
                )
            else:
                resolved[parameter.name] = parameter.default
        return resolved

    def _truncate(self, content: str) -> str:
        if len(content) <= self.max_output_chars:
            return content
        omitted = len(content) - self.max_output_chars
        return f"{content[: self.max_output_chars]}\n...[truncated {omitted} chars]"


def _try_json_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce(parameter: ToolParameter, value: Any) -> Any:
    if parameter.type == "string":
        return str(value)
    if parameter.type == "any":
        return value
    if parameter.type == "integer":
        if isinstance(value, bool):
            raise ToolValidationError(f"Argument {parameter.name!r} must be an integer")
        try:
            return int(str(value).strip())
        except ValueError as exc:
            raise ToolValidationError(
                f"Argument {parameter.name!r} must be an integer, got {value!r}"
            ) from exc
    if parameter.type == "number":
        if isinstance(value, bool):
            raise ToolValidationError(f"Argument {parameter.name!r} must be a number")
        try:
            return float(str(value).strip())
        except ValueError as exc:
            raise ToolValidationError(
                f"Argument {parameter.name!r} must be a number, got {value!r}"
            ) from exc
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ToolValidationError(f"Argument {parameter.name!r} must be a boolean, got {value!r}")


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
_MAX_EXPRESSION_CHARS = 500


def _numeric(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expression did not produce a number")
    return value


def _eval_node(node: ast.AST) -> int | float:
    """Evaluate an AST node using an explicit whitelist."""

    if isinstance(node, ast.Constant):
        return _numeric(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent is too large")
        return _numeric(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _numeric(_ALLOWED_UNARY[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only named functions are allowed")
        if node.func.id not in _ALLOWED_FUNCS:
            raise ValueError(f"function {node.func.id!r} is not allowed")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return _numeric(_ALLOWED_FUNCS[node.func.id](*[_eval_node(arg) for arg in node.args]))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _numeric(_ALLOWED_NAMES[node.id])
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression."""

    expression = (
        expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**").replace("π", "pi")
    )
    if not expression:
        return "Error: expression must not be empty"
    if len(expression) > _MAX_EXPRESSION_CHARS:
        return f"Error: expression is longer than {_MAX_EXPRESSION_CHARS} characters"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return f"Error: could not parse expression {expression!r}"
    try:
        value = _eval_node(tree.body)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return f"Error: {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def current_time(timezone: str = "local") -> str:
    """Return the current time in the given IANA timezone."""

    timezone = timezone.strip().strip("\"'")
    if timezone.lower() in {"", "local"}:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S (local)")
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(timezone)).strftime(f"%Y-%m-%d %H:%M:%S ({timezone})")
    except Exception:
        return (
            f"Error: unknown timezone {timezone!r}. Use an IANA name such as "
            "Asia/Shanghai, or omit the argument for local time."
        )


def web_search(query: str) -> str:
    """Search the web through Tavily, SerpApi or ddgs (first available)."""

    query = query.strip().strip("\"'")
    if not query:
        return "Error: search query must not be empty"

    errors: list[str] = []

    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient

            response = TavilyClient(api_key=tavily_key).search(query=query, max_results=3)
            parts: list[str] = []
            if response.get("answer"):
                parts.append(f"Answer: {response['answer']}")
            for index, item in enumerate(response.get("results", [])[:3], 1):
                parts.append(f"[{index}] {item.get('title', '')}\n{item.get('content', '')[:200]}")
            if parts:
                return "\n\n".join(parts)
            errors.append("Tavily returned no results")
        except ImportError:
            errors.append("tavily-python is not installed")
        except Exception as exc:  # noqa: BLE001 - provider failure is non-fatal
            errors.append(f"Tavily failed: {exc}")

    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            import serpapi

            data = serpapi.GoogleSearch({"q": query, "api_key": serpapi_key, "num": 3}).get_dict()
            items = data.get("organic_results", [])[:3]
            if items:
                return "\n\n".join(
                    f"[{index}] {item.get('title', '')}\n{item.get('snippet', '')}"
                    for index, item in enumerate(items, 1)
                )
            errors.append("SerpApi returned no results")
        except ImportError:
            errors.append("google-search-results is not installed")
        except Exception as exc:  # noqa: BLE001 - provider failure is non-fatal
            errors.append(f"SerpApi failed: {exc}")

    ddgs_items: list[dict[str, Any]] | None = None
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        ddgs_items = list(DDGS(timeout=10).text(query, max_results=3))
    except ImportError:
        ddgs_items = None
    except Exception as exc:  # noqa: BLE001 - provider failure is non-fatal
        errors.append(f"ddgs failed: {exc}")

    if ddgs_items:
        return "\n\n".join(
            f"[{index}] {item.get('title', '')}\n{item.get('body', '') or item.get('snippet', '')}"
            for index, item in enumerate(ddgs_items, 1)
        )
    if ddgs_items is not None:
        errors.append("ddgs returned no results")

    if errors:
        return "Search failed: " + "; ".join(errors)
    return (
        "No search backend configured. Use one of:\n"
        "1. TAVILY_API_KEY + pip install tavily-python\n"
        "2. SERPAPI_API_KEY + pip install google-search-results\n"
        "3. pip install ddgs (free, no key)"
    )


def build_default_registry(*, max_output_chars: int = 4000) -> ToolRegistry:
    """Create a registry with the built-in tools."""

    registry = ToolRegistry(max_output_chars=max_output_chars)
    registry.register(
        ToolSpec(
            name="calculator",
            description=(
                "Evaluate an arithmetic expression. Supports + - * / // % **, "
                "sqrt, abs, round, sin, cos, tan, log, exp, floor, ceil, pi and e."
            ),
            func=calculator,
            parameters=(
                ToolParameter(
                    "expression",
                    "string",
                    "Arithmetic expression, for example (25+15)*3-8",
                ),
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="current_time",
            description="Return the current time in an IANA timezone.",
            func=current_time,
            parameters=(
                ToolParameter(
                    "timezone",
                    "string",
                    "IANA timezone such as Asia/Shanghai; omit for local time.",
                    required=False,
                    default="local",
                ),
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="web_search",
            description=(
                "Search the web and return short snippets. Requires "
                "TAVILY_API_KEY, SERPAPI_API_KEY, or the optional ddgs package."
            ),
            func=web_search,
            parameters=(ToolParameter("query", "string", "Search keywords."),),
        )
    )
    return registry

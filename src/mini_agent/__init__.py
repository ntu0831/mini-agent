"""MiniAgent public API."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_env
from .errors import (
    ConfigError,
    LLMError,
    MiniAgentError,
    ToolError,
    ToolValidationError,
)
from .tools import (
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_default_registry,
    calculator,
    current_time,
    web_search,
)

__all__ = [
    "Config",
    "ConfigError",
    "LLMError",
    "MiniAgentError",
    "ToolError",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolValidationError",
    "build_default_registry",
    "calculator",
    "current_time",
    "load_env",
    "web_search",
    "__version__",
]

"""MiniAgent public API."""

from __future__ import annotations

__version__ = "0.1.0"

from .agent import AgentEvent, MiniAgent, RunResult, Turn
from .config import Config, load_env
from .errors import ConfigError, LLMError, MiniAgentError, ToolError, ToolValidationError
from .llm import (
    ChatModel,
    LLMClient,
    LLMResponse,
    RetryPolicy,
    Usage,
    build_model,
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
from .trace import TraceRecorder

__all__ = [
    "AgentEvent",
    "ChatModel",
    "Config",
    "ConfigError",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "MiniAgent",
    "MiniAgentError",
    "RetryPolicy",
    "RunResult",
    "ToolError",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolValidationError",
    "TraceRecorder",
    "Turn",
    "Usage",
    "build_default_registry",
    "build_model",
    "calculator",
    "current_time",
    "load_env",
    "web_search",
    "__version__",
]

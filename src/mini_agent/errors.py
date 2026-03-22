"""Exception hierarchy for MiniAgent."""


class MiniAgentError(Exception):
    """Base class for all MiniAgent errors."""


class ConfigError(MiniAgentError):
    """Raised when configuration is missing or invalid."""


class LLMError(MiniAgentError):
    """Raised when a language-model request fails."""


class ToolError(MiniAgentError):
    """Raised for tool registration or execution problems."""


class ToolValidationError(ToolError):
    """Raised when tool arguments do not match the tool schema."""

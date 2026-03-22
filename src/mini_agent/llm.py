"""OpenAI-compatible language-model client with retry and usage accounting."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from .config import Config
from .errors import ConfigError, LLMError

ChatMessage = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage reported by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def add_usage(left: Usage, right: Usage) -> Usage:
    """Add two usage records (useful when aggregating ReAct steps)."""

    return Usage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A normalised model response."""

    text: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


class ChatModel(Protocol):
    """Structural interface implemented by real and offline models."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
    ) -> LLMResponse: ...


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with optional full jitter."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be greater than or equal to base_delay")

    def delay_for(self, attempt: int, *, rand: Callable[[], float] = random.random) -> float:
        """Return the delay after a failed attempt (1-based)."""

        if attempt < 1:
            raise ValueError("attempt must be at least 1")
        delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return float(delay * rand()) if self.jitter else float(delay)


_RETRYABLE_EXCEPTION_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "TimeoutError",
    "ConnectionError",
    "ConnectError",
    "ReadTimeout",
    "TimeoutException",
}
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429}


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(exc: BaseException) -> bool:
    """Classify provider errors as transient or terminal.

    Status codes are checked first because most OpenAI-compatible SDKs expose
    them on the exception. Exception class names provide a fallback for SDKs
    that do not.
    """

    status = _status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES or 500 <= status <= 599
    return any(cls.__name__ in _RETRYABLE_EXCEPTION_NAMES for cls in type(exc).__mro__)


class LLMClient:
    """Synchronous client for any OpenAI-compatible ``/chat/completions`` API."""

    def __init__(
        self,
        config: Config,
        *,
        client: Any | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rand: Callable[[], float] = random.random,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.retry = retry or RetryPolicy(max_attempts=config.max_retries)
        self._sleep = sleep
        self._rand = rand
        self._logger = logger or logging.getLogger(__name__)

        if client is None:
            if not config.api_key:
                raise ConfigError(
                    "No API key configured. Set MINI_AGENT_API_KEY (or LLM_API_KEY / "
                    "OPENAI_API_KEY) or use the offline model."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ConfigError(
                    "The 'openai' package is required for real model calls. "
                    "Install the project with 'pip install -e .' or use --mock."
                ) from exc
            client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        self._client = client

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send a chat request, retrying transient failures."""

        payload = [dict(message) for message in messages]
        last_error: BaseException | None = None
        attempt = 0

        for attempt in range(1, self.retry.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=cast(Any, payload),
                    temperature=(self.config.temperature if temperature is None else temperature),
                )
            except Exception as exc:
                last_error = exc
                if not is_retryable(exc) or attempt == self.retry.max_attempts:
                    break
                delay = self.retry.delay_for(attempt, rand=self._rand)
                self._logger.warning(
                    "LLM request failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt,
                    self.retry.max_attempts,
                    exc,
                    delay,
                )
                if delay > 0:
                    self._sleep(delay)
                continue

            latency_ms = (time.perf_counter() - started) * 1000.0
            return self._parse_response(response, latency_ms)

        raise LLMError(
            f"LLM request failed after {attempt} attempt(s): {last_error}"
        ) from last_error

    def _parse_response(self, response: Any, latency_ms: float) -> LLMResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("The model returned no choices")
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None) or ""
        usage = Usage(
            prompt_tokens=_usage_value(response, "prompt_tokens"),
            completion_tokens=_usage_value(response, "completion_tokens"),
            total_tokens=_usage_value(response, "total_tokens"),
        )
        return LLMResponse(
            text=str(text).strip(),
            model=str(getattr(response, "model", "") or self.config.model),
            usage=usage,
            latency_ms=latency_ms,
        )


def _usage_value(response: Any, name: str) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, Mapping):
        value = usage.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_model(config: Config) -> ChatModel:
    """Create the offline model or a real client from configuration."""

    if config.mock:
        from .mock import MockLLM

        return MockLLM()
    return LLMClient(config)

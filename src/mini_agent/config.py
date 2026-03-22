"""Configuration loading and validation.

All settings can be provided through environment variables. The ``MINI_AGENT_*``
names take precedence; common ``LLM_*`` and provider-specific variables are
accepted as fallbacks so the project works in existing OpenAI-compatible setups.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1/"

_TRUTHY = {"1", "true", "yes", "on"}
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def load_env() -> None:
    """Load a ``.env`` file if python-dotenv is available.

    ``load_dotenv`` searches the current working directory upwards, which keeps
    the CLI and the Streamlit app working when started from the project root.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional convenience dependency
        return
    load_dotenv()


def _first(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def _as_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _as_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _is_local(base_url: str) -> bool:
    lowered = base_url.lower()
    return any(host in lowered for host in _LOCAL_HOSTS)


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable runtime configuration.

    Use :meth:`from_env` for environment-based loading and :meth:`validate` to
    fail fast with actionable messages before any network call is made.
    """

    model: str = DEFAULT_MODEL
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 60.0
    temperature: float = 0.0
    max_steps: int = 5
    max_retries: int = 3
    max_observation_chars: int = 4000
    mock: bool = False
    trace_path: Path | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Config:
        """Build a configuration from environment variables.

        Precedence for each value is documented in ``README.md``. Invalid
        numeric values raise :class:`ConfigError` immediately.
        """

        env: Mapping[str, str] = os.environ if environ is None else environ

        model = _first(env, "MINI_AGENT_MODEL", "LLM_MODEL_ID", "OPENAI_MODEL") or DEFAULT_MODEL
        api_key = (
            _first(
                env,
                "MINI_AGENT_API_KEY",
                "LLM_API_KEY",
                "OPENAI_API_KEY",
                "MODELSCOPE_API_KEY",
            )
            or ""
        )
        base_url = _first(env, "MINI_AGENT_BASE_URL", "LLM_BASE_URL") or ""
        if not base_url:
            if env.get("MODELSCOPE_API_KEY") and not env.get("OPENAI_API_KEY"):
                base_url = MODELSCOPE_BASE_URL
            else:
                base_url = DEFAULT_BASE_URL
        if not api_key and _is_local(base_url):
            # Local OpenAI-compatible servers usually ignore the key, but SDKs
            # require a non-empty value.
            api_key = "local"

        timeout = _as_float(
            env,
            "MINI_AGENT_TIMEOUT",
            _as_float(env, "LLM_TIMEOUT", 60.0),
        )
        temperature = _as_float(env, "MINI_AGENT_TEMPERATURE", 0.0)
        max_steps = _as_int(env, "MINI_AGENT_MAX_STEPS", 5)
        max_retries = _as_int(env, "MINI_AGENT_MAX_RETRIES", 3)
        max_observation_chars = _as_int(env, "MINI_AGENT_MAX_OBSERVATION_CHARS", 4000)
        mock = (_first(env, "MINI_AGENT_MOCK", "AGENT_MOCK") or "").lower() in _TRUTHY

        trace_raw = _first(env, "MINI_AGENT_TRACE")
        trace_path = Path(trace_raw).expanduser() if trace_raw else None

        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_steps=max_steps,
            max_retries=max_retries,
            max_observation_chars=max_observation_chars,
            mock=mock,
            trace_path=trace_path,
        )

    def validate(self, *, require_credentials: bool = True) -> None:
        """Validate bounds and, optionally, that credentials are present."""

        if not self.model.strip():
            raise ConfigError("MINI_AGENT_MODEL must not be empty")
        if self.timeout <= 0:
            raise ConfigError("MINI_AGENT_TIMEOUT must be greater than 0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("MINI_AGENT_TEMPERATURE must be between 0 and 2")
        if self.max_steps < 1:
            raise ConfigError("MINI_AGENT_MAX_STEPS must be at least 1")
        if self.max_retries < 0:
            raise ConfigError("MINI_AGENT_MAX_RETRIES must be zero or greater")
        if self.max_observation_chars < 1:
            raise ConfigError("MINI_AGENT_MAX_OBSERVATION_CHARS must be at least 1")
        if require_credentials and not self.mock and not self.api_key:
            raise ConfigError(
                "No API key configured. Set MINI_AGENT_API_KEY (or LLM_API_KEY / "
                "OPENAI_API_KEY), point MINI_AGENT_BASE_URL at a local server, or "
                "run with --mock for the deterministic offline model."
            )

    def safe_dict(self) -> dict[str, Any]:
        """Return a serialisable view with the API key redacted."""

        return {
            "model": self.model,
            "api_key": "***" if self.api_key else "",
            "base_url": self.base_url,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "max_observation_chars": self.max_observation_chars,
            "mock": self.mock,
            "trace_path": str(self.trace_path) if self.trace_path else None,
        }

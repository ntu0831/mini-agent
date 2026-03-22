"""Configuration parsing and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent import Config, ConfigError


def test_defaults() -> None:
    config = Config.from_env({})
    assert config.model == "gpt-4o-mini"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == ""
    assert config.timeout == 60.0
    assert config.max_steps == 5
    assert config.max_retries == 3
    assert config.mock is False
    assert config.trace_path is None


def test_generic_llm_variables() -> None:
    config = Config.from_env(
        {
            "LLM_MODEL_ID": "qwen",
            "LLM_API_KEY": "sk-test",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_TIMEOUT": "12",
        }
    )
    assert config.model == "qwen"
    assert config.api_key == "sk-test"
    assert config.base_url == "https://example.test/v1"
    assert config.timeout == 12


def test_mini_agent_variables_take_precedence() -> None:
    config = Config.from_env(
        {
            "MINI_AGENT_MODEL": "preferred",
            "LLM_MODEL_ID": "fallback",
            "MINI_AGENT_API_KEY": "primary",
            "LLM_API_KEY": "secondary",
            "MINI_AGENT_BASE_URL": "https://primary.test/v1",
            "LLM_BASE_URL": "https://secondary.test/v1",
        }
    )
    assert config.model == "preferred"
    assert config.api_key == "primary"
    assert config.base_url == "https://primary.test/v1"


def test_openai_key_fallback() -> None:
    config = Config.from_env({"OPENAI_API_KEY": "sk-openai"})
    assert config.api_key == "sk-openai"
    assert config.base_url == "https://api.openai.com/v1"


def test_modelscope_key_sets_default_base_url() -> None:
    config = Config.from_env({"MODELSCOPE_API_KEY": "ms-key"})
    assert config.api_key == "ms-key"
    assert "modelscope" in config.base_url


def test_local_base_url_gets_placeholder_key() -> None:
    config = Config.from_env({"MINI_AGENT_BASE_URL": "http://localhost:11434/v1"})
    assert config.api_key == "local"


def test_mock_truthy_values() -> None:
    for value in ["1", "true", "YES", "on"]:
        assert Config.from_env({"MINI_AGENT_MOCK": value}).mock is True
    assert Config.from_env({"MINI_AGENT_MOCK": "false"}).mock is False


def test_trace_path_is_expanded() -> None:
    config = Config.from_env({"MINI_AGENT_TRACE": "traces/run.jsonl"})
    assert config.trace_path == Path("traces/run.jsonl")


def test_invalid_numbers_raise_config_error() -> None:
    with pytest.raises(ConfigError):
        Config.from_env({"MINI_AGENT_TIMEOUT": "not-a-number"})
    with pytest.raises(ConfigError):
        Config.from_env({"MINI_AGENT_MAX_STEPS": "many"})


def test_validate_requires_credentials() -> None:
    with pytest.raises(ConfigError, match="No API key"):
        Config().validate()
    Config(mock=True).validate()


def test_validate_bounds() -> None:
    with pytest.raises(ConfigError):
        Config(api_key="k", max_steps=0).validate()
    with pytest.raises(ConfigError):
        Config(api_key="k", max_retries=-1).validate()
    with pytest.raises(ConfigError):
        Config(api_key="k", timeout=0).validate()
    with pytest.raises(ConfigError):
        Config(api_key="k", temperature=3.0).validate()
    with pytest.raises(ConfigError):
        Config(api_key="k", max_observation_chars=0).validate()


def test_safe_dict_redacts_secret() -> None:
    config = Config(api_key="super-secret", trace_path=Path("run.jsonl"))
    safe = config.safe_dict()
    assert safe["api_key"] == "***"
    assert safe["trace_path"] == "run.jsonl"
    assert "super-secret" not in str(safe)

"""LLM client retry, usage and error-handling tests."""

from __future__ import annotations

from typing import Any

import pytest

from mini_agent import Config, ConfigError, LLMClient, LLMError, RetryPolicy, build_model
from mini_agent.llm import is_retryable


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Usage:
    prompt_tokens = 7
    completion_tokens = 3
    total_tokens = 10


class _Response:
    def __init__(self, content: str = "ok") -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()
        self.model = "fake-model"


class _EmptyResponse:
    choices: list[Any] = []
    usage = None
    model = "fake-model"


class _DictUsageResponse:
    def __init__(self) -> None:
        self.choices = [_Choice("mapped")]
        self.usage = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        self.model = "fake-model"


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _Completions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_kwargs = kwargs
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Chat:
    def __init__(self, outcomes: list[Any]) -> None:
        self.completions = _Completions(outcomes)


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.chat = _Chat(outcomes)


MESSAGES = [{"role": "user", "content": "hi"}]


def test_successful_call(config: Config) -> None:
    fake = FakeClient([_Response("hello")])
    client = LLMClient(config, client=fake)
    response = client.chat(MESSAGES)
    assert response.text == "hello"
    assert response.model == "fake-model"
    assert response.usage.prompt_tokens == 7
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 10
    assert response.latency_ms >= 0
    assert fake.chat.completions.calls == 1
    assert fake.chat.completions.last_kwargs["model"] == "test-model"
    assert fake.chat.completions.last_kwargs["temperature"] == 0.0


def test_temperature_override(config: Config) -> None:
    fake = FakeClient([_Response()])
    LLMClient(config, client=fake).chat(MESSAGES, temperature=0.7)
    assert fake.chat.completions.last_kwargs["temperature"] == 0.7


def test_retry_on_rate_limit(config: Config) -> None:
    fake = FakeClient([_StatusError(429), _Response("recovered")])
    sleeps: list[float] = []
    client = LLMClient(
        config,
        client=fake,
        retry=RetryPolicy(max_attempts=2, base_delay=1.0, jitter=False),
        sleep=sleeps.append,
    )
    response = client.chat(MESSAGES)
    assert response.text == "recovered"
    assert fake.chat.completions.calls == 2
    assert sleeps == [1.0]


def test_retry_on_timeout_exception(config: Config) -> None:
    fake = FakeClient([TimeoutError("slow"), _Response("ok")])
    client = LLMClient(
        config,
        client=fake,
        retry=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=False),
        sleep=lambda _: None,
    )
    assert client.chat(MESSAGES).text == "ok"
    assert fake.chat.completions.calls == 2


def test_non_retryable_error(config: Config) -> None:
    fake = FakeClient([_StatusError(401)])
    sleeps: list[float] = []
    client = LLMClient(
        config,
        client=fake,
        retry=RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False),
        sleep=sleeps.append,
    )
    with pytest.raises(LLMError, match="after 1 attempt"):
        client.chat(MESSAGES)
    assert fake.chat.completions.calls == 1
    assert sleeps == []


def test_retries_are_exhausted(config: Config) -> None:
    fake = FakeClient([_StatusError(503)])
    sleeps: list[float] = []
    client = LLMClient(
        config,
        client=fake,
        retry=RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False),
        sleep=sleeps.append,
    )
    with pytest.raises(LLMError, match="after 3 attempt"):
        client.chat(MESSAGES)
    assert fake.chat.completions.calls == 3
    assert sleeps == [1.0, 2.0]


def test_empty_choices(config: Config) -> None:
    client = LLMClient(config, client=FakeClient([_EmptyResponse()]))
    with pytest.raises(LLMError, match="no choices"):
        client.chat(MESSAGES)


def test_dict_usage_is_supported(config: Config) -> None:
    response = LLMClient(config, client=FakeClient([_DictUsageResponse()])).chat(MESSAGES)
    assert response.text == "mapped"
    assert response.usage.total_tokens == 3


def test_missing_api_key() -> None:
    with pytest.raises(ConfigError, match="No API key"):
        LLMClient(Config(api_key=""))


def test_retry_policy_validation() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=-1)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=5, max_delay=1)


def test_backoff_delay_calculation() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=4.0, jitter=False)
    assert [policy.delay_for(attempt) for attempt in range(1, 5)] == [1.0, 2.0, 4.0, 4.0]
    jittered = RetryPolicy(base_delay=1.0, jitter=True)
    assert jittered.delay_for(2, rand=lambda: 0.25) == 0.5
    with pytest.raises(ValueError):
        policy.delay_for(0)


def test_retryable_classification() -> None:
    assert is_retryable(_StatusError(429)) is True
    assert is_retryable(_StatusError(500)) is True
    assert is_retryable(_StatusError(401)) is False
    assert is_retryable(TimeoutError()) is True
    assert is_retryable(ValueError("bad")) is False


def test_build_model_returns_mock() -> None:
    model = build_model(Config(mock=True))
    assert getattr(model, "name", "") == "mock"

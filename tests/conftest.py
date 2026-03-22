"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from mini_agent import Config


@pytest.fixture
def config() -> Config:
    return Config(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999/v1",
    )

"""Web search provider fallback tests (no network access)."""

from __future__ import annotations

import sys
import types

import pytest

from mini_agent import web_search


def _block(monkeypatch: pytest.MonkeyPatch, *module_names: str) -> None:
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, None)


def test_no_backend_returns_configuration_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    _block(monkeypatch, "ddgs", "duckduckgo_search")

    output = web_search("python")
    assert "No search backend configured" in output
    assert "TAVILY_API_KEY" in output


def test_tavily_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    _block(monkeypatch, "tavily", "ddgs", "duckduckgo_search")

    output = web_search("python")
    assert "Search failed" in output
    assert "tavily-python is not installed" in output


def test_tavily_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTavilyClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"

        def search(self, *, query: str, max_results: int) -> dict[str, object]:
            assert query == "python"
            assert max_results == 3
            return {
                "answer": "A programming language",
                "results": [{"title": "Python", "content": "Python is a language."}],
            }

    module = types.ModuleType("tavily")
    module.TavilyClient = FakeTavilyClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tavily", module)
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    output = web_search("python")
    assert "A programming language" in output
    assert "[1] Python" in output


def test_serpapi_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGoogleSearch:
        def __init__(self, params: dict[str, object]) -> None:
            assert params["q"] == "python"

        def get_dict(self) -> dict[str, object]:
            return {"organic_results": [{"title": "Python", "snippet": "A programming language."}]}

    module = types.ModuleType("serpapi")
    module.GoogleSearch = FakeGoogleSearch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serpapi", module)
    monkeypatch.setitem(sys.modules, "tavily", None)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    output = web_search("python")
    assert "[1] Python" in output
    assert "A programming language." in output

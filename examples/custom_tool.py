"""Register a custom tool and call it directly.

Run with::

    python examples/custom_tool.py

This example does not need an API key. To expose the tool to a model, pass the
same registry to ``MiniAgent``::

    agent = MiniAgent(build_model(Config(mock=True)), registry)
"""

from __future__ import annotations

from mini_agent import ToolParameter, ToolRegistry

registry = ToolRegistry()


@registry.tool(
    "word_count",
    "Count characters and whitespace-separated words.",
    (ToolParameter("text", "string", "Text to analyse."),),
)
def word_count(text: str) -> str:
    return f"characters={len(text)} words={len(text.split())}"


def main() -> None:
    result = registry.execute("word_count", "hello mini agent")
    print(result.content)
    print()
    print(registry.describe())


if __name__ == "__main__":
    main()

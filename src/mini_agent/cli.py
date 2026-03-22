"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import __version__
from .agent import AgentEvent, MiniAgent, RunResult
from .config import Config, load_env
from .errors import ConfigError, LLMError, MiniAgentError
from .llm import build_model
from .tools import build_default_registry
from .trace import TraceRecorder

_COMMANDS = {"run", "tools", "config"}
_HELP_FLAGS = {"-h", "--help", "--version"}

_BANNER = f"""\
MiniAgent {__version__} - a small, auditable ReAct agent
"""

_HELP = """\
Commands:
  /help      Show this help
  /tools     List available tools
  /config    Show effective configuration
  /history   Show conversation history
  /clear     Clear conversation history
  /exit      Exit
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description=("A small ReAct agent with tool calling, retries, tracing and tests."),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Run one question or start an interactive session"
    )
    run_parser.add_argument("-q", "--question", help="Question to answer once")
    run_parser.add_argument(
        "--mock", action="store_true", help="Use the deterministic offline model"
    )
    run_parser.add_argument("--max-steps", type=int, help="Override the step budget")
    run_parser.add_argument("--model", help="Override the model name")
    run_parser.add_argument("--trace", type=Path, help="Append a JSONL trace to this file")
    run_parser.add_argument("--json", action="store_true", help="Print a machine-readable result")
    run_parser.add_argument("--quiet", action="store_true", help="Do not print reasoning events")

    tools_parser = subparsers.add_parser("tools", help="List registered tools")
    tools_parser.add_argument("--json", action="store_true", help="Print machine-readable output")

    subparsers.add_parser("config", help="Print the effective configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""

    load_env()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["run"]
    elif arguments[0] not in _COMMANDS and arguments[0] not in _HELP_FLAGS:
        arguments.insert(0, "run")

    parser = build_parser()
    args = parser.parse_args(arguments)

    try:
        if args.command == "tools":
            return _command_tools(json_output=args.json)
        if args.command == "config":
            return _command_config()
        return _command_run(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1
    except MiniAgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def _command_config() -> int:
    config = Config.from_env()
    print(json.dumps(config.safe_dict(), indent=2, ensure_ascii=False))
    return 0


def _command_tools(*, json_output: bool) -> int:
    registry = build_default_registry()
    if json_output:
        payload = [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": [
                    {
                        "name": parameter.name,
                        "type": parameter.type,
                        "description": parameter.description,
                        "required": parameter.required,
                        "default": parameter.default,
                    }
                    for parameter in spec.parameters
                ],
            }
            for spec in registry.specs()
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(registry.describe())
    return 0


def _command_run(args: argparse.Namespace) -> int:
    config = Config.from_env()
    if args.model:
        config = replace(config, model=args.model)
    if args.max_steps is not None:
        config = replace(config, max_steps=args.max_steps)
    if args.mock:
        config = replace(config, mock=True)
    if args.trace:
        config = replace(config, trace_path=args.trace)
    config.validate()

    with TraceRecorder(config.trace_path) as trace:
        model = build_model(config)
        registry = build_default_registry(max_output_chars=config.max_observation_chars)
        agent = MiniAgent(
            model,
            registry,
            max_steps=config.max_steps,
            max_observation_chars=config.max_observation_chars,
            trace=trace,
        )

        if args.question:
            return _run_once(agent, args.question, json_output=args.json, quiet=args.quiet)
        return _interactive(agent, config, quiet=args.quiet)


def _run_once(
    agent: MiniAgent,
    question: str,
    *,
    json_output: bool,
    quiet: bool,
) -> int:
    callback = None if (json_output or quiet) else _print_event
    result = agent.run(question, on_event=callback)
    if json_output:
        print(json.dumps(_result_dict(result), ensure_ascii=False))
    else:
        print(f"\nAnswer: {result.answer}")
        print(_stats(result))
    return 0


def _interactive(agent: MiniAgent, config: Config, *, quiet: bool) -> int:
    print(_BANNER)
    mode = "offline mock model" if config.mock else f"{config.model} @ {config.base_url}"
    print(f"Model: {mode}")
    print(f"Tools: {', '.join(agent.tools.names())}")
    print("Type /help for commands.\n")

    while True:
        try:
            line = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(line, agent, config):
                return 0
            continue

        try:
            result = agent.run(line, on_event=None if quiet else _print_event)
        except (LLMError, ConfigError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        print(f"\nAssistant: {result.answer}")
        print(_stats(result))


def _handle_command(line: str, agent: MiniAgent, config: Config) -> bool:
    command = line.lower()
    if command in {"/exit", "/quit", "/q"}:
        print("Bye.")
        return False
    if command == "/help":
        print(_HELP)
    elif command == "/tools":
        print(agent.tools.describe())
    elif command == "/config":
        print(json.dumps(config.safe_dict(), indent=2, ensure_ascii=False))
    elif command == "/history":
        _print_history(agent)
    elif command == "/clear":
        agent.clear_history()
        print("Conversation history cleared.")
    else:
        print(f"Unknown command {line}. Type /help for help.")
    return True


def _print_history(agent: MiniAgent) -> None:
    if not agent.dialogue:
        print("(no conversation history)")
        return
    for index, turn in enumerate(agent.dialogue, 1):
        print(f"\n[{index}] You: {turn.question}")
        print(f"    Assistant: {turn.answer}")


def _print_event(event: AgentEvent) -> None:
    if event.type == "step":
        print(f"\n-- {event.content} --")
    elif event.type == "thought":
        print(f"Thought: {event.content}")
    elif event.type == "action":
        print(f"Action: {event.content}")
    elif event.type == "observation":
        content = event.content
        if len(content) > 1000:
            content = content[:1000] + "..."
        print(f"Observation: {content}")
    elif event.type == "error":
        print(f"Warning: {event.content}")
    # "final" is printed by the caller as part of the result.


def _stats(result: RunResult) -> str:
    return (
        f"[{result.steps} steps, {result.tool_calls} tool calls, "
        f"{result.duration_ms:.0f} ms, {result.stopped_reason}]"
    )


def _result_dict(result: RunResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "duration_ms": round(result.duration_ms, 2),
        "stopped_reason": result.stopped_reason,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }

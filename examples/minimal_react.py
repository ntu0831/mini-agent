"""A self-contained ReAct loop in ~130 lines.

This file is intentionally independent of the ``mini_agent`` package. It exists
as a teaching/reference example: no registry, no tracing, no retries, just the
loop. The library version is the one used in production and tests.

Run with::

    python examples/minimal_react.py "Calculate (25+15)*3-8"

Environment variables: ``LLM_MODEL_ID``, ``LLM_API_KEY``, ``LLM_BASE_URL``.
"""

from __future__ import annotations

import ast
import operator
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

PROMPT = """You are a ReAct agent. You have one tool:

- calculator(expression): evaluate an arithmetic expression.

Reply with exactly two lines:
Thought: your reasoning
Action: calculator[expression]    or    Finish[answer]

Question: {question}
History:
{history}"""


def calculate(expression: str) -> str:
    """Evaluate + - * / // % ** and sqrt/abs/round with a numeric AST whitelist."""
    import math

    binops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    funcs = {"sqrt": math.sqrt, "abs": abs, "round": round}

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in binops:
            return float(binops[type(node.op)](eval_node(node.left), eval_node(node.right)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in funcs:
                raise ValueError(f"function {node.func.id!r} is not allowed")
            return float(funcs[node.func.id](*[eval_node(arg) for arg in node.args]))
        raise ValueError("unsupported expression")

    try:
        value = eval_node(ast.parse(expression, mode="eval").body)
    except Exception as exc:
        return f"Error: {exc}"
    return str(int(value)) if value.is_integer() else str(value)


def parse(response: str) -> tuple[str, str]:
    response = response.strip().strip("`")
    action = response.split("Action:")[-1].strip()
    return response, action


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]) or input("Question: ")
    model = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )

    history = []
    for step in range(1, 6):
        prompt = PROMPT.format(question=question, history="\n".join(history) or "(none)")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        _, action = parse(text)
        print(f"\nStep {step}: {action}")

        if action.lower().startswith("finish"):
            print("Answer:", action[action.find("[") + 1 : action.rfind("]")])
            return

        match = re.match(r"calculator\[(.*)\]", action)
        if not match:
            history.append("Observation: invalid action, use calculator[...] or Finish[...]")
            continue

        observation = calculate(match.group(1))
        print("Observation:", observation)
        history.extend([f"Action: {action}", f"Observation: {observation}"])

    print("Stopped: step budget exhausted")


if __name__ == "__main__":
    main()

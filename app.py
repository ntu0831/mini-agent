"""Streamlit interface for MiniAgent.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from mini_agent import (
    AgentEvent,
    Config,
    MiniAgent,
    MiniAgentError,
    RunResult,
    build_default_registry,
    build_model,
    load_env,
)

load_env()
st.set_page_config(page_title="MiniAgent", page_icon="🤖", layout="centered")

EXAMPLES = [
    "Calculate (25+15)*3-8",
    "What time is it?",
    "Search for what Python is",
]


def build_agent(config: Config) -> MiniAgent:
    return MiniAgent(
        build_model(config),
        build_default_registry(max_output_chars=config.max_observation_chars),
        max_steps=config.max_steps,
        max_observation_chars=config.max_observation_chars,
    )


def result_stats(result: RunResult) -> str:
    return (
        f"{result.steps} steps · {result.tool_calls} tool calls · "
        f"{result.duration_ms:.0f} ms · {result.stopped_reason}"
    )


def render_events(events: list[AgentEvent]) -> None:
    for event in events:
        if event.type == "step":
            st.markdown(f"**-- {event.content} --**")
        elif event.type == "thought":
            st.markdown(f"**Thought:** {event.content}")
        elif event.type == "action":
            st.markdown(f"**Action:** `{event.content}`")
        elif event.type == "observation":
            content = event.content
            if len(content) > 1500:
                content = content[:1500] + "..."
            st.markdown("**Observation:**")
            st.code(content, language="text")
        elif event.type == "error":
            st.warning(event.content)


def render_live_event(event: AgentEvent) -> None:
    if event.type == "step":
        st.markdown(f"**-- {event.content} --**")
    elif event.type == "thought":
        st.markdown(f"**Thought:** {event.content}")
    elif event.type == "action":
        st.markdown(f"**Action:** `{event.content}`")
    elif event.type == "observation":
        content = event.content
        if len(content) > 600:
            content = content[:600] + "..."
        st.code(content, language="text")
    elif event.type == "error":
        st.warning(event.content)


if "config" not in st.session_state:
    try:
        st.session_state.config = Config.from_env()
    except MiniAgentError as exc:
        st.error(str(exc))
        st.stop()

config: Config = st.session_state.config

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.get("agent") is None and not st.session_state.get("init_error"):
    try:
        st.session_state.agent = build_agent(config)
        st.session_state.init_error = None
    except MiniAgentError as exc:
        st.session_state.agent = None
        st.session_state.init_error = str(exc)

agent: MiniAgent | None = st.session_state.get("agent")


with st.sidebar:
    st.header("Settings")

    mock_mode = st.checkbox("Offline mock model (no API key)", value=config.mock)
    if mock_mode != config.mock:
        st.session_state.config = replace(config, mock=mock_mode)
        st.session_state.agent = None
        st.session_state.init_error = None
        st.rerun()

    if config.mock:
        st.success("MockLLM · deterministic offline model")
    else:
        st.markdown(f"**Model:** `{config.model}`")
        st.markdown(f"**Base URL:** `{config.base_url}`")
        st.caption("API key detected" if config.api_key else "No API key detected")

    st.divider()
    max_steps = st.slider("Max reasoning steps", min_value=1, max_value=10, value=config.max_steps)
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(config.temperature),
        step=0.1,
    )
    if temperature != config.temperature:
        st.session_state.config = replace(config, temperature=temperature)
        st.session_state.agent = None
        st.session_state.init_error = None
        st.rerun()
    if agent is not None:
        agent.max_steps = max_steps

    st.divider()
    st.markdown("**Tools**")
    if agent is not None:
        for name in agent.tools.names():
            st.markdown(f"- `{name}`")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        if agent is not None:
            agent.clear_history()
        st.rerun()

    st.divider()
    st.markdown("**Examples**")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending_prompt = example


st.title("🤖 MiniAgent")
st.caption("A small, auditable ReAct agent runtime with tools, retries and tracing")

if st.session_state.get("init_error"):
    st.error(f"Agent initialization failed: {st.session_state.init_error}")
    st.info(
        "Copy `.env.example` to `.env` and set MINI_AGENT_API_KEY / "
        "MINI_AGENT_BASE_URL / MINI_AGENT_MODEL, then restart the app. "
        "You can also enable the offline mock model in the sidebar."
    )

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("events"):
            with st.expander("Reasoning trace", expanded=False):
                render_events(message["events"])
        st.markdown(message["content"])
        if message.get("stats"):
            st.caption(message["stats"])

prompt = st.chat_input("Ask a question, e.g. Calculate (25+15)*3-8")
if not prompt:
    prompt = st.session_state.pop("pending_prompt", None)

if prompt:
    if agent is None:
        st.error("The agent is not initialized. Configure a model or enable the mock mode.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    events: list[AgentEvent] = []
    with st.chat_message("assistant", avatar="🤖"):
        with st.status("Reasoning...", expanded=True) as status:
            try:
                result = agent.run(
                    prompt,
                    on_event=lambda event: (
                        events.append(event),
                        render_live_event(event),
                    ),
                )
                status.update(label="Done", state="complete", expanded=False)
                answer = result.answer
                stats: str | None = result_stats(result)
            except MiniAgentError as exc:
                answer = f"Error: {exc}"
                stats = None
                status.update(label="Failed", state="error", expanded=True)
            except Exception as exc:  # noqa: BLE001 - keep the UI alive
                answer = f"Unexpected error: {exc}"
                stats = None
                status.update(label="Failed", state="error", expanded=True)
        st.markdown(answer)
        if stats:
            st.caption(stats)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "events": events,
            "stats": stats,
        }
    )

# Design notes

Design decisions and trade-offs in MiniAgent.

## 001: Text ReAct protocol instead of native function calling

**Context.** Providers that expose an OpenAI-compatible endpoint do not all
support the `tools` parameter or structured tool calls. A portable agent should
work with a plain chat completion endpoint.

**Decision.** The agent uses a text contract: `Thought:` followed by either
`Action: tool[argument]` or `Finish[answer]`. The prompt lists available tools
and their parameter signatures.

**Consequences.**

- Works against any chat-completions endpoint, including small local models.
- Requires reliable instruction following; weak models produce malformed output
  more often.
- Parsing is a small, testable surface. The malformed-output path feeds a format
  reminder back to the model instead of aborting.
- Native function calling can be added later as another `ChatModel`/agent
  policy without changing the tool registry.

## 002: Synchronous, one tool call per step

**Context.** LLM calls and tools are I/O-bound, so async could improve
throughput. But concurrency also complicates cancellation, tracing and tests.

**Decision.** Keep the runtime synchronous and execute exactly one tool call per
step. The tool registry is not concurrent.

**Consequences.**

- The control flow is easy to read, debug and trace.
- Multi-tool steps require multiple model turns, which costs tokens.
- Users who need concurrency can run multiple agents in separate threads or
  processes; no global state is shared.

## 003: Retry only transient provider failures

**Context.** Retrying every error hides configuration mistakes and can multiply
cost. Retrying nothing makes the agent fragile under rate limits.

**Decision.** Classify by HTTP status first: 408, 409, 425, 429 and 5xx are
retryable; other 4xx are terminal. If no status is exposed, fall back to
exception class names (`RateLimitError`, `APITimeoutError`,
`APIConnectionError`, `InternalServerError`, `TimeoutError`,
`ConnectionError`). Backoff is exponential with optional full jitter and a
hard cap.

**Consequences.**

- Rate limits and transient outages recover without user intervention.
- Authentication and validation errors fail fast with a typed `LLMError`.
- A provider that returns non-standard errors may be classified incorrectly;
  the classifier is centralized in `is_retryable` and easy to extend.

## 004: Repeated-action guard

**Context.** Models can get stuck calling the same tool with the same argument.
A step budget eventually stops them, but wastes the remaining turns.

**Decision.** Canonicalize each action as `lower(tool)[trimmed argument]` and
count executions per run. With `loop_threshold=2`, the second execution adds a
warning observation; the third aborts with `stopped_reason="loop"`.

**Consequences.**

- Infinite loops end early and visibly.
- Repeating a tool with a different argument is allowed and common (for example,
  searching twice with different queries).
- A model that legitimately needs three identical calls must raise the
  threshold; this is documented and configurable.

## 005: Truncate observations twice

**Context.** A search result or stack trace can exceed the model's context
window. Letting tools return unbounded text also makes traces and UI expensive.

**Decision.** `ToolRegistry` truncates each tool output to
`max_output_chars` (default 4000). `MiniAgent` applies a second limit,
`max_observation_chars`, before adding the observation to history. Both markers
state how many characters were omitted.

**Consequences.**

- Context growth is bounded per step.
- The model may miss the end of a long document; callers can raise the limits
  or add a summarization tool.
- Two limits allow a tool-level policy that is stricter than the agent's.

## 006: Opt-in JSONL tracing

**Context.** Debugging a failed run from console output is painful, but always
writing prompts to disk creates a privacy hazard.

**Decision.** Tracing is disabled by default. `--trace` writes one JSON object
per line with a shared `run_id`; API keys are never included.

**Consequences.**

- Traces are easy to `grep`, stream, or load into a notebook.
- Traces may contain user data and should be treated as sensitive (see
  `SECURITY.md`).
- The format is append-only and schema-light; adding fields is backward
  compatible.

## 007: In-memory conversation history

**Context.** Long-term memory needs storage, schema and privacy decisions that
are out of scope for a small runtime.

**Decision.** Each `MiniAgent` keeps a list of `Turn` objects. The last five
turns are rendered into the prompt; `clear_history()` removes them.

**Consequences.**

- No accidental persistence of user conversations.
- Callers that need durable memory can persist `agent.dialogue` themselves or
  inject retrieved context through the system prompt.
- Prompt size is bounded by the five-turn window and per-field truncation.

## 008: Single-agent design without orchestration

**Context.** Multi-agent frameworks add routing, shared state and failure
propagation. They also make the execution path harder to audit.

**Decision.** Ship one agent. Composition is left to callers: run several
agents, or wrap a `MiniAgent` as a tool in another registry.

**Consequences.**

- The core stays readable and testable.
- Complex workflows need explicit caller code, which is clearer than an
  implicit planner.

## 009: Untrusted tool output

**Context.** Web search results and arbitrary tool output can contain prompt
injection attempts.

**Decision.** Observations are labelled, truncated and fed back as data. The
system prompt says not to invent observations. The calculator uses an explicit
AST whitelist. There is no automatic approval step for side-effecting tools
because the built-ins are read-only.

**Consequences.**

- The built-in tool set is relatively safe.
- Adding write-capable tools requires application-level authorization and an
  approval flow; the runtime does not provide one.

## Future work

- Native function-calling adapter with the same tool registry.
- Optional async execution and cancellation.
- Pluggable memory/retrieval hook.
- OpenTelemetry exporter in addition to JSONL.
- Property-based tests for the parser.

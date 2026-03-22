# Security Policy

## Supported versions

Security fixes are provided for the latest minor release on the `main` branch.
As the project is pre-1.0, older minor versions may not receive backports.

## Reporting a vulnerability

Please **do not** open a public issue. Use GitHub's private security advisory
feature for this repository (Security → Advisories → Report a vulnerability).
If advisories are unavailable, open a minimal issue asking a maintainer for a
private channel, without including exploit details. Include:

- A description of the issue and its impact.
- Reproduction steps or a proof of concept.
- Affected version/commit and Python version.
- Any suggested mitigation.

We aim to acknowledge reports within a few business days and will coordinate a
disclosure timeline with you.

## Threat model

MiniAgent is a small runtime that connects a model to tools. It is **not** a
sandbox and does not provide tenant isolation. Review the following before
deploying it:

### Prompt injection and untrusted tool output

Web pages and tool responses are untrusted data. A malicious page can contain
instructions that try to redirect the model. Current mitigations:

- Tool observations are truncated and clearly labelled.
- The system prompt instructs the model not to invent observations.
- The calculator only accepts a numeric AST whitelist.

They are not sufficient for hostile content. If you expose `web_search` to an
untrusted audience, add allow-lists, content sanitisation or a human approval
step for side-effecting tools.

### API keys

Keys are read from environment variables and are never written to traces. Do not
commit `.env`; it is ignored by git. Prefer a secret manager in deployed
environments. `mini-agent config` and the Streamlit sidebar print only a redacted
view.

### Trace files

Traces intentionally contain questions, thoughts, actions and observations.
They may include personal data or secrets copied into prompts. Treat trace files
as sensitive, restrict access, and redact them before sharing in a bug report.

### Local model servers

`MINI_AGENT_BASE_URL` may point at any host. Only connect to servers you trust:
the client sends your prompts and API key to that endpoint. The placeholder key
`local` is used for localhost addresses when no key is configured.

### Calculator

The calculator evaluates a restricted AST and rejects imports, attribute access,
comprehensions and oversized exponents. It is a convenience tool, not a
general-purpose execution sandbox. Do not route untrusted arbitrary code through
it.

### Dependency and supply chain

Runtime dependencies are declared in `pyproject.toml`. CI installs from PyPI and
runs the test suite without network access at test time. Pin or vendor
dependencies if your environment requires reproducible builds.

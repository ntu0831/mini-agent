# Contributing to MiniAgent

Thanks for taking the time to contribute. This project values small, focused
changes that keep the control loop easy to read and test.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,web]"
pre-commit install   # optional, enables the configured hooks
```

Run the full local gate before opening a pull request:

```bash
make lint
make typecheck
make test
```

## What makes a good change

- **Keep the loop explicit.** New behaviour should be visible in
  `src/mini_agent/agent.py` or a clearly named collaborator, not hidden in
  decorators or implicit hooks.
- **Add tests for behaviour, not implementation.** Use `ScriptedModel` for agent
  tests and injected fakes for the LLM client; tests must not require an API key
  or network access.
- **Preserve failure semantics.** Tool validation failures are observations for
  the model. Programming errors should raise. Do not silently swallow exceptions
  unless the caller explicitly opted into a fallback.
- **Document public API changes.** Update the README and `CHANGELOG.md` when a
  CLI flag, environment variable, event or result field changes.

## Pull request checklist

- [ ] `make lint` passes.
- [ ] `make typecheck` passes (mypy strict).
- [ ] `make test` passes and coverage stays at or above 85%.
- [ ] New behaviour has tests.
- [ ] User-facing docs/CHANGELOG are updated.
- [ ] No secrets, API keys or personal data are committed.

## Code style

- Python 3.10+, type annotations on all public and internal functions.
- Ruff for linting and formatting (`line-length = 100`).
- Prefer standard library and small, explicit functions.
- Comments explain *why*, not *what*.

## Reporting bugs

Please include:

1. MiniAgent version (`mini-agent --version`).
2. Python version and operating system.
3. The provider/model and whether `--mock` reproduces the issue.
4. A trace file (`--trace`) with secrets removed, when possible.
5. The smallest prompt or code snippet that reproduces the problem.

## Security

Do not open a public issue for vulnerabilities. Follow [SECURITY.md](SECURITY.md).

"""Append-only JSONL tracing for reproducible agent runs."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from .errors import ConfigError


class TraceRecorder:
    """Write structured events to a JSONL file.

    Tracing is opt-in because thoughts, actions and observations may contain
    user data. Passing ``path=None`` yields a zero-cost no-op recorder.
    """

    def __init__(
        self,
        path: str | Path | None,
        *,
        run_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path).expanduser() if path else None
        self.run_id = run_id or uuid.uuid4().hex
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._file: TextIO | None = None

        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("a", encoding="utf-8")
            except OSError as exc:
                raise ConfigError(f"Cannot open trace file {self.path}: {exc}") from exc

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def emit(self, event: str, **payload: Any) -> None:
        """Append one event. No-op when tracing is disabled."""

        if self._file is None:
            return
        record = {
            "ts": self._clock().isoformat(),
            "run_id": self.run_id,
            "event": event,
            **payload,
        }
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TraceRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

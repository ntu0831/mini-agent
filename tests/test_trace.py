"""Trace recorder tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mini_agent import ConfigError, TraceRecorder


def test_disabled_recorder_is_a_noop() -> None:
    recorder = TraceRecorder(None)
    assert recorder.enabled is False
    recorder.emit("run_start", question="ignored")
    recorder.close()
    recorder.close()


def test_writes_jsonl_and_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trace.jsonl"
    fixed = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    recorder = TraceRecorder(path, run_id="run-1", clock=lambda: fixed)
    assert recorder.enabled is True

    recorder.emit("run_start", question="hi")
    recorder.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {
            "ts": fixed.isoformat(),
            "run_id": "run-1",
            "event": "run_start",
            "question": "hi",
        }
    ]


def test_context_manager_closes_file(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path, run_id="run-2") as recorder:
        recorder.emit("run_end", answer="ok")
    assert recorder.enabled is False

    record = json.loads(path.read_text().strip())
    assert record["run_id"] == "run-2"
    assert record["answer"] == "ok"


def test_directory_path_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot open trace file"):
        TraceRecorder(tmp_path)

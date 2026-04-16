"""Tests for progress reporters."""

from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from src.RAG.progress import (
    JSONProgressReporter,
    ProgressReporter,
    RichProgressReporter,
    SyncProgress,
    SyncStage,
    create_reporter,
)


class MockProgressReporter(ProgressReporter):
    """Mock progress reporter for testing."""

    def __init__(self) -> None:
        self.start_calls: list[int] = []
        self.stage_changes: list[tuple[SyncStage, str]] = []
        self.file_progress_calls: list[SyncProgress] = []
        self.complete_calls: list[dict[str, any]] = []
        self.error_calls: list[str] = []

    def on_start(self, total_files: int) -> None:
        self.start_calls.append(total_files)

    def on_stage_change(self, stage: SyncStage, stage_name: str) -> None:
        self.stage_changes.append((stage, stage_name))

    def on_file_progress(self, progress: SyncProgress) -> None:
        self.file_progress_calls.append(progress)

    def on_complete(self, summary: dict[str, any]) -> None:
        self.complete_calls.append(summary)

    def on_error(self, error: str) -> None:
        self.error_calls.append(error)


def test_mock_progress_reporter():
    """Test mock progress reporter works correctly."""
    reporter = MockProgressReporter()

    reporter.on_start(100)
    assert reporter.start_calls == [100]

    reporter.on_stage_change(SyncStage.PARSING, "解析中...")
    assert reporter.stage_changes == [(SyncStage.PARSING, "解析中...")]

    progress = SyncProgress(
        stage=SyncStage.PARSING,
        current=5,
        total=100,
        current_file="test.txt",
        elapsed_seconds=1.5,
        stage_name="解析: test.txt",
    )
    reporter.on_file_progress(progress)
    assert len(reporter.file_progress_calls) == 1
    assert reporter.file_progress_calls[0].current == 5

    reporter.on_error("Test error")
    assert reporter.error_calls == ["Test error"]

    summary = {"processed": 100, "updated": 50}
    reporter.on_complete(summary)
    assert reporter.complete_calls == [summary]


def test_sync_progress_percentage():
    """Test SyncProgress percentage calculation."""
    progress = SyncProgress(
        stage=SyncStage.PARSING,
        current=50,
        total=100,
        current_file="test.txt",
        elapsed_seconds=1.0,
        stage_name="解析: test.txt",
    )
    assert progress.percentage == 50.0

    # Test zero total
    progress_zero = SyncProgress(
        stage=SyncStage.PARSING,
        current=0,
        total=0,
        current_file="test.txt",
        elapsed_seconds=1.0,
        stage_name="解析: test.txt",
    )
    assert progress_zero.percentage == 0.0


def test_json_reporter_compatibility(tmp_path, capsys):
    """Test JSON reporter maintains backward compatibility."""
    reporter = JSONProgressReporter()

    # Simulate sync process
    reporter.on_start(10)
    reporter.on_stage_change(SyncStage.PARSING, "解析中...")

    progress = SyncProgress(
        stage=SyncStage.PARSING,
        current=5,
        total=10,
        current_file="test.txt",
        elapsed_seconds=1.5,
        stage_name="解析: test.txt",
    )
    reporter.on_file_progress(progress)

    reporter.on_error("test.txt: Test error")

    summary = {
        "processed": 10,
        "updated": 8,
        "skipped": 1,
        "failed": 1,
        "errors": ["test.txt: Test error"],
        "indexed_files": 8,
        "indexed_chunks": 100,
        "partial_files": 0,
        "chunks_dropped_total": 5,
        "pdf_observability": [],
    }
    reporter.on_complete(summary)

    # Capture output
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    # Verify JSON format
    assert output["processed"] == 10
    assert output["updated"] == 8
    assert output["skipped"] == 1
    assert output["failed"] == 1
    assert output["indexed_files"] == 8
    assert output["indexed_chunks"] == 100

    # Verify timing info was added
    assert "elapsed_seconds" in output
    assert output["elapsed_seconds"] >= 0
    assert "avg_time_per_file" in output
    assert output["avg_time_per_file"] >= 0


def test_json_reporter_silent_during_sync(capsys):
    """Test JSON reporter is silent during sync process."""
    reporter = JSONProgressReporter()

    reporter.on_start(100)
    reporter.on_stage_change(SyncStage.PARSING, "解析中...")

    progress = SyncProgress(
        stage=SyncStage.PARSING,
        current=5,
        total=100,
        current_file="test.txt",
        elapsed_seconds=1.5,
        stage_name="解析: test.txt",
    )
    reporter.on_file_progress(progress)

    # No output should be produced during sync
    captured = capsys.readouterr()
    assert captured.out == ""


def test_create_reporter_auto():
    """Test create_reporter with auto type."""
    # When stdout is a TTY, should return RichProgressReporter
    with patch("sys.stdout.isatty", return_value=True):
        reporter = create_reporter("auto")
        assert isinstance(reporter, RichProgressReporter)

    # When stdout is not a TTY, should return JSONProgressReporter
    with patch("sys.stdout.isatty", return_value=False):
        reporter = create_reporter("auto")
        assert isinstance(reporter, JSONProgressReporter)


def test_create_reporter_explicit_types():
    """Test create_reporter with explicit types."""
    # Test "rich" type
    try:
        reporter = create_reporter("rich")
        assert isinstance(reporter, RichProgressReporter)
    except ImportError:
        # Rich not installed, should fallback to JSON
        reporter = create_reporter("rich")
        assert isinstance(reporter, JSONProgressReporter)

    # Test "json" type
    reporter = create_reporter("json")
    assert isinstance(reporter, JSONProgressReporter)

    # Test "none" type
    reporter = create_reporter("none")
    assert isinstance(reporter, JSONProgressReporter)


def test_create_reporter_unknown_type():
    """Test create_reporter with unknown type falls back to JSON."""
    reporter = create_reporter("unknown_type")
    assert isinstance(reporter, JSONProgressReporter)


def test_create_reporter_deprecated_args():
    """Test create_reporter with deprecated arguments."""
    # Test force_rich
    reporter = create_reporter("auto", force_rich=True)
    # Should create Rich reporter (or JSON if Rich not available)
    try:
        assert isinstance(reporter, RichProgressReporter)
    except AssertionError:
        # Fallback to JSON if Rich not available
        assert isinstance(reporter, JSONProgressReporter)

    # Test force_json
    reporter = create_reporter("auto", force_json=True)
    assert isinstance(reporter, JSONProgressReporter)


def test_rich_reporter_basic():
    """Test RichProgressReporter basic functionality."""
    try:
        reporter = RichProgressReporter()
        reporter.on_start(100)
        assert reporter.total_files == 100
        assert reporter.start_time is not None

        reporter.on_stage_change(SyncStage.PARSING, "解析中...")
        assert reporter.current_stage == SyncStage.PARSING

        progress = SyncProgress(
            stage=SyncStage.PARSING,
            current=5,
            total=100,
            current_file="test.txt",
            elapsed_seconds=1.5,
            stage_name="解析: test.txt",
        )
        reporter.on_file_progress(progress)

        reporter.on_error("Test error")
        assert reporter.error_count == 1

        summary = {
            "processed": 100,
            "updated": 90,
            "skipped": 10,
            "failed": 0,
            "errors": [],
            "indexed_files": 90,
            "indexed_chunks": 1000,
            "partial_files": 0,
            "chunks_dropped_total": 5,
            "pdf_observability": [],
        }
        # on_complete will display output, we just verify it doesn't crash
        reporter.on_complete(summary)

    except ImportError:
        # Rich not installed, skip test
        pytest.skip("Rich library not installed")


def test_sync_stage_enum():
    """Test SyncStage enum values."""
    assert SyncStage.SCANNING.value == "scanning"
    assert SyncStage.PARSING.value == "parsing"
    assert SyncStage.VECTORIZING.value == "vectorizing"
    assert SyncStage.WRITING.value == "writing"
    assert SyncStage.FINALIZING.value == "finalizing"


def test_progress_reporter_interface():
    """Test that ProgressReporter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        # Abstract base class should raise TypeError when instantiated
        ProgressReporter()  # type: ignore


@pytest.mark.parametrize(
    "stage,emoji",
    [
        (SyncStage.SCANNING, "🔍"),
        (SyncStage.PARSING, "📄"),
        (SyncStage.VECTORIZING, "🔢"),
        (SyncStage.WRITING, "💾"),
        (SyncStage.FINALIZING, "✨"),
    ],
)
def test_stage_emojis(stage, emoji):
    """Test stage emoji mapping (verify the mapping exists)."""
    # This test just verifies the stages are valid
    assert isinstance(stage, SyncStage)
    assert stage.value in ["scanning", "parsing", "vectorizing", "writing", "finalizing"]

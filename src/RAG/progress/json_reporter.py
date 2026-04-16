"""JSON progress reporter - silent during sync, outputs JSON on completion.

This reporter maintains backward compatibility with the existing JSON output format.
"""

from __future__ import annotations

import json
from time import time
from typing import Any

from src.RAG.progress.reporter import ProgressReporter, SyncProgress, SyncStage


class JSONProgressReporter(ProgressReporter):
    """Silent reporter that only outputs JSON summary on completion.

    This reporter maintains backward compatibility with the existing system by
    not producing any output during processing, then printing the final summary
    in JSON format when complete.
    """

    def __init__(self) -> None:
        """Initialize the JSON reporter."""
        self.total_files: int = 0
        self.start_time: float | None = None

    def on_start(self, total_files: int) -> None:
        """Record start time and total file count."""
        self.total_files = total_files
        self.start_time = time()

    def on_stage_change(self, stage: SyncStage, stage_name: str) -> None:
        """Silent - no output during processing."""
        pass

    def on_file_progress(self, progress: SyncProgress) -> None:
        """Silent - no output during processing."""
        pass

    def on_complete(self, summary: dict[str, Any]) -> None:
        """Output final summary as JSON.

        Adds timing information to the summary while preserving all existing fields.
        """
        if self.start_time is None:
            elapsed = 0.0
        else:
            elapsed = time() - self.start_time

        # Create enhanced summary with timing info
        enhanced = dict(summary)
        enhanced["elapsed_seconds"] = elapsed

        # Add optional performance metrics
        if self.total_files > 0:
            enhanced["avg_time_per_file"] = elapsed / self.total_files

        # Output JSON - fully compatible with existing format
        print(json.dumps(enhanced, ensure_ascii=False, indent=2))

    def on_error(self, error: str) -> None:
        """Silent - errors are collected in summary and reported on completion."""
        pass

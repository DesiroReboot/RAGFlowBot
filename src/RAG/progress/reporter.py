"""Progress reporting infrastructure for knowledge base synchronization.

This module provides the abstract interface and data structures for reporting
progress during KB sync operations, supporting multiple output modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from time import time
from typing import Any


class SyncStage(Enum):
    """Stages of the KB synchronization process."""

    SCANNING = "scanning"
    PARSING = "parsing"
    VECTORIZING = "vectorizing"
    WRITING = "writing"
    FINALIZING = "finalizing"


@dataclass(frozen=True)
class SyncProgress:
    """Progress update for KB synchronization."""

    stage: SyncStage
    current: int
    total: int
    current_file: str
    elapsed_seconds: float
    stage_name: str

    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        return (self.current / self.total * 100) if self.total > 0 else 0


class ProgressReporter(ABC):
    """Abstract interface for progress reporting during KB sync."""

    @abstractmethod
    def on_start(self, total_files: int) -> None:
        """Called when sync starts, after file scanning completes.

        Args:
            total_files: Total number of files to process
        """
        ...

    @abstractmethod
    def on_stage_change(self, stage: SyncStage, stage_name: str) -> None:
        """Called when processing stage changes.

        Args:
            stage: The new stage
            stage_name: Human-readable stage description
        """
        ...

    @abstractmethod
    def on_file_progress(self, progress: SyncProgress) -> None:
        """Called for each file progress update.

        Args:
            progress: Current progress information
        """
        ...

    @abstractmethod
    def on_complete(self, summary: dict[str, Any]) -> None:
        """Called when sync completes successfully.

        Args:
            summary: Final summary statistics
        """
        ...

    @abstractmethod
    def on_error(self, error: str) -> None:
        """Called when an error occurs during processing.

        Args:
            error: Error message
        """
        ...

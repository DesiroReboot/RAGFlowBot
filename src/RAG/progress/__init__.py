"""Progress reporting for knowledge base synchronization.

This package provides progress reporting infrastructure for KB sync operations,
supporting multiple output modes (visual progress bars, JSON output, etc.).

Main exports:
    - ProgressReporter: Abstract base class for progress reporters
    - SyncStage: Enum representing sync stages
    - SyncProgress: Data class for progress updates
    - create_reporter: Factory function to create reporters
    - JSONProgressReporter: Silent reporter with JSON output
    - RichProgressReporter: Visual progress bar reporter

Example:
    >>> from src.RAG.progress import create_reporter, SyncStage, SyncProgress
    >>> reporter = create_reporter("rich")
    >>> reporter.on_start(100)
    >>> reporter.on_stage_change(SyncStage.PARSING, "解析文件中...")
"""

from src.RAG.progress.factory import create_reporter
from src.RAG.progress.json_reporter import JSONProgressReporter
from src.RAG.progress.reporter import ProgressReporter, SyncStage, SyncProgress
from src.RAG.progress.rich_reporter import RichProgressReporter

__all__ = [
    "ProgressReporter",
    "SyncStage",
    "SyncProgress",
    "create_reporter",
    "JSONProgressReporter",
    "RichProgressReporter",
]

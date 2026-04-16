"""Rich progress reporter - visual progress bars for terminal environments.

This reporter provides beautiful, informative progress bars using the Rich library.
"""

from __future__ import annotations

import sys
from time import time
from typing import Any

# Import Rich components with lazy loading in the class
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.RAG.progress.reporter import ProgressReporter, SyncProgress, SyncStage


class RichProgressReporter(ProgressReporter):
    """Visual progress reporter using Rich library.

    Features:
    - Multi-stage progress tracking
    - Time estimation and speed calculation
    - Beautiful completion statistics panel
    - Graceful fallback to JSON if Rich is not available
    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the Rich progress reporter.

        Args:
            console: Optional Rich console instance. If None, creates a new one
                     that outputs to stderr to avoid interfering with JSON output.
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library is not installed. "
                "Install it with: pip install rich>=13.7.0"
            )

        self.console = console or Console(file=sys.stderr)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        self.main_task_id: str | None = None
        self.total_files: int = 0
        self.start_time: float | None = None
        self.current_stage: SyncStage | None = None
        self.error_count: int = 0

    def on_start(self, total_files: int) -> None:
        """Initialize progress bar with total file count."""
        self.total_files = total_files
        self.start_time = time()
        self.error_count = 0

        self.progress.start()
        self.main_task_id = self.progress.add_task(
            f"同步知识库 - 共 {total_files} 个文件",
            total=total_files,
        )

    def on_stage_change(self, stage: SyncStage, stage_name: str) -> None:
        """Update progress bar description with new stage."""
        self.current_stage = stage
        if self.main_task_id:
            stage_emoji = {
                SyncStage.SCANNING: "🔍",
                SyncStage.PARSING: "📄",
                SyncStage.VECTORIZING: "🔢",
                SyncStage.WRITING: "💾",
                SyncStage.FINALIZING: "✨",
            }.get(stage, "📋")

            self.progress.update(
                self.main_task_id,
                description=f"{stage_emoji} {stage_name}",
            )

    def on_file_progress(self, progress: SyncProgress) -> None:
        """Update progress bar with current file information."""
        if self.main_task_id:
            # Update description with current file
            self.progress.update(
                self.main_task_id,
                description=f"{progress.stage_name} - {progress.current_file}",
                completed=progress.current,
            )

    def on_complete(self, summary: dict[str, Any]) -> None:
        """Display completion statistics panel."""
        self.progress.stop()

        # Calculate timing
        elapsed = time() - self.start_time if self.start_time else 0
        avg_time = elapsed / self.total_files if self.total_files > 0 else 0

        # Create beautiful statistics table
        table = Table(title="✓ 知识库同步完成", show_header=True)
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")

        # Add summary statistics
        table.add_row("处理文件数", str(summary.get("processed", 0)))
        table.add_row("更新文件数", str(summary.get("updated", 0)))
        table.add_row("跳过文件数", str(summary.get("skipped", 0)))
        table.add_row("失败文件数", str(summary.get("failed", 0)))
        table.add_row("索引文件数", str(summary.get("indexed_files", 0)))
        table.add_row("索引块数", str(summary.get("indexed_chunks", 0)))

        if summary.get("partial_files", 0) > 0:
            table.add_row("部分成功文件", str(summary["partial_files"]))

        if summary.get("chunks_dropped_total", 0) > 0:
            table.add_row("丢弃块数", str(summary["chunks_dropped_total"]))

        # Add timing information
        table.add_row("总耗时", self._format_duration(elapsed))
        if avg_time > 0:
            table.add_row("平均耗时", f"{avg_time:.2f} 秒/文件")

        # Add errors if any
        if summary.get("errors"):
            errors = summary["errors"][:5]  # Show first 5 errors
            error_text = "\n".join(f"  • {e}" for e in errors)
            if len(summary["errors"]) > 5:
                error_text += f"\n  ... 还有 {len(summary['errors']) - 5} 个错误"
            table.add_row("[red]错误信息[/red]", error_text)

        # Print the panel
        self.console.print(Panel(table))

    def on_error(self, error: str) -> None:
        """Track error count (errors displayed in completion panel)."""
        self.error_count += 1

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} 秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} 分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} 小时"

"""Factory for creating progress reporters.

This module provides a factory function to create the appropriate progress reporter
based on the specified type and environment.
"""

from __future__ import annotations

import logging
import sys

from src.RAG.progress.json_reporter import JSONProgressReporter
from src.RAG.progress.reporter import ProgressReporter
from src.RAG.progress.rich_reporter import RichProgressReporter

logger = logging.getLogger(__name__)


def create_reporter(
    reporter_type: str = "auto",
    *,
    force_rich: bool = False,
    force_json: bool = False,
) -> ProgressReporter:
    """Create a progress reporter based on type and environment.

    Args:
        reporter_type: Type of reporter to create. Options:
            - "auto": Automatically detect based on TTY environment
            - "rich": Force Rich progress bar
            - "json": Force JSON reporter (silent during sync)
            - "none": Equivalent to JSON reporter
        force_rich: Deprecated: Use reporter_type="rich" instead
        force_json: Deprecated: Use reporter_type="json" instead

    Returns:
        A progress reporter instance

    Examples:
        >>> # Auto-detect
        >>> reporter = create_reporter()
        >>> # Force Rich
        >>> reporter = create_reporter("rich")
        >>> # Force JSON (silent)
        >>> reporter = create_reporter("json")
    """
    # Handle deprecated arguments
    if force_rich:
        reporter_type = "rich"
    elif force_json:
        reporter_type = "json"

    # Normalize type
    reporter_type = reporter_type.lower().strip()

    # Auto-detect based on TTY
    if reporter_type == "auto":
        # Check if stdout is a TTY (interactive terminal)
        reporter_type = "rich" if sys.stdout.isatty() else "json"

    # Create the appropriate reporter
    if reporter_type == "rich":
        try:
            return RichProgressReporter()
        except ImportError as exc:
            logger.warning(
                f"Rich library not available, falling back to JSON reporter: {exc}"
            )
            return JSONProgressReporter()
    elif reporter_type in ("json", "none"):
        return JSONProgressReporter()
    else:
        logger.warning(
            f"Unknown reporter type '{reporter_type}', falling back to JSON"
        )
        return JSONProgressReporter()

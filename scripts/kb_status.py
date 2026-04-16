#!/usr/bin/env python3
"""CLI script to check KB index status with staleness detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.KB.status_service import KBStatusService


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    if not ts:
        return "N/A"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def print_status(status: KBStatus, *, json_output: bool = False) -> None:
    """Print the KB status."""
    if json_output:
        print(json.dumps({
            "state": status.state,
            "reason": status.reason,
            "indexed_files": status.indexed_files,
            "indexed_chunks": status.indexed_chunks,
            "source_file_count": status.source_file_count,
            "source_scanned_at": status.source_scanned_at,
            "last_index_run_id": status.last_index_run_id,
        }, indent=2, ensure_ascii=False))
    else:
        print(f"State: {status.state}")
        print(f"Reason: {status.reason}")
        if status.state != "no_index":
            print(f"Indexed files: {status.indexed_files}")
            print(f"Indexed chunks: {status.indexed_chunks}")
        if status.source_file_count > 0:
            print(f"Source files: {status.source_file_count}")
        if status.source_scanned_at:
            print(f"Source scanned at: {format_timestamp(status.source_scanned_at)}")
        if status.last_index_run_id:
            print(f"Last run ID: {status.last_index_run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check KB index status with staleness detection"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config.toml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    args = parser.parse_args()

    try:
        # Load config
        config = Config(args.config)

        # Create status service
        status_service = KBStatusService(
            db_path=config.database.db_path,
            source_dir=config.knowledge_base.source_dir,
        )

        # Get and print status
        status = status_service.get_status()
        print_status(status, json_output=args.json)

        # Exit code based on state
        if status.state == "no_index":
            return 1
        elif status.state == "failed":
            return 2
        elif status.state == "stale":
            return 3
        elif status.state == "partial":
            return 4
        else:  # ready
            return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 100


if __name__ == "__main__":
    sys.exit(main())

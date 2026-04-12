from __future__ import annotations

import argparse
import gc
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


TARGET_PREFIXES: tuple[str, ...] = ("ragv2-atomic-", "ragv2-search-")


@dataclass(frozen=True)
class CleanupReport:
    removed_files: tuple[str, ...]
    removed_dirs: tuple[str, ...]
    skipped: tuple[str, ...]


def _matches_target(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in TARGET_PREFIXES)


def _delete_file_with_retry(path: Path, retries: int = 20, delay_sec: float = 0.05) -> bool:
    for _ in range(retries):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            gc.collect()
            time.sleep(delay_sec)
    return False


def _delete_dir_with_retry(path: Path, retries: int = 20, delay_sec: float = 0.05) -> bool:
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            gc.collect()
            time.sleep(delay_sec)
    return False


def cleanup_oneoff_artifacts(root: Path, *, dry_run: bool = False) -> CleanupReport:
    removed_files: list[str] = []
    removed_dirs: list[str] = []
    skipped: list[str] = []

    for path in sorted(root.iterdir(), key=lambda p: p.name):
        name = path.name
        if path.is_file() and name.endswith(".db") and _matches_target(name):
            if dry_run:
                removed_files.append(name)
                continue
            if _delete_file_with_retry(path):
                removed_files.append(name)
            else:
                skipped.append(name)
            continue

        if path.is_dir() and _matches_target(name):
            if dry_run:
                removed_dirs.append(name)
                continue
            if _delete_dir_with_retry(path):
                removed_dirs.append(name)
            else:
                skipped.append(name)

    return CleanupReport(
        removed_files=tuple(removed_files),
        removed_dirs=tuple(removed_dirs),
        skipped=tuple(skipped),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean one-off RAGv2 test artifacts.")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to clean. Default: current directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    report = cleanup_oneoff_artifacts(root, dry_run=bool(args.dry_run))

    print(f"root={root}")
    print(f"removed_files={len(report.removed_files)}")
    print(f"removed_dirs={len(report.removed_dirs)}")
    print(f"skipped={len(report.skipped)}")
    for name in report.removed_files:
        print(f"file:{name}")
    for name in report.removed_dirs:
        print(f"dir:{name}")
    for name in report.skipped:
        print(f"skipped:{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

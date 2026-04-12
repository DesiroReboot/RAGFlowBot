from __future__ import annotations

import gc
import os
from pathlib import Path
import shutil
import tempfile
import time
from uuid import uuid4

from scripts.cleanup_oneoff_artifacts import cleanup_oneoff_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ECBOT_DOTENV_PATH", str(PROJECT_ROOT / ".env.test"))
LOCAL_TMP = PROJECT_ROOT / "DB" / "tmp_runtime"
LOCAL_TMP.mkdir(parents=True, exist_ok=True)
PYTEST_BASETMP_PARENT = PROJECT_ROOT / "tmp_runtime"
PYTEST_BASETMP_PARENT.mkdir(parents=True, exist_ok=True)

for key in ("TMPDIR", "TMP", "TEMP"):
    os.environ[key] = str(LOCAL_TMP)

# Ensure tempfile APIs used in unittest-style tests also use workspace-local dir.
tempfile.tempdir = str(LOCAL_TMP)


def _safe_mkdtemp(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | None = None,
) -> str:
    base = Path(dir or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    safe_prefix = prefix or "tmp"
    safe_suffix = suffix or ""
    for _ in range(256):
        candidate = base / f"{safe_prefix}{uuid4().hex}{safe_suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return str(candidate)
        except FileExistsError:
            continue
    raise FileExistsError(f"unable to allocate temp dir under {base}")


tempfile.mkdtemp = _safe_mkdtemp

_ORIGINAL_PATH_MKDIR = Path.mkdir


def _safe_path_mkdir(
    self: Path,
    mode: int = 0o777,
    parents: bool = False,
    exist_ok: bool = False,
) -> None:
    adjusted_mode = mode
    if os.name == "nt" and int(mode) == 0o700 and "pytest" in str(self).lower():
        # On some Windows volumes, mode=0o700 creates directories that cannot be listed/deleted.
        adjusted_mode = 0o777
    _ORIGINAL_PATH_MKDIR(self, mode=adjusted_mode, parents=parents, exist_ok=exist_ok)


Path.mkdir = _safe_path_mkdir

_ORIGINAL_RMTREE = shutil.rmtree


def _safe_rmtree(path: str, *args, **kwargs) -> None:
    last_error: Exception | None = None
    for _ in range(20):
        try:
            _ORIGINAL_RMTREE(path, *args, **kwargs)
            return
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            time.sleep(0.05)
    if last_error is not None:
        raise last_error


shutil.rmtree = _safe_rmtree


def _cleanup_ragv2_oneoff_artifacts() -> None:
    cleanup_oneoff_artifacts(PROJECT_ROOT, dry_run=False)


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    _cleanup_ragv2_oneoff_artifacts()


def pytest_sessionfinish(session, exitstatus) -> None:  # type: ignore[no-untyped-def]
    _cleanup_ragv2_oneoff_artifacts()

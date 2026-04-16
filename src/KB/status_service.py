from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.KB.manifest_store import ManifestStore
from src.RAG.storage.sqlite_conn import connect
from src.RAG.storage.sqlite_schema import ensure_schema


@dataclass(frozen=True)
class SourceSnapshot:
    """O(1) snapshot of source files."""
    file_count: int
    max_mtime: float
    scanned_at: str


@dataclass(frozen=True)
class KBStatus:
    """Knowledge base index status."""
    state: str  # no_index, empty, partial, ready, failed, stale
    reason: str
    indexed_files: int = 0
    indexed_chunks: int = 0
    source_file_count: int = 0
    source_scanned_at: str = ""
    last_index_run_id: str = ""


class KBStatusService:
    """Unified service for KB status queries with staleness detection."""

    def __init__(self, db_path: str, source_dir: str):
        self.db_path = db_path
        self.source_dir = source_dir
        self.manifest_store = ManifestStore(db_path, ensure_schema=False)

    def get_status(self) -> KBStatus:
        """Get the current KB status with staleness detection."""
        manifest = self.manifest_store.get_manifest()

        # 1. 无索引
        if manifest is None:
            return KBStatus(
                state="no_index",
                reason="从未构建",
            )

        # 2. 索引为空
        if manifest.get("status") == "empty":
            source_count = int(manifest.get("source_file_count", 0))
            return KBStatus(
                state="empty",
                reason=f"已扫描但无文件 (source_count={source_count})",
                indexed_files=int(manifest.get("indexed_files", 0)),
                indexed_chunks=int(manifest.get("indexed_chunks", 0)),
                source_file_count=source_count,
                source_scanned_at=str(manifest.get("source_scanned_at", "")),
                last_index_run_id=str(manifest.get("last_index_run_id", "")),
            )

        # 3. 部分为空
        last_index_files = int(manifest.get("last_index_files", manifest.get("indexed_files", 0)))
        source_file_count = int(manifest.get("source_file_count", 0))
        if last_index_files < source_file_count:
            return KBStatus(
                state="partial",
                reason=f"部分失败: {last_index_files}/{source_file_count}",
                indexed_files=int(manifest.get("indexed_files", 0)),
                indexed_chunks=int(manifest.get("indexed_chunks", 0)),
                source_file_count=source_file_count,
                source_scanned_at=str(manifest.get("source_scanned_at", "")),
                last_index_run_id=str(manifest.get("last_index_run_id", "")),
            )

        # 4. 索引过期
        current_snapshot = self._scan_source_snapshot()
        manifest_max_mtime = float(manifest.get("source_max_mtime", 0))
        if (current_snapshot.file_count != source_file_count or
            current_snapshot.max_mtime > manifest_max_mtime):
            return KBStatus(
                state="stale",
                reason=f"源文件已变: {source_file_count} -> {current_snapshot.file_count}",
                indexed_files=int(manifest.get("indexed_files", 0)),
                indexed_chunks=int(manifest.get("indexed_chunks", 0)),
                source_file_count=current_snapshot.file_count,
                source_scanned_at=current_snapshot.scanned_at,
                last_index_run_id=str(manifest.get("last_index_run_id", "")),
            )

        # 5. 正常
        return KBStatus(
            state="ready",
            reason="索引就绪",
            indexed_files=int(manifest.get("indexed_files", 0)),
            indexed_chunks=int(manifest.get("indexed_chunks", 0)),
            source_file_count=source_file_count,
            source_scanned_at=str(manifest.get("source_scanned_at", "")),
            last_index_run_id=str(manifest.get("last_index_run_id", "")),
        )

    def is_ready_for_query(self) -> bool:
        """Check if the index is ready for queries."""
        status = self.get_status()
        return status.state in {"ready", "partial"}

    def _scan_source_snapshot(self) -> SourceSnapshot:
        """O(1) scan to collect source file metadata."""
        source_path = Path(self.source_dir)
        if not source_path.exists():
            return SourceSnapshot(
                file_count=0,
                max_mtime=0.0,
                scanned_at=datetime.now().isoformat(),
            )

        try:
            files = list(source_path.rglob("*"))
            actual_files = [f for f in files if f.is_file()]

            if not actual_files:
                return SourceSnapshot(
                    file_count=0,
                    max_mtime=0.0,
                    scanned_at=datetime.now().isoformat(),
                )

            max_mtime = max(f.stat().st_mtime for f in actual_files)

            return SourceSnapshot(
                file_count=len(actual_files),
                max_mtime=max_mtime,
                scanned_at=datetime.now().isoformat(),
            )
        except Exception:
            return SourceSnapshot(
                file_count=0,
                max_mtime=0.0,
                scanned_at=datetime.now().isoformat(),
            )

    def get_index_counts(self) -> dict[str, int]:
        """Get index table counts."""
        counts = {
            "indexed_files": 0,
            "indexed_chunks": 0,
            "fts_documents": 0,
            "vec_rows": 0,
        }
        try:
            with connect(self.db_path) as conn:
                ensure_schema(conn)
                counts["indexed_files"] = self._safe_count(conn, "files")
                counts["indexed_chunks"] = self._safe_count(conn, "chunks")
                counts["fts_documents"] = self._safe_count(conn, "fts_index")
                counts["vec_rows"] = self._safe_count(conn, "vec_index")
        except Exception:
            return counts
        return counts

    @staticmethod
    def _safe_count(conn: Any, table_name: str) -> int:
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        except Exception:
            return 0

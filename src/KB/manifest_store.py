from __future__ import annotations

import sqlite3
from typing import Any

from src.RAG.storage.sqlite_conn import connect
from src.RAG.storage.sqlite_schema import ensure_schema as ensure_sqlite_schema


class ManifestStore:
    def __init__(self, db_path: str, ensure_schema: bool = True):
        self.db_path = db_path
        if ensure_schema:
            with connect(self.db_path) as conn:
                ensure_sqlite_schema(conn)

    def get_manifest(self) -> dict[str, Any] | None:
        try:
            with connect(self.db_path) as conn:
                row = conn.execute("SELECT * FROM index_manifest WHERE id = 1").fetchone()
                return dict(row) if row else None
        except sqlite3.OperationalError as exc:
            if "no such table: index_manifest" not in str(exc).lower():
                raise
            with connect(self.db_path) as conn:
                ensure_sqlite_schema(conn)
                row = conn.execute("SELECT * FROM index_manifest WHERE id = 1").fetchone()
                return dict(row) if row else None

    def upsert_manifest(
        self,
        *,
        status: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        build_version: str,
        indexed_files: int,
        indexed_chunks: int,
        partial_files: int = 0,
        last_error: str | None = None,
        source_file_count: int = 0,
        source_max_mtime: str = "",
        source_scanned_at: str = "",
        last_index_run_id: str = "",
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO index_manifest (
                    id, status, embedding_provider, embedding_model,
                    embedding_dimension, build_version, indexed_files, indexed_chunks,
                    partial_files, last_error, updated_at, source_file_count,
                    source_max_mtime, source_scanned_at, last_index_files, last_index_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    embedding_provider=excluded.embedding_provider,
                    embedding_model=excluded.embedding_model,
                    embedding_dimension=excluded.embedding_dimension,
                    build_version=excluded.build_version,
                    indexed_files=excluded.indexed_files,
                    indexed_chunks=excluded.indexed_chunks,
                    partial_files=excluded.partial_files,
                    last_error=excluded.last_error,
                    updated_at=CURRENT_TIMESTAMP,
                    source_file_count=excluded.source_file_count,
                    source_max_mtime=excluded.source_max_mtime,
                    source_scanned_at=excluded.source_scanned_at,
                    last_index_files=excluded.last_index_files,
                    last_index_run_id=excluded.last_index_run_id
                """,
                (
                    1,
                    status,
                    embedding_provider,
                    embedding_model,
                    embedding_dimension,
                    build_version,
                    indexed_files,
                    indexed_chunks,
                    partial_files,
                    last_error,
                    source_file_count,
                    source_max_mtime,
                    source_scanned_at,
                    indexed_files,
                    last_index_run_id,
                ),
            )
            conn.commit()

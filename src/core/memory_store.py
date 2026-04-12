from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.RAG.storage.sqlite_conn import connect

logger = logging.getLogger(__name__)

QA_MEMORY_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS qa_run (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at TEXT,
        channel TEXT NOT NULL DEFAULT 'gateway',
        event_id TEXT DEFAULT '',
        message_id TEXT DEFAULT '',
        query_text TEXT NOT NULL DEFAULT '',
        query_hash TEXT NOT NULL DEFAULT '',
        retrieval_provider TEXT NOT NULL DEFAULT 'legacy',
        success INTEGER NOT NULL DEFAULT 0,
        fallback_type TEXT DEFAULT '',
        retrieval_confidence REAL NOT NULL DEFAULT 0.0,
        duration_ms INTEGER NOT NULL DEFAULT 0,
        error_stage TEXT DEFAULT '',
        error_type TEXT DEFAULT '',
        error_message TEXT DEFAULT '',
        final_answer_preview TEXT DEFAULT ''
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_decision_trace (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        seq_no INTEGER NOT NULL,
        stage TEXT NOT NULL,
        decision_code TEXT DEFAULT '',
        decision_summary TEXT DEFAULT '',
        fallback_used INTEGER NOT NULL DEFAULT 0,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES qa_run(run_id) ON DELETE CASCADE,
        UNIQUE(run_id, seq_no)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_io_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        io_type TEXT NOT NULL,
        producer TEXT NOT NULL DEFAULT '',
        content_json TEXT NOT NULL DEFAULT '{}',
        content_hash TEXT NOT NULL DEFAULT '',
        content_size INTEGER NOT NULL DEFAULT 0,
        truncated INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES qa_run(run_id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_qa_run_created_at ON qa_run(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_qa_run_query_hash_created ON qa_run(query_hash, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_qa_run_success_fallback_created ON qa_run(success, fallback_type, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_qa_decision_trace_run_seq ON qa_decision_trace(run_id, seq_no);",
    "CREATE INDEX IF NOT EXISTS idx_qa_decision_trace_stage_code_created ON qa_decision_trace(stage, decision_code, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_qa_io_snapshot_run_type ON qa_io_snapshot(run_id, io_type);",
]


class QAMemoryStore:
    def __init__(self, db_path: str, *, max_snapshot_chars: int = 12000) -> None:
        self.db_path = db_path
        self.max_snapshot_chars = max(1024, int(max_snapshot_chars))
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with connect(self.db_path) as conn:
            for sql in QA_MEMORY_DDL:
                conn.execute(sql)
            conn.commit()

    def start_run(
        self,
        *,
        run_id: str,
        channel: str = "gateway",
        event_id: str = "",
        message_id: str = "",
        query_text: str = "",
        query_hash: str = "",
        retrieval_provider: str = "legacy",
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO qa_run (
                    run_id, channel, event_id, message_id, query_text, query_hash, retrieval_provider
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id).strip(),
                    str(channel or "gateway").strip() or "gateway",
                    str(event_id or "").strip(),
                    str(message_id or "").strip(),
                    str(query_text or "").strip(),
                    str(query_hash or "").strip(),
                    str(retrieval_provider or "legacy").strip() or "legacy",
                ),
            )
            conn.commit()

    def finish_run(
        self,
        *,
        run_id: str,
        success: bool,
        fallback_type: str = "",
        retrieval_provider: str = "legacy",
        retrieval_confidence: float = 0.0,
        duration_ms: int = 0,
        error_stage: str = "",
        error_type: str = "",
        error_message: str = "",
        final_answer_preview: str = "",
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE qa_run
                SET
                    finished_at = CURRENT_TIMESTAMP,
                    success = ?,
                    fallback_type = ?,
                    retrieval_provider = ?,
                    retrieval_confidence = ?,
                    duration_ms = ?,
                    error_stage = ?,
                    error_type = ?,
                    error_message = ?,
                    final_answer_preview = ?
                WHERE run_id = ?
                """,
                (
                    1 if bool(success) else 0,
                    str(fallback_type or "").strip(),
                    str(retrieval_provider or "legacy").strip() or "legacy",
                    float(retrieval_confidence or 0.0),
                    max(0, int(duration_ms or 0)),
                    str(error_stage or "").strip(),
                    str(error_type or "").strip(),
                    str(error_message or "").strip()[:500],
                    str(final_answer_preview or "").strip()[:400],
                    str(run_id).strip(),
                ),
            )
            conn.commit()

    def append_decision_trace(
        self,
        *,
        run_id: str,
        seq_no: int,
        stage: str,
        decision_code: str = "",
        decision_summary: str = "",
        fallback_used: bool = False,
        metrics: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO qa_decision_trace (
                    run_id, seq_no, stage, decision_code, decision_summary,
                    fallback_used, metrics_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id).strip(),
                    max(1, int(seq_no)),
                    str(stage or "").strip() or "unknown",
                    str(decision_code or "").strip(),
                    str(decision_summary or "").strip(),
                    1 if bool(fallback_used) else 0,
                    self._to_json(metrics or {}),
                    self._to_json(payload or {}),
                ),
            )
            conn.commit()

    def append_io_snapshot(
        self,
        *,
        run_id: str,
        io_type: str,
        producer: str,
        content: dict[str, Any] | list[Any] | str | None,
    ) -> None:
        content_payload, content_hash, content_size, truncated = self._pack_content(content)
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO qa_io_snapshot (
                    run_id, io_type, producer, content_json, content_hash, content_size, truncated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id).strip(),
                    str(io_type or "").strip() or "unknown",
                    str(producer or "").strip(),
                    content_payload,
                    content_hash,
                    content_size,
                    1 if truncated else 0,
                ),
            )
            conn.commit()

    def safe_call(self, fn_name: str, **kwargs: Any) -> None:
        fn = getattr(self, fn_name, None)
        if fn is None:
            return
        try:
            fn(**kwargs)
        except Exception as exc:
            logger.warning("qa_memory_%s_failed: %s", fn_name, exc)

    def _pack_content(
        self, content: dict[str, Any] | list[Any] | str | None
    ) -> tuple[str, str, int, bool]:
        raw_json = self._to_json(content if content is not None else {})
        raw_size = len(raw_json.encode("utf-8"))
        content_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
        if len(raw_json) <= self.max_snapshot_chars:
            return raw_json, content_hash, raw_size, False
        preview = raw_json[: self.max_snapshot_chars]
        wrapped = self._to_json(
            {
                "truncated": True,
                "preview_json": preview,
                "original_hash": content_hash,
                "original_size": raw_size,
            }
        )
        return wrapped, content_hash, raw_size, True

    @staticmethod
    def _to_json(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
from threading import Lock, Thread
from typing import Any
import sys

from src.config import Config
from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.kbase_manager import KBaseManager
from src.RAG.readiness import is_index_ready, normalize_manifest_status
from src.KB.manifest_store import ManifestStore
from src.RAG.storage.sqlite_conn import connect
from src.RAG.storage.sqlite_schema import ensure_schema
from src.RAG.progress import create_reporter

logger = logging.getLogger(__name__)


class KBaseStartupBootstrap:
    _PROCESS_LOCK = Lock()

    def __init__(self, config: Config):
        self.config = config
        self.db_path = str(config.database.db_path)
        self.manifest_store = ManifestStore(self.db_path, ensure_schema=False)
        self._state_lock = Lock()
        self._worker: Thread | None = None
        self._state: dict[str, Any] = {
            "enabled": bool(getattr(config.knowledge_base, "auto_init_on_startup", False)),
            "blocking": bool(getattr(config.knowledge_base, "init_blocking", False)),
            "fail_open": bool(getattr(config.knowledge_base, "init_fail_open", True)),
            "effective_blocking": bool(getattr(config.knowledge_base, "init_blocking", False)),
            "launched_async": False,
            "started": False,
            "completed": False,
            "success": False,
            "attempted": False,
            "skipped_reason": "",
            "error": "",
            "manifest_before": {},
            "manifest_after": {},
            "counts_before": {},
            "counts_after": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def start(self) -> dict[str, Any]:
        enabled = bool(getattr(self.config.knowledge_base, "auto_init_on_startup", False))
        blocking = bool(getattr(self.config.knowledge_base, "init_blocking", False))
        fail_open = bool(getattr(self.config.knowledge_base, "init_fail_open", True))
        effective_blocking = bool(blocking or not fail_open)
        readiness = self.readiness_snapshot()

        with self._state_lock:
            self._state.update(
                {
                    "enabled": enabled,
                    "blocking": blocking,
                    "fail_open": fail_open,
                    "effective_blocking": effective_blocking,
                    "manifest_before": dict(readiness.get("manifest", {})),
                    "counts_before": dict(readiness.get("counts", {})),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        logger.info(
            "kb_startup_bootstrap check=%s",
            json.dumps(
                {
                    "enabled": enabled,
                    "blocking": blocking,
                    "fail_open": fail_open,
                    "effective_blocking": effective_blocking,
                    "status": readiness.get("status"),
                    "counts": readiness.get("counts", {}),
                    "needs_init": readiness.get("needs_init", False),
                },
                ensure_ascii=False,
            ),
        )

        if not enabled:
            self._mark_skipped("auto_init_disabled")
            return self.status()
        if not bool(readiness.get("needs_init", False)):
            self._mark_skipped("index_already_ready")
            return self.status()

        if effective_blocking:
            result = self._run_once()
            self._update_state_from_result(result)
            if not result.get("success", False) and not fail_open:
                raise RuntimeError(
                    f"kb_startup_init_failed: {result.get('error', 'unknown_error')}"
                )
            return self.status()

        with self._state_lock:
            if self._worker is not None and self._worker.is_alive():
                self._state["skipped_reason"] = "init_already_running"
                return self.status()
            self._state["started"] = True
            self._state["attempted"] = True
            self._state["launched_async"] = True
            self._state["timestamp"] = datetime.now(UTC).isoformat()
            worker = Thread(target=self._run_async, name="kb-startup-init", daemon=True)
            self._worker = worker
            worker.start()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            snapshot = dict(self._state)
        snapshot["readiness"] = self.readiness_snapshot()
        return snapshot

    def readiness_snapshot(self) -> dict[str, Any]:
        manifest = self.manifest_store.get_manifest() or {}
        counts = self._read_counts()
        status = normalize_manifest_status(manifest)
        ready, reason, _status = is_index_ready(manifest, counts)
        empty_index = (
            int(counts.get("indexed_files", 0)) <= 0
            and int(counts.get("indexed_chunks", 0)) <= 0
            and int(counts.get("fts_documents", 0)) <= 0
            and int(counts.get("vec_rows", 0)) <= 0
        )
        needs_init = not bool(ready)
        return {
            "status": status,
            "ready": bool(ready),
            "reason": reason,
            "needs_init": needs_init,
            "empty_index": empty_index,
            "manifest": manifest,
            "counts": counts,
        }

    def _run_async(self) -> None:
        result = self._run_once()
        self._update_state_from_result(result)

    def _run_once(self) -> dict[str, Any]:
        started_at = datetime.now(UTC).isoformat()
        with self._PROCESS_LOCK:
            lock_fd = self._acquire_lock_file()
            if lock_fd is None:
                readiness = self.readiness_snapshot()
                return {
                    "attempted": False,
                    "success": bool(readiness.get("ready", False)),
                    "skipped_reason": "lock_already_held",
                    "error": "",
                    "manifest_after": readiness.get("manifest", {}),
                    "counts_after": readiness.get("counts", {}),
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            try:
                # Determine if we should show progress bar
                # Only show progress for blocking mode in a TTY environment
                blocking = bool(self._state.get("effective_blocking", False))
                is_tty = sys.stdout.isatty()
                should_show_progress = blocking and is_tty

                # Create appropriate progress reporter
                progress_type = "rich" if should_show_progress else "json"
                reporter = create_reporter(progress_type)

                manager = KBaseManager(self._build_kbase_config(), progress_reporter=reporter)
                manager.sync_configured_source()
                readiness = self.readiness_snapshot()
                success = bool(readiness.get("ready", False))
                logger.info(
                    "kb_startup_bootstrap init_done=%s",
                    json.dumps(
                        {
                            "success": success,
                            "status": readiness.get("status"),
                            "counts": readiness.get("counts", {}),
                        },
                        ensure_ascii=False,
                    ),
                )
                return {
                    "attempted": True,
                    "success": success,
                    "skipped_reason": "",
                    "error": "",
                    "manifest_after": readiness.get("manifest", {}),
                    "counts_after": readiness.get("counts", {}),
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            except Exception as exc:
                error_text = str(exc)
                self._record_failed_manifest(error_text)
                readiness = self.readiness_snapshot()
                logger.exception("kb_startup_bootstrap init_failed: %s", error_text)
                return {
                    "attempted": True,
                    "success": False,
                    "skipped_reason": "",
                    "error": error_text,
                    "manifest_after": readiness.get("manifest", {}),
                    "counts_after": readiness.get("counts", {}),
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            finally:
                self._release_lock_file(lock_fd)

    def _mark_skipped(self, reason: str) -> None:
        readiness = self.readiness_snapshot()
        with self._state_lock:
            self._state.update(
                {
                    "started": False,
                    "completed": True,
                    "success": bool(readiness.get("ready", False)),
                    "attempted": False,
                    "launched_async": False,
                    "skipped_reason": reason,
                    "error": "",
                    "manifest_after": dict(readiness.get("manifest", {})),
                    "counts_after": dict(readiness.get("counts", {})),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        logger.info("kb_startup_bootstrap skipped=%s", reason)

    def _update_state_from_result(self, result: dict[str, Any]) -> None:
        with self._state_lock:
            self._state.update(
                {
                    "started": True,
                    "completed": True,
                    "attempted": bool(result.get("attempted", False)),
                    "success": bool(result.get("success", False)),
                    "skipped_reason": str(result.get("skipped_reason", "")),
                    "error": str(result.get("error", "")),
                    "manifest_after": dict(result.get("manifest_after", {})),
                    "counts_after": dict(result.get("counts_after", {})),
                    "started_at": str(result.get("started_at", "")),
                    "finished_at": str(result.get("finished_at", "")),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    def _build_kbase_config(self) -> KBaseConfig:
        return KBaseConfig(
            db_path=self.config.database.db_path,
            source_dir=self.config.knowledge_base.source_dir,
            supported_extensions=self.config.knowledge_base.supported_extensions,
            auto_sync_on_startup=False,
            ocr_enabled=self.config.knowledge_base.ocr_enabled,
            ocr_language=self.config.knowledge_base.ocr_language,
            ocr_dpi_scale=self.config.knowledge_base.ocr_dpi_scale,
            ocr_trigger_readability=self.config.knowledge_base.ocr_trigger_readability,
            min_chunk_readability=self.config.knowledge_base.min_chunk_readability,
            chunk_size=self.config.knowledge_base.chunk_size,
            chunk_overlap=self.config.knowledge_base.chunk_overlap,
            vector_dimension=self.config.embedding.dimension,
            rag_top_k=self.config.search.rag_top_k,
            fts_top_k=self.config.search.fts_top_k,
            vec_top_k=self.config.search.vec_top_k,
            fusion_rrf_k=self.config.search.fusion_rrf_k,
            context_top_k=self.config.search.context_top_k,
            embedding_provider=self.config.embedding.provider,
            embedding_base_url=self.config.embedding.base_url,
            embedding_api_key=self.config.embedding.api_key,
            embedding_model=self.config.embedding.model,
            embedding_batch_size=self.config.embedding.batch_size,
            embedding_timeout=self.config.embedding.timeout,
            embedding_max_retries=self.config.embedding.max_retries,
            build_version=self.config.knowledge_base.build_version,
        )

    def _record_failed_manifest(self, error_text: str) -> None:
        readiness = self.readiness_snapshot()
        manifest = readiness.get("manifest", {})
        counts = readiness.get("counts", {})
        self.manifest_store.upsert_manifest(
            status="failed",
            embedding_provider=self.config.embedding.provider,
            embedding_model=self.config.embedding.model,
            embedding_dimension=int(self.config.embedding.dimension),
            build_version=self.config.knowledge_base.build_version,
            indexed_files=int(counts.get("indexed_files", manifest.get("indexed_files", 0)) or 0),
            indexed_chunks=int(counts.get("indexed_chunks", manifest.get("indexed_chunks", 0)) or 0),
            partial_files=int(manifest.get("partial_files", 0) or 0),
            last_error=error_text,
        )

    def _lock_file_path(self) -> Path:
        db_path = Path(self.db_path)
        return db_path.with_suffix(db_path.suffix + ".kb_init.lock")

    def _acquire_lock_file(self) -> int | None:
        lock_file = self._lock_file_path()
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        payload = f"pid={os.getpid()} ts={datetime.now(UTC).isoformat()}".encode("utf-8")
        try:
            os.write(fd, payload)
        except Exception:
            pass
        return fd

    def _release_lock_file(self, fd: int | None) -> None:
        if fd is None:
            return
        try:
            os.close(fd)
        except Exception:
            pass
        lock_file = self._lock_file_path()
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass

    def _read_counts(self) -> dict[str, int]:
        with connect(self.db_path) as conn:
            ensure_schema(conn)
            return {
                "indexed_files": self._safe_count(conn, "files"),
                "indexed_chunks": self._safe_count(conn, "chunks"),
                "fts_documents": self._safe_count(conn, "fts_index"),
                "vec_rows": self._safe_count(conn, "vec_index"),
            }

    @staticmethod
    def _safe_count(conn: Any, table_name: str) -> int:
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        except Exception:
            return 0

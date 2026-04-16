from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.config import Config
from src.KB.status_service import KBStatus


def create_kb_router(config: Config) -> APIRouter:
    """Create KB status router."""
    router = APIRouter()

    @router.get("/kb/status")
    def get_kb_status() -> dict[str, Any]:
        """Get KB index status with staleness detection."""
        try:
            from src.KB.status_service import KBStatusService

            status_service = KBStatusService(
                db_path=config.database.db_path,
                source_dir=config.knowledge_base.source_dir,
            )
            status = status_service.get_status()

            return {
                "state": status.state,
                "reason": status.reason,
                "indexed_files": status.indexed_files,
                "indexed_chunks": status.indexed_chunks,
                "source_file_count": status.source_file_count,
                "source_scanned_at": status.source_scanned_at,
                "last_index_run_id": status.last_index_run_id,
                "ready_for_query": status.state in {"ready", "partial"},
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router

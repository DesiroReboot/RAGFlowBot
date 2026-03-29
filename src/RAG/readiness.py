from __future__ import annotations

from typing import Any

READY_STATUSES = {"ready", "partial"}
KNOWN_STATUSES = {"empty", "ready", "partial", "failed"}


def normalize_manifest_status(manifest: dict[str, Any] | None) -> str:
    payload = manifest or {}
    status = str(payload.get("status", "")).strip().lower()
    if status in KNOWN_STATUSES:
        return status
    return "ready" if payload else "empty"


def is_index_ready(
    manifest: dict[str, Any] | None,
    counts: dict[str, Any] | None = None,
) -> tuple[bool, str, str]:
    payload = manifest or {}
    count_payload = counts or {}
    status = normalize_manifest_status(payload)

    indexed_files = _as_int(count_payload.get("indexed_files", payload.get("indexed_files", 0)))
    indexed_chunks = _as_int(count_payload.get("indexed_chunks", payload.get("indexed_chunks", 0)))
    fts_documents = _as_int(count_payload.get("fts_documents", 0))
    vec_rows = _as_int(count_payload.get("vec_rows", 0))
    has_indexed_rows = any(value > 0 for value in (indexed_files, indexed_chunks, fts_documents, vec_rows))

    if status == "failed":
        return False, "manifest_failed", status

    if status in READY_STATUSES:
        if has_indexed_rows:
            return True, "ready_with_index_data", status
        return True, "ready_from_manifest", status

    if has_indexed_rows:
        return True, "ready_from_index_counts", status

    return False, "index_not_ready", status


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0

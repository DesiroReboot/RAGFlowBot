from __future__ import annotations

from pathlib import Path

from src.KB.manifest_store import ManifestStore


def test_get_manifest_bootstraps_schema_when_manifest_table_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.db"
    db_path.touch()
    store = ManifestStore(str(db_path), ensure_schema=False)

    manifest = store.get_manifest()

    assert manifest is not None
    assert manifest["id"] == 1
    assert manifest["status"] == "empty"

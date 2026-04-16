from __future__ import annotations

import json
from pathlib import Path

from src.config import Config
from src.RAG.startup_bootstrap import KBaseStartupBootstrap
from src.KB.manifest_store import ManifestStore


def _build_config(tmp_path: Path, *, auto_init: bool, blocking: bool, fail_open: bool) -> Config:
    source_dir = tmp_path / "kb_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "guide.txt").write_text(
        "Amazon listing optimization requires demand, conversion and compliance checks.",
        encoding="utf-8",
    )
    db_path = tmp_path / "ecbot.db"
    payload = {
        "database": {"db_path": str(db_path)},
        "embedding": {"provider": "mock", "model": "mock-embedding-v1"},
        "knowledge_base": {
            "source_dir": str(source_dir),
            "auto_init_on_startup": auto_init,
            "init_blocking": blocking,
            "init_fail_open": fail_open,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return Config(str(config_path))


def test_bootstrap_runs_kb_init_when_empty_and_enabled(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path, auto_init=True, blocking=True, fail_open=False)
    bootstrap = KBaseStartupBootstrap(cfg)

    state = bootstrap.start()

    assert state["attempted"] is True
    assert state["completed"] is True
    assert state["success"] is True
    manifest = ManifestStore(cfg.database.db_path, ensure_schema=False).get_manifest() or {}
    assert manifest.get("status") in {"ready", "partial"}
    assert int(manifest.get("indexed_files", 0)) >= 1
    assert int(manifest.get("indexed_chunks", 0)) >= 1


def test_bootstrap_skips_when_auto_init_disabled(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path, auto_init=False, blocking=True, fail_open=False)
    bootstrap = KBaseStartupBootstrap(cfg)

    state = bootstrap.start()

    assert state["attempted"] is False
    assert state["completed"] is True
    assert state["skipped_reason"] == "auto_init_disabled"

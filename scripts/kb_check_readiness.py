from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config  # noqa: E402
from src.RAG.config.kbase_config import KBaseConfig  # noqa: E402
from src.RAG.kbase_manager import KBaseManager  # noqa: E402
from src.RAG.readiness import is_index_ready  # noqa: E402
from src.KB.manifest_store import ManifestStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check KB index readiness.")
    parser.add_argument("--config", default=None, help="Config file path. Defaults to config/config.json")
    args = parser.parse_args()

    config = Config(args.config)
    store = ManifestStore(config.database.db_path, ensure_schema=True)
    manifest = store.get_manifest() or {}
    manager = KBaseManager(
        KBaseConfig(
            db_path=config.database.db_path,
            source_dir=config.knowledge_base.source_dir,
            auto_sync_on_startup=False,
        )
    )
    stats = manager.get_statistics()
    counts = {
        "indexed_files": int(stats.get("total_files", 0) or 0),
        "indexed_chunks": int(stats.get("total_chunks", 0) or 0),
        "fts_documents": int(stats.get("fts_documents", 0) or 0),
        "vec_rows": int(stats.get("vec_rows", 0) or 0),
    }
    ready, reason, status = is_index_ready(manifest, counts)
    payload = {
        "ready": bool(ready),
        "reason": reason,
        "status": status,
        "counts": counts,
        "manifest": manifest,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

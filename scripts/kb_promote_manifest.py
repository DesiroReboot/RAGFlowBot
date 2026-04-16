from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config  # noqa: E402
from src.KB.manifest_store import ManifestStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote manifest build version as active.")
    parser.add_argument("--config", default=None, help="Config file path. Defaults to config/config.json")
    parser.add_argument(
        "--build-version",
        default="",
        help="Target build version to promote. Defaults to current config build_version",
    )
    args = parser.parse_args()

    config = Config(args.config)
    store = ManifestStore(config.database.db_path, ensure_schema=True)
    before = store.get_manifest() or {}

    build_version = str(args.build_version).strip() or str(config.knowledge_base.build_version).strip()
    if not build_version:
        print(json.dumps({"ok": False, "error": "build_version_required"}, ensure_ascii=False, indent=2))
        return 2

    store.upsert_manifest(
        status="ready",
        embedding_provider=str(before.get("embedding_provider", config.embedding.provider)),
        embedding_model=str(before.get("embedding_model", config.embedding.model)),
        embedding_dimension=int(before.get("embedding_dimension", config.embedding.dimension) or 0),
        build_version=build_version,
        indexed_files=int(before.get("indexed_files", 0) or 0),
        indexed_chunks=int(before.get("indexed_chunks", 0) or 0),
        partial_files=int(before.get("partial_files", 0) or 0),
        last_error=None,
    )
    after = store.get_manifest() or {}
    print(
        json.dumps(
            {
                "ok": True,
                "before": before,
                "after": after,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

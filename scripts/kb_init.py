from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.RAG.config.kbase_config import KBaseConfig  # noqa: E402
from src.RAG.kbase_manager import KBaseManager  # noqa: E402


def _build_kbase_config(cfg: Config, source_dir: str | None = None) -> KBaseConfig:
    resolved_source_dir = source_dir or cfg.knowledge_base.source_dir
    return KBaseConfig(
        db_path=cfg.database.db_path,
        source_dir=resolved_source_dir,
        supported_extensions=cfg.knowledge_base.supported_extensions,
        auto_sync_on_startup=False,
        ocr_enabled=cfg.knowledge_base.ocr_enabled,
        ocr_language=cfg.knowledge_base.ocr_language,
        ocr_dpi_scale=cfg.knowledge_base.ocr_dpi_scale,
        ocr_trigger_readability=cfg.knowledge_base.ocr_trigger_readability,
        min_chunk_readability=cfg.knowledge_base.min_chunk_readability,
        chunk_size=cfg.knowledge_base.chunk_size,
        chunk_overlap=cfg.knowledge_base.chunk_overlap,
        vector_dimension=cfg.embedding.dimension,
        rag_top_k=cfg.search.rag_top_k,
        fts_top_k=cfg.search.fts_top_k,
        vec_top_k=cfg.search.vec_top_k,
        fusion_rrf_k=cfg.search.fusion_rrf_k,
        context_top_k=cfg.search.context_top_k,
        embedding_provider=cfg.embedding.provider,
        embedding_base_url=cfg.embedding.base_url,
        embedding_api_key=cfg.embedding.api_key,
        embedding_model=cfg.embedding.model,
        embedding_batch_size=cfg.embedding.batch_size,
        embedding_timeout=cfg.embedding.timeout,
        embedding_max_retries=cfg.embedding.max_retries,
        build_version=cfg.knowledge_base.build_version,
    )


def _build_output(*, cfg: Config, source_dir: str, sync_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "db_path": str(cfg.database.db_path),
        "source_dir": source_dir,
        "sync_result": sync_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize or sync KB index from configured source_dir")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config file path. Default uses ECBOT_CONFIG_PATH or config/config.json",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Override knowledge_base.source_dir for this run",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild index for all scanned files even when file hash is unchanged",
    )
    args = parser.parse_args()

    try:
        cfg = Config(config_path=args.config)
        source_dir = str(args.source_dir or cfg.knowledge_base.source_dir)
        manager = KBaseManager(_build_kbase_config(cfg, source_dir=source_dir))
        sync_result = manager.sync_configured_source(force_reindex=bool(args.force_reindex))
        payload = _build_output(cfg=cfg, source_dir=source_dir, sync_result=sync_result)
        print(json.dumps(payload, ensure_ascii=False, indent=2))

        failed = int(sync_result.get("failed", 0) or 0)
        errors = sync_result.get("errors", [])
        if failed > 0 or (isinstance(errors, list) and len(errors) > 0):
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

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
from src.RAG.progress import create_reporter  # noqa: E402


def _build_kbase_config(config: Config, source_dir: str | None) -> KBaseConfig:
    return KBaseConfig(
        db_path=config.database.db_path,
        source_dir=source_dir or config.knowledge_base.source_dir,
        supported_extensions=config.knowledge_base.supported_extensions,
        auto_sync_on_startup=False,
        ocr_enabled=config.knowledge_base.ocr_enabled,
        ocr_language=config.knowledge_base.ocr_language,
        ocr_dpi_scale=config.knowledge_base.ocr_dpi_scale,
        ocr_trigger_readability=config.knowledge_base.ocr_trigger_readability,
        min_chunk_readability=config.knowledge_base.min_chunk_readability,
        chunk_size=config.knowledge_base.chunk_size,
        chunk_overlap=config.knowledge_base.chunk_overlap,
        vector_dimension=config.embedding.dimension,
        rag_top_k=config.search.rag_top_k,
        fts_top_k=config.search.fts_top_k,
        vec_top_k=config.search.vec_top_k,
        fusion_rrf_k=config.search.fusion_rrf_k,
        context_top_k=config.search.context_top_k,
        embedding_provider=config.embedding.provider,
        embedding_base_url=config.embedding.base_url,
        embedding_api_key=config.embedding.api_key,
        embedding_model=config.embedding.model,
        embedding_batch_size=config.embedding.batch_size,
        embedding_timeout=config.embedding.timeout,
        embedding_max_retries=config.embedding.max_retries,
        build_version=config.knowledge_base.build_version,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or refresh KB index.")
    parser.add_argument(
        "--config", default=None, help="Config file path. Defaults to config/config.json"
    )
    parser.add_argument("--source-dir", default=None, help="Override knowledge base source directory")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild index for all scanned files even when file hash is unchanged",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="禁用进度条（仅输出 JSON 结果）"
    )
    parser.add_argument(
        "--progress-type",
        choices=["auto", "rich", "json", "none"],
        default="auto",
        help=(
            "进度条类型 (默认: auto) "
            "auto - 自动检测终端环境 "
            "rich - 始终显示 Rich 进度条 "
            "json - 仅输出 JSON 格式结果 "
            "none - 无进度输出"
        ),
    )
    args = parser.parse_args()

    config = Config(args.config)
    kbase_config = _build_kbase_config(config, args.source_dir)

    # Determine progress reporter type
    progress_type = args.progress_type
    if args.no_progress or progress_type == "none":
        progress_type = "json"

    # Create progress reporter
    reporter = create_reporter(progress_type)

    manager = KBaseManager(kbase_config, progress_reporter=reporter)
    result = manager.scan_and_process(
        kbase_config.source_dir,
        force_reindex=bool(args.force_reindex),
    )

    # Only output JSON if not using rich reporter (rich reporter already displays results)
    if progress_type != "rich":
        print(json.dumps(result, ensure_ascii=False, indent=2))

    failed = int(result.get("failed", 0) or 0)
    errors = result.get("errors", [])
    return 1 if failed > 0 or (isinstance(errors, list) and bool(errors)) else 0


if __name__ == "__main__":
    raise SystemExit(main())

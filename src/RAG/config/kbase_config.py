from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass
class KBaseConfig:
    db_path: str = "DB/ec_bot.db"
    source_dir: str = r"E:\DATA\外贸电商知识库"
    supported_extensions: tuple[str, ...] = (".md", ".txt", ".pdf", ".json", ".xml")
    auto_sync_on_startup: bool = False
    auto_classification: bool = True

    ocr_enabled: bool = True
    ocr_language: str = "chi_sim+eng"
    ocr_dpi_scale: float = 2.0
    ocr_trigger_readability: float = 0.58
    min_chunk_readability: float = 0.38

    chunk_size: int = 400
    chunk_overlap: int = 80

    vector_dimension: int = 768
    rag_top_k: int = 5
    fts_top_k: int = 20
    vec_top_k: int = 20
    fusion_rrf_k: int = 60
    context_top_k: int = 6

    embedding_provider: str = "dashscope"
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_batch_size: int = 10
    embedding_timeout: int = 20
    embedding_max_retries: int = 3

    build_version: str = "rag-v2"

    @classmethod
    def from_env(cls) -> "KBaseConfig":
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        def _env_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or str(raw).strip() == "":
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or str(raw).strip() == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        defaults = cls()
        data: dict[str, Any] = {
            "db_path": os.getenv("KBASE_DB_PATH", defaults.db_path),
            "source_dir": os.getenv("KBASE_SOURCE_DIR", defaults.source_dir),
            "auto_sync_on_startup": _env_bool("KBASE_AUTO_SYNC_ON_STARTUP", defaults.auto_sync_on_startup),
            "auto_classification": _env_bool("KBASE_AUTO_CLASSIFICATION", defaults.auto_classification),
            "ocr_enabled": _env_bool("KBASE_OCR_ENABLED", defaults.ocr_enabled),
            "ocr_language": os.getenv("KBASE_OCR_LANGUAGE", defaults.ocr_language),
            "ocr_dpi_scale": _env_float("KBASE_OCR_DPI_SCALE", defaults.ocr_dpi_scale),
            "ocr_trigger_readability": _env_float(
                "KBASE_OCR_TRIGGER_READABILITY",
                defaults.ocr_trigger_readability,
            ),
            "min_chunk_readability": _env_float(
                "KBASE_MIN_CHUNK_READABILITY",
                defaults.min_chunk_readability,
            ),
            "chunk_size": _env_int("KBASE_CHUNK_SIZE", defaults.chunk_size),
            "chunk_overlap": _env_int("KBASE_CHUNK_OVERLAP", defaults.chunk_overlap),
            "vector_dimension": _env_int("KBASE_VECTOR_DIMENSION", defaults.vector_dimension),
            "rag_top_k": _env_int("KBASE_RAG_TOP_K", defaults.rag_top_k),
            "fts_top_k": _env_int("KBASE_FTS_TOP_K", defaults.fts_top_k),
            "vec_top_k": _env_int("KBASE_VEC_TOP_K", defaults.vec_top_k),
            "fusion_rrf_k": _env_int("KBASE_FUSION_RRF_K", defaults.fusion_rrf_k),
            "context_top_k": _env_int("KBASE_CONTEXT_TOP_K", defaults.context_top_k),
            "embedding_provider": os.getenv("KBASE_EMBEDDING_PROVIDER", defaults.embedding_provider),
            "embedding_base_url": os.getenv("KBASE_EMBEDDING_BASE_URL", defaults.embedding_base_url),
            "embedding_api_key": os.getenv("KBASE_EMBEDDING_API_KEY", defaults.embedding_api_key),
            "embedding_model": os.getenv("KBASE_EMBEDDING_MODEL", defaults.embedding_model),
            "embedding_batch_size": _env_int(
                "KBASE_EMBEDDING_BATCH_SIZE",
                defaults.embedding_batch_size,
            ),
            "embedding_timeout": _env_int("KBASE_EMBEDDING_TIMEOUT", defaults.embedding_timeout),
            "embedding_max_retries": _env_int(
                "KBASE_EMBEDDING_MAX_RETRIES",
                defaults.embedding_max_retries,
            ),
            "build_version": os.getenv("KBASE_BUILD_VERSION", defaults.build_version),
        }
        return cls(**data)

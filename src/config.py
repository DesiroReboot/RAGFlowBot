from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        items = [str(item).strip().lower() for item in value if str(item).strip()]
        return tuple(items) if items else default
    return default


def _as_dict(value: Any, default: dict[str, Any]) -> dict[str, Any]:
    if value is None:
        return dict(default)
    if isinstance(value, dict):
        merged = dict(default)
        merged.update(value)
        return merged
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return dict(default)
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return dict(default)
        if isinstance(loaded, dict):
            merged = dict(default)
            merged.update(loaded)
            return merged
    return dict(default)


def _as_str_dict(value: Any, default: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = dict(default)
    raw = _as_dict(value, {})
    for key, item in raw.items():
        key_text = str(key).strip()
        item_text = str(item).strip()
        if not key_text or not item_text:
            continue
        merged[key_text] = item_text
    return merged


def _is_invalid_config_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True
        return _is_placeholder_env_value(text)
    return False


def _env(name: str, default: Any) -> Any:
    if not _is_invalid_config_value(default):
        return default
    raw = os.getenv(name)
    if raw is not None:
        return raw
    return default


def _resolve_env_first(name: str, config_value: Any, default: Any) -> Any:
    raw = os.getenv(name)
    if raw is not None:
        return raw
    if config_value is not None:
        return config_value
    return default


def _resolve_config_first(name: str, config_value: Any, default: Any) -> Any:
    if not _is_invalid_config_value(config_value):
        return config_value
    raw = os.getenv(name)
    if raw is not None:
        return raw
    return default


def _resolve_by_authority(
    *,
    name: str,
    config_value: Any,
    default: Any,
    authority: str,
    prefer: str,
) -> Any:
    del authority, prefer
    return _resolve_config_first(name, config_value, default)


def _strip_wrapped_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def _is_placeholder_env_value(value: str) -> bool:
    normalized = _strip_wrapped_quotes(str(value)).strip().upper()
    return bool(normalized) and normalized.startswith("YOUR_")


def _load_dotenv_file(path: str = ".env", *, ignore_placeholders: bool = False) -> bool:
    dotenv_path = Path(path)
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return False

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_wrapped_quotes(value.strip())
        if ignore_placeholders and _is_placeholder_env_value(value):
            continue
        os.environ.setdefault(key, value)
    return True


def _load_env_layers() -> None:
    # Load order (high -> low priority) with setdefault semantics:
    # 1) project-local .env
    # 2) .env.example as a non-secret fallback template
    #
    # This intentionally avoids reading dotenv paths from external environment pointers
    # such as ECBOT_DOTENV_PATH, so runtime config only comes from the current project.
    _load_dotenv_file(".env")
    _load_dotenv_file(".env.example", ignore_placeholders=True)


@dataclass
class SearchConfig:
    rag_provider: str = "legacy"
    rag_top_k: int = 5
    fts_top_k: int = 20
    vec_top_k: int = 20
    fusion_rrf_k: int = 60
    context_top_k: int = 6
    source_quota_mode: str = "unbounded"
    max_chunks_per_source: int = 0
    web_search_enabled: bool = False
    web_search_provider: str = "mock"
    web_search_timeout: int = 8
    web_search_retries: int = 1
    web_search_tavily_api_key: str = ""
    web_search_tavily_base_url: str = "https://api.tavily.com/search"
    web_search_max_results: int = 8
    web_search_depth: str = "basic"
    web_direct_fusion_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "result_count_max": 8.0,
            "top3_mean_min": 0.72,
            "score_gap_min": 0.08,
            "noise_ratio_max": 0.25,
        }
    )
    web_rag_max_docs: int = 16
    phase_a_rag_confidence_threshold: float = 0.58
    l1_trigger_threshold: float = 0.58
    l1_template_enabled: bool = True
    l2_max_top_k: int = 8
    merge_web_trigger_requires_rag_gap: bool = True
    merge_trigger_on_kb_empty: bool = True
    merge_trigger_on_low_confidence: bool = True
    merge_step_min_evidence: int = 1
    merge_evidence_rag_top_k: int = 3
    merge_evidence_search_top_k: int = 2
    search_progress_enabled: bool = True
    search_progress_keyword_top_k: int = 4
    qa_anchor_enabled: bool = True
    semantic_guard_enabled: bool = True
    paragraph_output_enabled: bool = True


@dataclass
class DatabaseConfig:
    db_path: str = "DB/ec_bot.db"


@dataclass
class EmbeddingConfig:
    provider: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model: str = "text-embedding-v4"
    dimension: int = 768
    batch_size: int = 10
    timeout: int = 20
    max_retries: int = 3


@dataclass
class GenerationConfig:
    mode: str = "hybrid"
    provider: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model: str = "qwen3-32b"
    temperature: float = 0.2
    timeout: int = 25
    max_retries: int = 2
    min_quality_score: float = 0.55
    min_claim_support_rate: float = 0.35
    min_citation_coverage: float = 0.6
    default_answer_mode: str = "fact_qa"
    force_point_source_format: bool = True
    min_point_source_binding_rate: float = 1.0


@dataclass
class KnowledgeBaseConfig:
    source_dir: str = r"E:\知识库\RailKB"
    supported_extensions: tuple[str, ...] = (".md", ".txt", ".pdf")
    auto_sync_on_startup: bool = False
    auto_init_on_startup: bool = False
    init_blocking: bool = False
    init_fail_open: bool = True
    ocr_enabled: bool = True
    ocr_language: str = "chi_sim+eng"
    ocr_dpi_scale: float = 2.0
    ocr_trigger_readability: float = 0.58
    min_chunk_readability: float = 0.38
    chunk_size: int = 400
    chunk_overlap: int = 80
    build_version: str = "rag-v2"


@dataclass
class GatewayFeishuConfig:
    enabled: bool = False
    openapi_base_url: str = "https://open.feishu.cn/open-apis"
    app_id: str = ""
    app_secret: str = ""
    receive_mode: str = "long_connection"
    long_conn_log_level: str = "INFO"
    encrypt_key: str = ""
    verification_token: str = ""
    bot_name: str = "ECBot"
    target_chat_id: str = ""
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8000
    webhook_path: str = "/webhook/feishu"
    request_timeout: int = 30


@dataclass
class GatewayConfig:
    feishu: GatewayFeishuConfig


@dataclass
class OutputGuardrailConfig:
    enabled: bool = False
    min_retrieval_confidence: float = 0.3


@dataclass
class GuardrailsConfig:
    output: OutputGuardrailConfig


@dataclass
class EvaluationConfig:
    quality_gate_enabled: bool = False
    report_dir: str = "Eval/report"
    trace_dir: str = "Eval/trace"
    html_dir: str = "Eval/HTML"


@dataclass
class RagFlowConfig:
    base_url: str = ""
    api_key: str = ""
    dataset_map: dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 2500
    top_k: int = 5
    min_score: float = 0.1
    fallback_to_legacy: bool = True


class Config:
    DEFAULT_CONFIG_PATH = "config/config.json"
    DEFAULT_KB_SOURCE_DIR = r"E:\知识库\RailKB"

    def __init__(self, config_path: str | None = None):
        _load_env_layers()

        resolved_config_path = config_path or os.getenv("ECBOT_CONFIG_PATH") or self.DEFAULT_CONFIG_PATH
        # Backward-compatible fallback for legacy single-file layout.
        if (
            resolved_config_path == self.DEFAULT_CONFIG_PATH
            and not Path(resolved_config_path).exists()
            and Path("config.json").exists()
        ):
            resolved_config_path = "config.json"
        self.config_path = resolved_config_path
        data = self._load_json(self.config_path)

        search_data = data.get("search", {})
        ragflow_data = data.get("ragflow", {})
        db_data = data.get("database", {})
        embedding_data = data.get("embedding", {})
        generation_data = data.get("generation", {})
        knowledge_base_data = data.get("knowledge_base", {})
        gateway_data = data.get("gateway", {}).get("feishu", {})
        config_authority = str(os.getenv("ECBOT_CONFIG_AUTHORITY", "config")).strip().lower() or "config"
        output_guardrail_data = data.get("guardrails", {}).get("output", {})
        evaluation_data = data.get("evaluation", {})
        web_threshold_defaults = {
            "result_count_max": 8.0,
            "top3_mean_min": 0.72,
            "score_gap_min": 0.08,
            "noise_ratio_max": 0.25,
        }
        web_thresholds = _as_dict(
            _env(
                "ECBOT_WEB_DIRECT_FUSION_THRESHOLDS",
                search_data.get("web_direct_fusion_thresholds"),
            ),
            web_threshold_defaults,
        )
        merge_web_trigger_requires_rag_gap = _as_bool(
            _env(
                "ECBOT_MERGE_WEB_TRIGGER_REQUIRES_RAG_GAP",
                search_data.get(
                    "merge_web_trigger_requires_rag_gap",
                    search_data.get("web_trigger_requires_rag_gap", True),
                ),
            ),
            True,
        )
        merge_trigger_on_kb_empty = _as_bool(
            _env(
                "ECBOT_MERGE_TRIGGER_ON_KB_EMPTY",
                search_data.get(
                    "merge_trigger_on_kb_empty",
                    search_data.get("trigger_on_kb_empty", True),
                ),
            ),
            True,
        )
        merge_trigger_on_low_confidence = _as_bool(
            _env(
                "ECBOT_MERGE_TRIGGER_ON_LOW_CONFIDENCE",
                search_data.get(
                    "merge_trigger_on_low_confidence",
                    search_data.get("trigger_on_low_confidence", True),
                ),
            ),
            True,
        )
        merge_step_min_evidence = int(
            _env(
                "ECBOT_MERGE_STEP_MIN_EVIDENCE",
                search_data.get(
                    "merge_step_min_evidence",
                    search_data.get("step_min_evidence", 1),
                ),
            )
        )
        merge_evidence_rag_top_k = int(
            _env(
                "ECBOT_MERGE_EVIDENCE_RAG_TOP_K",
                search_data.get(
                    "merge_evidence_rag_top_k",
                    search_data.get("evidence_rag_top_k", 3),
                ),
            )
        )
        merge_evidence_search_top_k = int(
            _env(
                "ECBOT_MERGE_EVIDENCE_SEARCH_TOP_K",
                search_data.get(
                    "merge_evidence_search_top_k",
                    search_data.get("evidence_search_top_k", 2),
                ),
            )
        )
        kb_auto_config_value = knowledge_base_data.get(
            "auto_init_on_startup",
            knowledge_base_data.get("auto_sync_on_startup"),
        )
        kb_auto_startup = _as_bool(
            _resolve_by_authority(
                name="ECBOT_KB_AUTO_INIT_ON_STARTUP",
                config_value=kb_auto_config_value,
                default=_env("ECBOT_KB_AUTO_SYNC", False),
                authority=config_authority,
                prefer="config",
            ),
            False,
        )
        kb_init_blocking = _as_bool(
            _resolve_by_authority(
                name="ECBOT_KB_INIT_BLOCKING",
                config_value=knowledge_base_data.get("init_blocking"),
                default=False,
                authority=config_authority,
                prefer="config",
            ),
            False,
        )
        kb_init_fail_open = _as_bool(
            _resolve_by_authority(
                name="ECBOT_KB_INIT_FAIL_OPEN",
                config_value=knowledge_base_data.get("init_fail_open"),
                default=True,
                authority=config_authority,
                prefer="config",
            ),
            True,
        )
        shared_model_name = str(os.getenv("ECBOT_MODEL", "")).strip()
        embedding_model_default = str(embedding_data.get("model", "text-embedding-v4")).strip()
        generation_model_default = str(generation_data.get("model", "qwen-plus")).strip()
        resolved_embedding_model = str(
            _resolve_by_authority(
                name="ECBOT_EMBEDDING_MODEL",
                config_value=embedding_data.get("model"),
                default=shared_model_name or embedding_model_default,
                authority=config_authority,
                prefer="config",
            )
        ).strip()
        resolved_generation_model = str(
            _resolve_by_authority(
                name="ECBOT_GENERATION_MODEL",
                config_value=generation_data.get("model"),
                default=shared_model_name or generation_model_default,
                authority=config_authority,
                prefer="config",
            )
        ).strip()

        self.search = SearchConfig(
            rag_provider=str(_env("ECBOT_RAG_PROVIDER", search_data.get("rag_provider", "legacy")))
            .strip()
            .lower(),
            rag_top_k=int(_env("ECBOT_RAG_TOP_K", search_data.get("rag_top_k", 5))),
            fts_top_k=int(_env("ECBOT_FTS_TOP_K", search_data.get("fts_top_k", 20))),
            vec_top_k=int(_env("ECBOT_VEC_TOP_K", search_data.get("vec_top_k", 20))),
            fusion_rrf_k=int(_env("ECBOT_FUSION_RRF_K", search_data.get("fusion_rrf_k", 60))),
            context_top_k=int(_env("ECBOT_CONTEXT_TOP_K", search_data.get("context_top_k", 6))),
            source_quota_mode=str(
                _env(
                    "ECBOT_SOURCE_QUOTA_MODE",
                    search_data.get("source_quota_mode", "unbounded"),
                )
            )
            .strip()
            .lower(),
            max_chunks_per_source=max(
                0,
                int(
                    _env(
                        "ECBOT_MAX_CHUNKS_PER_SOURCE",
                        search_data.get("max_chunks_per_source", 0),
                    )
                ),
            ),
            web_search_enabled=_as_bool(
                _env("ECBOT_WEB_SEARCH_ENABLED", search_data.get("web_search_enabled", False)),
                False,
            ),
            web_search_provider=str(
                _env("ECBOT_WEB_SEARCH_PROVIDER", search_data.get("web_search_provider", "mock"))
            ).strip(),
            web_search_timeout=int(
                _env("ECBOT_WEB_SEARCH_TIMEOUT", search_data.get("web_search_timeout", 8))
            ),
            web_search_retries=int(
                _env("ECBOT_WEB_SEARCH_RETRIES", search_data.get("web_search_retries", 1))
            ),
            web_search_tavily_api_key=str(
                _env(
                    "ECBOT_TAVILY_API_KEY",
                    search_data.get("web_search_tavily_api_key", ""),
                )
            ).strip(),
            web_search_tavily_base_url=str(
                _env(
                    "ECBOT_TAVILY_BASE_URL",
                    search_data.get(
                        "web_search_tavily_base_url",
                        "https://api.tavily.com/search",
                    ),
                )
            ).strip(),
            web_search_max_results=int(
                _env(
                    "ECBOT_WEB_SEARCH_MAX_RESULTS",
                    search_data.get("web_search_max_results", 8),
                )
            ),
            web_search_depth=str(
                _env(
                    "ECBOT_WEB_SEARCH_DEPTH",
                    search_data.get("web_search_depth", "basic"),
                )
            ).strip(),
            web_direct_fusion_thresholds={
                "result_count_max": float(
                    web_thresholds.get(
                        "result_count_max", web_threshold_defaults["result_count_max"]
                    )
                ),
                "top3_mean_min": float(
                    web_thresholds.get("top3_mean_min", web_threshold_defaults["top3_mean_min"])
                ),
                "score_gap_min": float(
                    web_thresholds.get("score_gap_min", web_threshold_defaults["score_gap_min"])
                ),
                "noise_ratio_max": float(
                    web_thresholds.get(
                        "noise_ratio_max", web_threshold_defaults["noise_ratio_max"]
                    )
                ),
            },
            web_rag_max_docs=int(
                _env("ECBOT_WEB_RAG_MAX_DOCS", search_data.get("web_rag_max_docs", 16))
            ),
            phase_a_rag_confidence_threshold=float(
                _env(
                    "ECBOT_PHASE_A_RAG_CONFIDENCE_THRESHOLD",
                    search_data.get("phase_a_rag_confidence_threshold", 0.58),
                )
            ),
            l1_trigger_threshold=float(
                _env(
                    "ECBOT_L1_TRIGGER_THRESHOLD",
                    search_data.get("l1_trigger_threshold", 0.58),
                )
            ),
            l1_template_enabled=_as_bool(
                _env(
                    "ECBOT_L1_TEMPLATE_ENABLED",
                    search_data.get("l1_template_enabled", True),
                ),
                True,
            ),
            l2_max_top_k=int(
                _env(
                    "ECBOT_L2_MAX_TOP_K",
                    search_data.get("l2_max_top_k", 8),
                )
            ),
            merge_web_trigger_requires_rag_gap=merge_web_trigger_requires_rag_gap,
            merge_trigger_on_kb_empty=merge_trigger_on_kb_empty,
            merge_trigger_on_low_confidence=merge_trigger_on_low_confidence,
            merge_step_min_evidence=merge_step_min_evidence,
            merge_evidence_rag_top_k=merge_evidence_rag_top_k,
            merge_evidence_search_top_k=merge_evidence_search_top_k,
            search_progress_enabled=_as_bool(
                _env(
                    "ECBOT_SEARCH_PROGRESS_ENABLED",
                    search_data.get("search_progress_enabled", True),
                ),
                True,
            ),
            search_progress_keyword_top_k=int(
                _env(
                    "ECBOT_SEARCH_PROGRESS_KEYWORD_TOP_K",
                    search_data.get("search_progress_keyword_top_k", 4),
                )
            ),
            qa_anchor_enabled=_as_bool(
                _env(
                    "ECBOT_QA_ANCHOR_ENABLED",
                    search_data.get("qa_anchor_enabled", True),
                ),
                True,
            ),
            semantic_guard_enabled=_as_bool(
                _env(
                    "ECBOT_SEMANTIC_GUARD_ENABLED",
                    search_data.get("semantic_guard_enabled", True),
                ),
                True,
            ),
            paragraph_output_enabled=_as_bool(
                _env(
                    "ECBOT_PARAGRAPH_OUTPUT_ENABLED",
                    search_data.get("paragraph_output_enabled", True),
                ),
                True,
            ),
        )
        self.database = DatabaseConfig(
            db_path=str(_env("ECBOT_DB_PATH", db_data.get("db_path", "DB/ec_bot.db")))
        )
        self.embedding = EmbeddingConfig(
            provider=str(_env("ECBOT_EMBEDDING_PROVIDER", embedding_data.get("provider", "dashscope"))),
            base_url=str(_env("ECBOT_EMBEDDING_BASE_URL", embedding_data.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))),
            api_key=str(_env("ECBOT_EMBEDDING_API_KEY", embedding_data.get("api_key", ""))),
            model=resolved_embedding_model,
            dimension=int(_env("ECBOT_EMBEDDING_DIMENSION", embedding_data.get("dimension", 768))),
            batch_size=int(_env("ECBOT_EMBEDDING_BATCH_SIZE", embedding_data.get("batch_size", 10))),
            timeout=int(_env("ECBOT_EMBEDDING_TIMEOUT", embedding_data.get("timeout", 20))),
            max_retries=int(_env("ECBOT_EMBEDDING_MAX_RETRIES", embedding_data.get("max_retries", 3))),
        )
        self.generation = GenerationConfig(
            mode=str(_env("ECBOT_GENERATION_MODE", generation_data.get("mode", "hybrid"))).strip().lower(),
            provider=str(_env("ECBOT_GENERATION_PROVIDER", generation_data.get("provider", "dashscope"))),
            base_url=str(
                _env(
                    "ECBOT_GENERATION_BASE_URL",
                    generation_data.get("base_url", embedding_data.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
                )
            ),
            api_key=str(
                _env(
                    "ECBOT_GENERATION_API_KEY",
                    generation_data.get("api_key", embedding_data.get("api_key", "")),
                )
            ),
            model=resolved_generation_model,
            temperature=float(_env("ECBOT_GENERATION_TEMPERATURE", generation_data.get("temperature", 0.2))),
            timeout=int(_env("ECBOT_GENERATION_TIMEOUT", generation_data.get("timeout", 25))),
            max_retries=int(_env("ECBOT_GENERATION_MAX_RETRIES", generation_data.get("max_retries", 2))),
            min_quality_score=float(
                _env(
                    "ECBOT_GENERATION_MIN_QUALITY_SCORE",
                    generation_data.get("min_quality_score", 0.55),
                )
            ),
            min_claim_support_rate=float(
                _env(
                    "ECBOT_GENERATION_MIN_CLAIM_SUPPORT_RATE",
                    generation_data.get("min_claim_support_rate", 0.35),
                )
            ),
            min_citation_coverage=float(
                _env(
                    "ECBOT_GENERATION_MIN_CITATION_COVERAGE",
                    generation_data.get("min_citation_coverage", 0.6),
                )
            ),
            default_answer_mode=str(
                _env(
                    "ECBOT_GENERATION_DEFAULT_ANSWER_MODE",
                    generation_data.get("default_answer_mode", "fact_qa"),
                )
            )
            .strip()
            .lower(),
            force_point_source_format=_as_bool(
                _env(
                    "ECBOT_GENERATION_FORCE_POINT_SOURCE_FORMAT",
                    generation_data.get("force_point_source_format", True),
                ),
                True,
            ),
            min_point_source_binding_rate=float(
                _env(
                    "ECBOT_GENERATION_MIN_POINT_SOURCE_BINDING_RATE",
                    generation_data.get("min_point_source_binding_rate", 1.0),
                )
            ),
        )
        self.knowledge_base = KnowledgeBaseConfig(
            source_dir=str(
                _env(
                    "ECBOT_KB_SOURCE_DIR",
                    knowledge_base_data.get("source_dir", self.DEFAULT_KB_SOURCE_DIR),
                )
            ),
            supported_extensions=_as_tuple(
                knowledge_base_data.get("supported_extensions"),
                (".md", ".txt", ".pdf"),
            ),
            auto_sync_on_startup=kb_auto_startup,
            auto_init_on_startup=kb_auto_startup,
            init_blocking=kb_init_blocking,
            init_fail_open=kb_init_fail_open,
            ocr_enabled=_as_bool(
                _env("ECBOT_KB_OCR_ENABLED", knowledge_base_data.get("ocr_enabled", True)),
                True,
            ),
            ocr_language=str(
                _env("ECBOT_KB_OCR_LANGUAGE", knowledge_base_data.get("ocr_language", "chi_sim+eng"))
            ),
            ocr_dpi_scale=float(
                _env("ECBOT_KB_OCR_DPI_SCALE", knowledge_base_data.get("ocr_dpi_scale", 2.0))
            ),
            ocr_trigger_readability=float(
                _env(
                    "ECBOT_KB_OCR_TRIGGER_READABILITY",
                    knowledge_base_data.get("ocr_trigger_readability", 0.58),
                )
            ),
            min_chunk_readability=float(
                _env(
                    "ECBOT_KB_MIN_CHUNK_READABILITY",
                    knowledge_base_data.get("min_chunk_readability", 0.38),
                )
            ),
            chunk_size=int(_env("ECBOT_CHUNK_SIZE", knowledge_base_data.get("chunk_size", 400))),
            chunk_overlap=int(_env("ECBOT_CHUNK_OVERLAP", knowledge_base_data.get("chunk_overlap", 80))),
            build_version=str(_env("ECBOT_BUILD_VERSION", knowledge_base_data.get("build_version", "rag-v2"))),
        )
        self.gateway = GatewayConfig(
            feishu=GatewayFeishuConfig(
                enabled=_as_bool(
                    _resolve_by_authority(
                        name="ECBOT_FEISHU_ENABLED",
                        config_value=gateway_data.get("enabled"),
                        default=False,
                        authority=config_authority,
                        prefer="config",
                    ),
                    False,
                ),
                openapi_base_url=str(
                    _env(
                        "ECBOT_FEISHU_OPENAPI_BASE_URL",
                        gateway_data.get("openapi_base_url", "https://open.feishu.cn/open-apis"),
                    )
                ),
                app_id=str(_env("ECBOT_FEISHU_APP_ID", gateway_data.get("app_id", ""))),
                app_secret=str(_env("ECBOT_FEISHU_APP_SECRET", gateway_data.get("app_secret", ""))),
                receive_mode=str(
                    _resolve_by_authority(
                        name="ECBOT_FEISHU_RECEIVE_MODE",
                        config_value=gateway_data.get("receive_mode"),
                        default="long_connection",
                        authority=config_authority,
                        prefer="config",
                    )
                )
                .strip()
                .lower(),
                long_conn_log_level=str(
                    _env(
                        "ECBOT_FEISHU_LONG_CONN_LOG_LEVEL",
                        gateway_data.get("long_conn_log_level", "INFO"),
                    )
                ).strip().upper(),
                encrypt_key=str(_env("ECBOT_FEISHU_ENCRYPT_KEY", gateway_data.get("encrypt_key", ""))),
                verification_token=str(
                    _env(
                        "ECBOT_FEISHU_VERIFICATION_TOKEN",
                        gateway_data.get("verification_token", ""),
                    )
                ),
                bot_name=str(_env("ECBOT_FEISHU_BOT_NAME", gateway_data.get("bot_name", "ECBot"))),
                target_chat_id=str(
                    _env("ECBOT_FEISHU_TARGET_CHAT_ID", gateway_data.get("target_chat_id", ""))
                ),
                webhook_host=str(
                    _resolve_by_authority(
                        name="ECBOT_FEISHU_WEBHOOK_HOST",
                        config_value=gateway_data.get("webhook_host"),
                        default="127.0.0.1",
                        authority=config_authority,
                        prefer="config",
                    )
                ),
                webhook_port=int(
                    _resolve_by_authority(
                        name="ECBOT_FEISHU_WEBHOOK_PORT",
                        config_value=gateway_data.get("webhook_port"),
                        default=8000,
                        authority=config_authority,
                        prefer="config",
                    )
                ),
                webhook_path=str(
                    _resolve_by_authority(
                        name="ECBOT_FEISHU_WEBHOOK_PATH",
                        config_value=gateway_data.get("webhook_path"),
                        default="/webhook/feishu",
                        authority=config_authority,
                        prefer="config",
                    )
                ),
                request_timeout=int(gateway_data.get("request_timeout", 30)),
            )
        )
        self.guardrails = GuardrailsConfig(
            output=OutputGuardrailConfig(
                enabled=_as_bool(output_guardrail_data.get("enabled"), False),
                min_retrieval_confidence=float(
                    output_guardrail_data.get("min_retrieval_confidence", 0.3)
                ),
            )
        )
        self.evaluation = EvaluationConfig(
            quality_gate_enabled=_as_bool(
                evaluation_data.get("quality_gate_enabled"),
                False,
            ),
            report_dir=str(evaluation_data.get("report_dir", "Eval/report")),
            trace_dir=str(evaluation_data.get("trace_dir", "Eval/trace")),
            html_dir=str(evaluation_data.get("html_dir", "Eval/HTML")),
        )
        self.ragflow = RagFlowConfig(
            base_url=str(_env("ECBOT_RAGFLOW_BASE_URL", ragflow_data.get("base_url", ""))).strip(),
            api_key=str(_env("ECBOT_RAGFLOW_API_KEY", ragflow_data.get("api_key", ""))).strip(),
            dataset_map=_as_str_dict(
                _env("ECBOT_RAGFLOW_DATASET_MAP", ragflow_data.get("dataset_map", {})),
                {},
            ),
            timeout_ms=max(
                200,
                int(_env("ECBOT_RAGFLOW_TIMEOUT_MS", ragflow_data.get("timeout_ms", 2500))),
            ),
            top_k=max(
                1,
                int(_env("ECBOT_RAGFLOW_TOP_K", ragflow_data.get("top_k", 5))),
            ),
            min_score=float(_env("ECBOT_RAGFLOW_MIN_SCORE", ragflow_data.get("min_score", 0.1))),
            fallback_to_legacy=_as_bool(
                _env(
                    "ECBOT_RAGFLOW_FALLBACK_TO_LEGACY",
                    ragflow_data.get("fallback_to_legacy", True),
                ),
                True,
            ),
        )

    def _load_json(self, path_str: str) -> dict[str, Any]:
        path = Path(path_str)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)

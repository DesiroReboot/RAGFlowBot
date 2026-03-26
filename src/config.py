from __future__ import annotations

from dataclasses import dataclass
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


def _env(name: str, default: Any) -> Any:
    return os.getenv(name, default)


def _strip_wrapped_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def _load_dotenv_file(path: str = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

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
        os.environ.setdefault(key, value)


@dataclass
class SearchConfig:
    rag_top_k: int = 5
    fts_top_k: int = 20
    vec_top_k: int = 20
    fusion_rrf_k: int = 60
    context_top_k: int = 6
    domain_filter_enabled: bool = True
    domain_filter_threshold: float = 0.45
    domain_filter_fail_open: bool = True


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
    model: str = "qwen-plus"
    temperature: float = 0.2
    timeout: int = 25
    max_retries: int = 2
    min_quality_score: float = 0.55
    min_claim_support_rate: float = 0.35
    min_citation_coverage: float = 0.6


@dataclass
class KnowledgeBaseConfig:
    source_dir: str = r"E:\DATA\外贸电商知识库"
    supported_extensions: tuple[str, ...] = (".md", ".txt", ".pdf")
    auto_sync_on_startup: bool = False
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


class Config:
    DEFAULT_CONFIG_PATH = "config/config.json"

    def __init__(self, config_path: str | None = None):
        dotenv_path = os.getenv("ECBOT_DOTENV_PATH", ".env")
        _load_dotenv_file(dotenv_path)

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
        db_data = data.get("database", {})
        embedding_data = data.get("embedding", {})
        generation_data = data.get("generation", {})
        knowledge_base_data = data.get("knowledge_base", {})
        gateway_data = data.get("gateway", {}).get("feishu", {})
        output_guardrail_data = data.get("guardrails", {}).get("output", {})
        evaluation_data = data.get("evaluation", {})

        self.search = SearchConfig(
            rag_top_k=int(_env("ECBOT_RAG_TOP_K", search_data.get("rag_top_k", 5))),
            fts_top_k=int(_env("ECBOT_FTS_TOP_K", search_data.get("fts_top_k", 20))),
            vec_top_k=int(_env("ECBOT_VEC_TOP_K", search_data.get("vec_top_k", 20))),
            fusion_rrf_k=int(_env("ECBOT_FUSION_RRF_K", search_data.get("fusion_rrf_k", 60))),
            context_top_k=int(_env("ECBOT_CONTEXT_TOP_K", search_data.get("context_top_k", 6))),
            domain_filter_enabled=_as_bool(
                _env(
                    "ECBOT_DOMAIN_FILTER_ENABLED",
                    search_data.get("domain_filter_enabled", True),
                ),
                True,
            ),
            domain_filter_threshold=float(
                _env(
                    "ECBOT_DOMAIN_FILTER_THRESHOLD",
                    search_data.get("domain_filter_threshold", 0.45),
                )
            ),
            domain_filter_fail_open=_as_bool(
                _env(
                    "ECBOT_DOMAIN_FILTER_FAIL_OPEN",
                    search_data.get("domain_filter_fail_open", True),
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
            model=str(_env("ECBOT_EMBEDDING_MODEL", embedding_data.get("model", "text-embedding-v4"))),
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
            model=str(_env("ECBOT_GENERATION_MODEL", generation_data.get("model", "qwen-plus"))),
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
        )
        self.knowledge_base = KnowledgeBaseConfig(
            source_dir=str(_env("ECBOT_KB_SOURCE_DIR", knowledge_base_data.get("source_dir", r"E:\DATA\外贸电商知识库"))),
            supported_extensions=_as_tuple(
                knowledge_base_data.get("supported_extensions"),
                (".md", ".txt", ".pdf"),
            ),
            auto_sync_on_startup=_as_bool(
                _env("ECBOT_KB_AUTO_SYNC", knowledge_base_data.get("auto_sync_on_startup", False)),
                False,
            ),
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
                enabled=_as_bool(gateway_data.get("enabled"), False),
                openapi_base_url=str(
                    _env(
                        "ECBOT_FEISHU_OPENAPI_BASE_URL",
                        gateway_data.get("openapi_base_url", "https://open.feishu.cn/open-apis"),
                    )
                ),
                app_id=str(_env("ECBOT_FEISHU_APP_ID", gateway_data.get("app_id", ""))),
                app_secret=str(_env("ECBOT_FEISHU_APP_SECRET", gateway_data.get("app_secret", ""))),
                receive_mode=str(
                    _env(
                        "ECBOT_FEISHU_RECEIVE_MODE",
                        gateway_data.get("receive_mode", "long_connection"),
                    )
                ).strip().lower(),
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
                webhook_port=int(gateway_data.get("webhook_port", 8000)),
                webhook_path=str(gateway_data.get("webhook_path", "/webhook/feishu")),
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

    def _load_json(self, path_str: str) -> dict[str, Any]:
        path = Path(path_str)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)

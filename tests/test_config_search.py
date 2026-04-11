from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from src.config import Config


def _write_config(payload: dict) -> Path:
    base = Path("DB") / "tmp_runtime" / f"config-test-{uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_search_config_reads_tavily_env_and_provider(monkeypatch) -> None:
    config_path = _write_config(
        {
            "search": {
                "web_search_provider": "mock",
                "web_search_max_results": 6,
            }
        },
    )
    monkeypatch.setenv("ECBOT_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("ECBOT_TAVILY_API_KEY", "env-tavily-key")
    monkeypatch.setenv("ECBOT_WEB_SEARCH_MAX_RESULTS", "12")
    monkeypatch.setenv("ECBOT_WEB_SEARCH_DEPTH", "advanced")
    monkeypatch.setenv("ECBOT_WEB_SEARCH_RETRIES", "3")
    monkeypatch.setenv("ECBOT_PHASE_A_RAG_CONFIDENCE_THRESHOLD", "0.66")
    monkeypatch.setenv("ECBOT_L1_TRIGGER_THRESHOLD", "0.72")
    monkeypatch.setenv("ECBOT_L1_TEMPLATE_ENABLED", "false")
    monkeypatch.setenv("ECBOT_L2_MAX_TOP_K", "11")
    monkeypatch.setenv("ECBOT_MERGE_WEB_TRIGGER_REQUIRES_RAG_GAP", "false")
    monkeypatch.setenv("ECBOT_MERGE_TRIGGER_ON_KB_EMPTY", "true")
    monkeypatch.setenv("ECBOT_MERGE_TRIGGER_ON_LOW_CONFIDENCE", "false")
    monkeypatch.setenv("ECBOT_MERGE_STEP_MIN_EVIDENCE", "2")
    monkeypatch.setenv("ECBOT_MERGE_EVIDENCE_RAG_TOP_K", "4")
    monkeypatch.setenv("ECBOT_MERGE_EVIDENCE_SEARCH_TOP_K", "3")
    monkeypatch.setenv("ECBOT_KB_AUTO_INIT_ON_STARTUP", "true")
    monkeypatch.setenv("ECBOT_KB_INIT_BLOCKING", "true")
    monkeypatch.setenv("ECBOT_KB_INIT_FAIL_OPEN", "false")
    monkeypatch.setenv("ECBOT_RAG_PROVIDER", "ragflow")
    monkeypatch.setenv("ECBOT_RAGFLOW_BASE_URL", "https://ragflow.local")
    monkeypatch.setenv("ECBOT_RAGFLOW_API_KEY", "ragflow-key")
    monkeypatch.setenv("ECBOT_RAGFLOW_DATASET_MAP", '{"default":"ds_123"}')
    monkeypatch.setenv("ECBOT_RAGFLOW_TIMEOUT_MS", "3200")
    monkeypatch.setenv("ECBOT_RAGFLOW_TOP_K", "7")
    monkeypatch.setenv("ECBOT_RAGFLOW_MIN_SCORE", "0.23")
    monkeypatch.setenv("ECBOT_RAGFLOW_FALLBACK_TO_LEGACY", "false")

    cfg = Config(str(config_path))

    assert cfg.search.web_search_provider == "tavily"
    assert cfg.search.web_search_tavily_api_key == "env-tavily-key"
    assert cfg.search.web_search_max_results == 12
    assert cfg.search.web_search_depth == "advanced"
    assert cfg.search.web_search_retries == 3
    assert cfg.search.phase_a_rag_confidence_threshold == 0.66
    assert cfg.search.l1_trigger_threshold == 0.72
    assert cfg.search.l1_template_enabled is False
    assert cfg.search.l2_max_top_k == 11
    assert cfg.search.merge_web_trigger_requires_rag_gap is False
    assert cfg.search.merge_trigger_on_kb_empty is True
    assert cfg.search.merge_trigger_on_low_confidence is False
    assert cfg.search.merge_step_min_evidence == 2
    assert cfg.search.merge_evidence_rag_top_k == 4
    assert cfg.search.merge_evidence_search_top_k == 3
    assert cfg.knowledge_base.auto_init_on_startup is True
    assert cfg.knowledge_base.init_blocking is True
    assert cfg.knowledge_base.init_fail_open is False
    assert cfg.search.rag_provider == "ragflow"
    assert cfg.ragflow.base_url == "https://ragflow.local"
    assert cfg.ragflow.api_key == "ragflow-key"
    assert cfg.ragflow.dataset_map == {"default": "ds_123"}
    assert cfg.ragflow.timeout_ms == 3200
    assert cfg.ragflow.top_k == 7
    assert cfg.ragflow.min_score == 0.23
    assert cfg.ragflow.fallback_to_legacy is False


def test_search_config_uses_defaults_when_values_missing(monkeypatch) -> None:
    config_path = _write_config({})
    monkeypatch.setenv("ECBOT_TAVILY_API_KEY", "")
    monkeypatch.delenv("ECBOT_WEB_SEARCH_DEPTH", raising=False)
    monkeypatch.delenv("ECBOT_WEB_SEARCH_MAX_RESULTS", raising=False)
    monkeypatch.delenv("ECBOT_WEB_SEARCH_RETRIES", raising=False)
    monkeypatch.delenv("ECBOT_PHASE_A_RAG_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("ECBOT_L1_TRIGGER_THRESHOLD", raising=False)
    monkeypatch.delenv("ECBOT_L1_TEMPLATE_ENABLED", raising=False)
    monkeypatch.delenv("ECBOT_L2_MAX_TOP_K", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_WEB_TRIGGER_REQUIRES_RAG_GAP", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_TRIGGER_ON_KB_EMPTY", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_TRIGGER_ON_LOW_CONFIDENCE", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_STEP_MIN_EVIDENCE", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_EVIDENCE_RAG_TOP_K", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_EVIDENCE_SEARCH_TOP_K", raising=False)
    monkeypatch.delenv("ECBOT_KB_AUTO_INIT_ON_STARTUP", raising=False)
    monkeypatch.delenv("ECBOT_KB_INIT_BLOCKING", raising=False)
    monkeypatch.delenv("ECBOT_KB_INIT_FAIL_OPEN", raising=False)
    monkeypatch.delenv("ECBOT_RAG_PROVIDER", raising=False)
    monkeypatch.delenv("ECBOT_RAGFLOW_BASE_URL", raising=False)
    monkeypatch.delenv("ECBOT_RAGFLOW_API_KEY", raising=False)
    monkeypatch.delenv("ECBOT_RAGFLOW_DATASET_MAP", raising=False)
    monkeypatch.delenv("ECBOT_RAGFLOW_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("ECBOT_RAGFLOW_TOP_K", raising=False)
    monkeypatch.delenv("ECBOT_RAGFLOW_MIN_SCORE", raising=False)
    monkeypatch.delenv("ECBOT_RAGFLOW_FALLBACK_TO_LEGACY", raising=False)

    cfg = Config(str(config_path))

    assert cfg.search.web_search_tavily_api_key == ""
    assert cfg.search.web_search_tavily_base_url == "https://api.tavily.com/search"
    assert cfg.search.web_search_max_results == 8
    assert cfg.search.web_search_depth == "basic"
    assert cfg.search.web_search_retries == 1
    assert cfg.search.phase_a_rag_confidence_threshold == 0.58
    assert cfg.search.l1_trigger_threshold == 0.58
    assert cfg.search.l1_template_enabled is True
    assert cfg.search.l2_max_top_k == 8
    assert cfg.search.merge_web_trigger_requires_rag_gap is True
    assert cfg.search.merge_trigger_on_kb_empty is True
    assert cfg.search.merge_trigger_on_low_confidence is True
    assert cfg.search.merge_step_min_evidence == 1
    assert cfg.search.merge_evidence_rag_top_k == 3
    assert cfg.search.merge_evidence_search_top_k == 2
    assert cfg.knowledge_base.auto_init_on_startup is False
    assert cfg.knowledge_base.init_blocking is False
    assert cfg.knowledge_base.init_fail_open is True
    assert cfg.search.rag_provider == "legacy"
    assert cfg.ragflow.base_url == ""
    assert cfg.ragflow.api_key == ""
    assert cfg.ragflow.dataset_map == {}
    assert cfg.ragflow.timeout_ms == 2500
    assert cfg.ragflow.top_k == 5
    assert cfg.ragflow.min_score == 0.1
    assert cfg.ragflow.fallback_to_legacy is True


def test_model_sync_uses_shared_env_when_specific_values_missing(monkeypatch) -> None:
    config_path = _write_config(
        {
            "embedding": {"model": "embed-from-json"},
            "generation": {"model": "gen-from-json"},
        }
    )
    monkeypatch.setenv("ECBOT_MODEL", "shared-model")
    monkeypatch.delenv("ECBOT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("ECBOT_GENERATION_MODEL", raising=False)

    cfg = Config(str(config_path))

    assert cfg.embedding.model == "shared-model"
    assert cfg.generation.model == "shared-model"


def test_model_sync_specific_env_has_higher_priority_than_shared(monkeypatch) -> None:
    config_path = _write_config({})
    monkeypatch.setenv("ECBOT_MODEL", "shared-model")
    monkeypatch.setenv("ECBOT_EMBEDDING_MODEL", "embedding-specific")
    monkeypatch.setenv("ECBOT_GENERATION_MODEL", "generation-specific")

    cfg = Config(str(config_path))

    assert cfg.embedding.model == "embedding-specific"
    assert cfg.generation.model == "generation-specific"


def test_kb_source_dir_default_is_railkb(monkeypatch) -> None:
    config_path = _write_config({})
    monkeypatch.delenv("ECBOT_KB_SOURCE_DIR", raising=False)

    cfg = Config(str(config_path))

    assert cfg.knowledge_base.source_dir == r"E:\知识库\RailKB"

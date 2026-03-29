from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any


class TraceFallbackReason(str, Enum):
    DOMAIN_OUT_OF_SCOPE = "domain_out_of_scope"
    DIRECT_FUSION_EMPTY = "direct_fusion_empty"
    INDEX_NOT_READY = "index_not_ready"
    NO_RETRIEVAL_RESULTS = "no_retrieval_results"
    PROVIDER_MISCONFIGURED = "provider_misconfigured"
    RAG_FUSION_EMPTY = "rag_fusion_empty"
    WEB_NO_RESULTS = "web_no_results"
    WEB_NOT_REQUIRED = "web_not_required"
    WEB_ROUTING_ERROR = "web_routing_error"
    WEB_ROUTING_UNAVAILABLE = "web_routing_unavailable"
    WEB_SEARCH_DISABLED = "web_search_disabled"
    WEB_SEARCH_ERROR = "web_search_error"


class GenerationFallbackReason(str, Enum):
    CLAIM_SUPPORT_BELOW_THRESHOLD = "claim_support_below_threshold"
    CITATION_COVERAGE_BELOW_THRESHOLD = "citation_coverage_below_threshold"
    FTS_BRANCH_ERROR_NO_VECTOR_BACKUP = "fts_branch_error_no_vector_backup"
    FTS_NO_HIT = "fts_no_hit"
    HYBRID_UNAVAILABLE_OR_ERROR = "hybrid_unavailable_or_error"
    NO_RETRIEVAL_RESULTS = "no_retrieval_results"
    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    SEARCH_ERROR = "search_error"
    SEARCH_GENERATION_ERROR = "search_generation_error"
    VECTOR_BRANCH_ERROR_NO_LEXICAL_BACKUP = "vector_branch_error_no_lexical_backup"


ReasonCode = str | TraceFallbackReason | GenerationFallbackReason


def reason_text(reason: ReasonCode) -> str:
    if isinstance(reason, Enum):
        return str(reason.value).strip()
    return str(reason).strip()


def merge_reason_codes(left: Iterable[ReasonCode], right: Iterable[ReasonCode]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(left) + list(right):
        code = reason_text(item)
        if not code or code in seen:
            continue
        merged.append(code)
        seen.add(code)
    return merged


def build_web_trace(
    *,
    requested: bool = False,
    source_route: str = "",
    route_mode: str = "serial",
    need_web_search: bool = False,
    phase: str = "",
    reasons: Iterable[ReasonCode] | None = None,
    route_mode_from_analysis: str = "",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "requested": bool(requested),
        "executed": False,
        "execution_skipped": False,
        "skip_reason": "",
        "source_route": str(source_route or ""),
        "route_mode": str(route_mode or "serial"),
        "fusion_strategy": "none",
        "need_web_search": bool(need_web_search),
        "phase": str(phase or ""),
        "reasons": [reason_text(item) for item in (reasons or []) if reason_text(item)],
        "route_mode_from_analysis": str(route_mode_from_analysis or ""),
        "metrics": dict(metrics or {}),
        "fallback_used": False,
    }


def normalize_web_trace(search_trace: dict[str, Any]) -> None:
    web_trace = search_trace.get("web")
    if not isinstance(web_trace, dict):
        web_trace = {}
        search_trace["web"] = web_trace

    if "fallback_used" in web_trace:
        web_trace["fallback_used"] = bool(web_trace.get("fallback_used"))
        return

    if "fallback" in web_trace:
        web_trace["fallback_used"] = bool(web_trace.get("fallback"))
        compat = search_trace.get("compat")
        if not isinstance(compat, dict):
            compat = {}
            search_trace["compat"] = compat
        if not compat.get("trace_web_fallback_legacy_read"):
            compat["trace_web_fallback_legacy_read"] = True
        return

    web_trace["fallback_used"] = False


def build_strategy_fallback_step(
    *,
    reason: ReasonCode,
    filter_reason: str = "",
) -> dict[str, str]:
    return {
        "stage": "fallback_answer",
        "reason": reason_text(reason),
        "filter_reason": str(filter_reason or "").strip(),
    }


def extract_first_strategy_reason(trace: dict[str, Any]) -> str:
    strategy_execution = trace.get("strategy_execution", []) if isinstance(trace, dict) else []
    if isinstance(strategy_execution, list) and strategy_execution:
        first = strategy_execution[0]
        if isinstance(first, dict):
            return str(first.get("reason", "")).strip()
    return ""


def build_agent_trace(
    *,
    query: str,
    search_trace: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "query": str(query or ""),
        "search": dict(search_trace or {}),
        "strategy_execution": [],
        "manifest": dict(manifest or {}),
    }


def build_orchestrator_trace(
    *,
    query: str,
    rag_trace: dict[str, Any] | None,
    analysis: dict[str, Any],
    planner: dict[str, Any],
    rag_executed: bool,
    web_trace: dict[str, Any],
    web_search_interface_ready: bool,
    final_results: list[dict[str, Any]],
) -> dict[str, Any]:
    trace = dict(rag_trace or {})
    trace["query"] = {"text": str(query or "")}
    trace["analysis"] = dict(analysis or {})
    trace["planner"] = dict(planner or {})
    trace["rag"] = {
        "executed": bool(rag_executed),
        "skip_reason": "",
    }
    trace["web"] = dict(web_trace or {})
    trace["orchestrator"] = {
        "active_architecture": "planner_orchestrator_rag_web",
        "web_search_interface_ready": bool(web_search_interface_ready),
        "web_routing_owner": "search_orchestrator",
    }
    trace["final_results"] = list(final_results or [])
    return trace


def build_debug_trace(
    *,
    query_hash: str,
    query_preview: str,
    allow_rag: bool,
    filter_reason: str,
    rag_executed: bool,
    rag_skip_reason: str,
    result_count: int,
    fallback_reason: str,
) -> dict[str, Any]:
    return {
        "query_hash": str(query_hash or "").strip(),
        "query_preview": str(query_preview or "").strip(),
        "allow_rag": bool(allow_rag),
        "filter_reason": str(filter_reason or "").strip(),
        "rag_executed": bool(rag_executed),
        "rag_skip_reason": str(rag_skip_reason or "").strip(),
        "result_count": max(0, int(result_count)),
        "fallback_reason": str(fallback_reason or "").strip(),
    }

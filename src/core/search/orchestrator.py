from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import math
import re
from typing import Any, Literal, Protocol

from src.core.search.planner import Planner, PlannerOutput
from src.core.search.query_analyzer import QueryAnalysis
from src.core.search.rag_search import SearchResult
from src.core.search.source_utils import build_grouped_citations
from src.core.search.web_search_client import WebSearchResult
from src.core.trace_builder import (
    TraceFallbackReason,
    build_orchestrator_trace,
    build_web_trace,
    merge_reason_codes,
)


@dataclass
class UnifiedSearchHit:
    source_type: Literal["kb", "web", "web_db"]
    source: str
    content: str
    score: float
    file_uuid: str = ""
    chunk_id: int = 0
    source_path: str = ""
    section_title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    hits: list[UnifiedSearchHit]
    citations: list[dict[str, Any]]
    retrieval_confidence: float
    trace_search: dict[str, Any]


class QueryAnalyzerProtocol(Protocol):
    def analyze(
        self,
        *,
        query: str,
        local_results: list[Any],
        search_trace: dict[str, Any] | None = None,
    ) -> QueryAnalysis:
        ...


class WebSearcher(Protocol):
    def search(self, query: str, *, limit: int) -> list[Any]:
        ...


class WebRouteDecisionProtocol(Protocol):
    fusion_strategy: str
    reasons: list[str]
    metrics: dict[str, Any]
    fallback: bool


class WebResultEvaluatorProtocol(Protocol):
    def evaluate(self, *, query: str, results: list[WebSearchResult]) -> Any:
        ...


class WebRouterProtocol(Protocol):
    def route(
        self,
        *,
        query: str,
        analysis: QueryAnalysis,
        evaluation: Any,
    ) -> WebRouteDecisionProtocol:
        ...


class SearchOrchestrator:
    def __init__(
        self,
        *,
        planner: Planner,
        rag_searcher: Any,
        web_searcher: WebSearcher | None,
        config: Any,
        query_analyzer: QueryAnalyzerProtocol | None = None,
        web_result_evaluator: WebResultEvaluatorProtocol | None = None,
        web_router: WebRouterProtocol | None = None,
        answer_top_k: int = 3,
    ) -> None:
        self.planner = planner
        self.rag_searcher = rag_searcher
        self.web_searcher = web_searcher
        self.config = config
        self.query_analyzer = query_analyzer
        self.web_result_evaluator = web_result_evaluator
        self.web_router = web_router
        self.answer_top_k = max(1, int(answer_top_k))

    def search_with_trace(self, query: str) -> OrchestratorResult:
        self.planner.plan(
            query,
            trace_context={"query_analysis": {"need_web_search": False, "reason_codes": ["bootstrap"]}},
        )

        rag_hits, rag_trace = self.rag_searcher.search_with_trace(query)
        local_hits = [self._to_unified_hit(item) for item in rag_hits]
        rag_executed = True

        query_analysis = self._analyze_query(
            query=query,
            local_hits=local_hits,
            rag_trace=rag_trace,
        )
        self._apply_phase_a_serial_signals(
            query_analysis=query_analysis,
            local_hits=local_hits,
        )

        planner_output = self.planner.plan(
            query,
            trace_context={"query_analysis": query_analysis.to_dict()},
        )

        merged_hits, web_trace = self._apply_web_routing(
            query=query,
            local_hits=local_hits,
            planner_output=planner_output,
            query_analysis=query_analysis,
        )

        citations = build_grouped_citations(merged_hits)
        retrieval_confidence = self._compute_confidence(merged_hits)
        trace_search = self._build_trace(
            query=query,
            planner_output=planner_output,
            query_analysis=query_analysis,
            rag_trace=rag_trace,
            hits=merged_hits,
            rag_executed=rag_executed,
            web_trace=web_trace,
        )
        return OrchestratorResult(
            hits=merged_hits,
            citations=citations,
            retrieval_confidence=retrieval_confidence,
            trace_search=trace_search,
        )

    def _analyze_query(
        self,
        *,
        query: str,
        local_hits: list[UnifiedSearchHit],
        rag_trace: dict[str, Any],
    ) -> QueryAnalysis:
        fallback = QueryAnalysis(
            temporal_intent_score=0.0,
            domain_relevance_score=0.0,
            oov_entity_score=0.0,
            kb_coverage_score=1.0 if local_hits else 0.0,
            need_web_search=False,
            reasons=[],
            route_mode="kb_only",
            query_intent={},
        )
        if self.query_analyzer is None:
            fallback.reasons.append("query_analyzer_unavailable")
            return fallback
        try:
            return self.query_analyzer.analyze(
                query=query,
                local_results=local_hits,
                search_trace=rag_trace,
            )
        except Exception:
            fallback.reasons.append("query_analyzer_error")
            return fallback

    def _apply_web_routing(
        self,
        *,
        query: str,
        local_hits: list[UnifiedSearchHit],
        planner_output: PlannerOutput,
        query_analysis: QueryAnalysis,
    ) -> tuple[list[UnifiedSearchHit], dict[str, Any]]:
        phase_a_threshold = self._phase_a_rag_confidence_threshold()
        kb_confidence = self._phase_a_kb_confidence(local_hits)
        need_web_search = bool(planner_output.need_web_search)
        reasons = [str(reason) for reason in query_analysis.reasons if str(reason).strip()]
        route_mode = str(query_analysis.route_mode or "kb_only")
        if not local_hits:
            need_web_search = True
            reasons = self._merge_reasons(reasons, ["kb_empty_triggered_web_fallback"])
            if route_mode == "kb_only":
                route_mode = "web_dominant"
        elif kb_confidence < phase_a_threshold:
            need_web_search = True
            reasons = self._merge_reasons(reasons, ["phase_a_low_confidence_trigger"])
            if route_mode == "kb_only":
                route_mode = "hybrid"

        web_trace = build_web_trace(
            requested=need_web_search,
            source_route=planner_output.source_route,
            route_mode=str(getattr(planner_output, "route_mode", "serial") or "serial"),
            need_web_search=need_web_search,
            phase="A",
            reasons=list(reasons),
            route_mode_from_analysis=route_mode,
            metrics={
                "temporal_intent_score": float(query_analysis.temporal_intent_score),
                "domain_relevance_score": float(query_analysis.domain_relevance_score),
                "oov_entity_score": float(query_analysis.oov_entity_score),
                "kb_coverage_score": float(query_analysis.kb_coverage_score),
                "kb_confidence_score": kb_confidence,
                "phase_a_rag_confidence_threshold": phase_a_threshold,
                "kb_result_count": len(local_hits),
            },
        )

        if not self._web_routing_ready():
            web_trace["execution_skipped"] = True
            web_trace["skip_reason"] = TraceFallbackReason.WEB_ROUTING_UNAVAILABLE.value
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.WEB_ROUTING_UNAVAILABLE.value]
            )
            return local_hits, web_trace

        if not bool(getattr(self.config.search, "web_search_enabled", False)):
            web_trace["execution_skipped"] = True
            web_trace["skip_reason"] = TraceFallbackReason.WEB_SEARCH_DISABLED.value
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.WEB_SEARCH_DISABLED.value]
            )
            return local_hits, web_trace

        if not need_web_search:
            web_trace["execution_skipped"] = True
            web_trace["skip_reason"] = TraceFallbackReason.WEB_NOT_REQUIRED.value
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.WEB_NOT_REQUIRED.value]
            )
            return local_hits, web_trace

        try:
            web_results = self._search_web(
                query=query,
                limit=max(int(self.config.search.web_rag_max_docs), self.answer_top_k),
            )
            web_trace["executed"] = True
        except Exception as exc:
            error_text = str(exc)
            error_reasons = [TraceFallbackReason.WEB_SEARCH_ERROR.value]
            if TraceFallbackReason.PROVIDER_MISCONFIGURED.value in error_text:
                error_reasons.append(TraceFallbackReason.PROVIDER_MISCONFIGURED.value)
            web_trace["fallback_used"] = True
            web_trace["reasons"] = self._merge_reasons(web_trace["reasons"], error_reasons)
            web_trace["error"] = error_text
            return local_hits, web_trace

        web_trace["metrics"]["provider"] = str(getattr(self.config.search, "web_search_provider", "")).strip()
        web_trace["metrics"]["result_count_raw"] = len(web_results)
        if not web_results:
            web_trace["fallback_used"] = True
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.WEB_NO_RESULTS.value]
            )
            return local_hits, web_trace

        evaluator = self.web_result_evaluator
        router = self.web_router
        if evaluator is None or router is None:
            web_trace["execution_skipped"] = True
            web_trace["skip_reason"] = TraceFallbackReason.WEB_ROUTING_UNAVAILABLE.value
            web_trace["fallback_used"] = True
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.WEB_ROUTING_UNAVAILABLE.value]
            )
            return local_hits, web_trace

        evaluation = evaluator.evaluate(query=query, results=web_results)
        decision = router.route(
            query=query,
            analysis=query_analysis,
            evaluation=evaluation,
        )
        web_trace["fusion_strategy"] = decision.fusion_strategy
        web_trace["reasons"] = self._merge_reasons(web_trace["reasons"], list(decision.reasons))
        web_trace["metrics"].update(decision.metrics)
        web_trace["fallback_used"] = bool(decision.fallback)

        if decision.fusion_strategy == "direct_fusion":
            fused, fusion_detail = self._build_direct_fusion_hits(
                query=query,
                local_hits=local_hits,
                web_results=web_results,
                query_analysis=query_analysis,
            )
            if fusion_detail:
                web_trace["dynamic_fusion"] = fusion_detail
            if fused:
                return fused, web_trace
            web_trace["fallback_used"] = True
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.DIRECT_FUSION_EMPTY.value]
            )
            return local_hits, web_trace

        if decision.fusion_strategy == "rag_fusion":
            fused, fusion_detail = self._build_rag_fusion_hits(
                query=query,
                local_hits=local_hits,
                web_results=web_results,
                query_analysis=query_analysis,
            )
            if fusion_detail:
                web_trace["dynamic_fusion"] = fusion_detail
            if fused:
                return fused, web_trace
            web_trace["fallback_used"] = True
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.RAG_FUSION_EMPTY.value]
            )
            return local_hits, web_trace

        web_trace["fallback_used"] = True
        return local_hits, web_trace

    def _web_routing_ready(self) -> bool:
        return (
            self.web_searcher is not None
            and self.web_result_evaluator is not None
            and self.web_router is not None
        )

    def _search_web(self, *, query: str, limit: int) -> list[WebSearchResult]:
        searcher = self.web_searcher
        if searcher is None:
            return []

        rows: list[Any]
        if hasattr(searcher, "search"):
            rows = list(searcher.search(query, limit=limit))
        elif hasattr(searcher, "search_with_trace"):
            payload, _trace = searcher.search_with_trace(query, top_k=limit)
            rows = list(payload)
        else:
            return []

        coerced: list[WebSearchResult] = []
        for row in rows:
            if isinstance(row, WebSearchResult):
                coerced.append(row)
                continue
            if isinstance(row, dict):
                coerced.append(WebSearchResult.from_payload(row))
                continue
            coerced.append(
                WebSearchResult(
                    title=str(getattr(row, "title", "")).strip(),
                    url=str(getattr(row, "url", "")).strip(),
                    snippet=str(getattr(row, "snippet", "")).strip(),
                    score=float(getattr(row, "score", 0.0) or 0.0),
                    source_domain=str(getattr(row, "source_domain", "")).strip(),
                    published_at=str(getattr(row, "published_at", "")).strip(),
                )
            )
        return coerced

    def _build_direct_fusion_hits(
        self,
        *,
        query: str,
        local_hits: list[UnifiedSearchHit],
        web_results: list[WebSearchResult],
        query_analysis: QueryAnalysis,
    ) -> tuple[list[UnifiedSearchHit], dict[str, Any]]:
        limit = max(self.answer_top_k * 3, self.answer_top_k)
        web_limit = min(int(self.config.search.web_rag_max_docs), 8)
        web_rows = self._convert_web_results(query=query, web_results=web_results[:web_limit])
        return self._dynamic_fuse_hits(
            query=query,
            local_hits=local_hits,
            web_hits=web_rows,
            limit=limit,
            query_analysis=query_analysis,
            fusion_label="direct_fusion",
        )

    def _build_rag_fusion_hits(
        self,
        *,
        query: str,
        local_hits: list[UnifiedSearchHit],
        web_results: list[WebSearchResult],
        query_analysis: QueryAnalysis,
    ) -> tuple[list[UnifiedSearchHit], dict[str, Any]]:
        web_rows = self._convert_web_results(
            query=query,
            web_results=web_results[: int(self.config.search.web_rag_max_docs)],
        )
        return self._dynamic_fuse_hits(
            query=query,
            local_hits=local_hits,
            web_hits=web_rows,
            limit=max(self.answer_top_k * 4, int(self.config.search.web_rag_max_docs)),
            query_analysis=query_analysis,
            fusion_label="rag_fusion",
        )

    def _dynamic_fuse_hits(
        self,
        *,
        query: str,
        local_hits: list[UnifiedSearchHit],
        web_hits: list[UnifiedSearchHit],
        limit: int,
        query_analysis: QueryAnalysis,
        fusion_label: str,
    ) -> tuple[list[UnifiedSearchHit], dict[str, Any]]:
        query_terms = set(self._query_terms(query))
        temporal_query = float(query_analysis.temporal_intent_score) >= 0.6
        alpha, alpha_components = self._dynamic_alpha(query_analysis=query_analysis)

        candidates: list[dict[str, Any]] = []
        for row in list(local_hits) + list(web_hits):
            breakdown = self._hit_score_breakdown(
                hit=row,
                query_terms=query_terms,
                temporal_query=temporal_query,
            )
            candidates.append(
                {
                    "hit": row,
                    "breakdown": breakdown,
                    "stance": self._conflict_stance(str(row.content).lower()),
                }
            )

        has_restrict = any(item["stance"] < 0 for item in candidates)
        has_relax = any(item["stance"] > 0 for item in candidates)
        has_conflict = has_restrict and has_relax
        accepted: list[UnifiedSearchHit] = []
        rows_for_trace: list[dict[str, Any]] = []
        eliminated: list[dict[str, Any]] = []
        conflict_pool: list[dict[str, Any]] = []

        min_evidence = 0.26
        min_freshness_temporal = 0.32
        conflict_threshold = 0.84
        for item in candidates:
            hit = item["hit"]
            breakdown = dict(item["breakdown"])
            stance = int(item["stance"])
            if has_conflict and stance != 0:
                breakdown["conflict_risk"] = max(float(breakdown["conflict_risk"]), 0.84)

            reason = ""
            if float(breakdown["evidence_score"]) < min_evidence:
                reason = "evidence_below_threshold"
            elif temporal_query and float(breakdown["freshness_score"]) < min_freshness_temporal:
                reason = "freshness_below_threshold_for_temporal_query"
            elif float(breakdown["conflict_risk"]) >= conflict_threshold:
                reason = "high_conflict_risk_pool"

            trace_row = {
                "source_type": hit.source_type,
                "source": hit.source,
                "source_path": hit.source_path,
                "score_kb": round(float(breakdown["score_kb"]), 6),
                "score_web": round(float(breakdown["score_web"]), 6),
                "relevance_score": round(float(breakdown["relevance_score"]), 6),
                "evidence_score": round(float(breakdown["evidence_score"]), 6),
                "freshness_score": round(float(breakdown["freshness_score"]), 6),
                "authority_score": round(float(breakdown["authority_score"]), 6),
                "conflict_risk": round(float(breakdown["conflict_risk"]), 6),
            }
            if reason:
                trace_row["eliminated_reason"] = reason
                if reason == "high_conflict_risk_pool":
                    conflict_pool.append(trace_row)
                else:
                    eliminated.append(trace_row)
                rows_for_trace.append(trace_row)
                continue

            final_score = (1.0 - alpha) * float(breakdown["score_kb"]) + alpha * float(
                breakdown["score_web"]
            )
            trace_row["final_score"] = round(final_score, 6)
            rows_for_trace.append(trace_row)

            merged_meta = dict(hit.meta)
            grading = dict(merged_meta.get("grading", {}))
            grading.update(
                {
                    "relevance_score": round(float(breakdown["relevance_score"]), 6),
                    "evidence_score": round(float(breakdown["evidence_score"]), 6),
                    "freshness_score": round(float(breakdown["freshness_score"]), 6),
                    "authority_score": round(float(breakdown["authority_score"]), 6),
                    "conflict_risk": round(float(breakdown["conflict_risk"]), 6),
                    "score_kb": round(float(breakdown["score_kb"]), 6),
                    "score_web": round(float(breakdown["score_web"]), 6),
                    "final_score": round(final_score, 6),
                    "fusion_label": fusion_label,
                }
            )
            merged_meta["grading"] = grading
            merged_meta["dynamic_fusion"] = {
                "alpha": round(alpha, 6),
                "route_mode": str(query_analysis.route_mode or "kb_only"),
            }

            accepted.append(
                UnifiedSearchHit(
                    source_type=hit.source_type,
                    source=hit.source,
                    content=hit.content,
                    score=float(final_score),
                    file_uuid=hit.file_uuid,
                    chunk_id=hit.chunk_id,
                    source_path=hit.source_path,
                    section_title=hit.section_title,
                    meta=merged_meta,
                )
            )

        accepted.sort(key=lambda row: float(row.score), reverse=True)
        final_hits = self._dedupe_hits(accepted, limit=limit)
        detail = {
            "alpha": round(alpha, 6),
            "alpha_components": alpha_components,
            "route_mode": str(query_analysis.route_mode or "kb_only"),
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "returned_count": len(final_hits),
            "eliminated": eliminated[:24],
            "conflict_pool": conflict_pool[:24],
            "rows": rows_for_trace[:36],
        }
        return final_hits, detail

    def _convert_web_results(
        self,
        *,
        query: str,
        web_results: list[WebSearchResult],
    ) -> list[UnifiedSearchHit]:
        query_terms = set(self._query_terms(query))
        converted: list[UnifiedSearchHit] = []
        for index, row in enumerate(web_results, start=1):
            text = f"{row.title} {row.snippet}".strip().lower()
            if query_terms:
                overlap = sum(1 for token in query_terms if token in text) / len(query_terms)
            else:
                overlap = 0.0
            relevance_score = self._clamp(max(float(row.score), 0.42 + overlap * 0.5))
            source = row.title.strip() or row.source_domain or f"web_result_{index}"
            content = row.snippet.strip() or row.title.strip() or row.url.strip()
            if row.url:
                content = f"{content}\nURL: {row.url}"
            evidence_score = self._fallback_evidence_score(content=content.lower(), overlap=overlap)
            provisional_hit = UnifiedSearchHit(
                source_type="web",
                source=source,
                content=content,
                score=relevance_score,
                file_uuid=f"web-{index}",
                chunk_id=index,
                source_path=row.url,
                section_title=row.source_domain,
                meta={},
            )
            freshness_score = self._fallback_freshness_score(
                hit=UnifiedSearchHit(
                    source_type="web",
                    source=source,
                    content=content,
                    score=relevance_score,
                    file_uuid=f"web-{index}",
                    chunk_id=index,
                    source_path=row.url,
                    section_title=row.source_domain,
                    meta={"published_at": row.published_at},
                ),
                temporal_query=True,
            )
            authority_score = self._fallback_authority_score(hit=provisional_hit)
            conflict_risk = 0.14 if self._conflict_stance(content.lower()) == 0 else 0.4
            score = self._clamp(
                0.35 * relevance_score
                + 0.25 * evidence_score
                + 0.2 * freshness_score
                + 0.15 * authority_score
                + 0.05 * (1.0 - conflict_risk)
            )
            converted.append(
                UnifiedSearchHit(
                    source_type="web",
                    source=source,
                    content=content,
                    score=score,
                    file_uuid=f"web-{index}",
                    chunk_id=index,
                    source_path=row.url,
                    section_title=row.source_domain,
                    meta={
                        "retrieval_paths": [{"source": "web", "rank": index, "score": score}],
                        "grading": {
                            "web_score": score,
                            "relevance_score": relevance_score,
                            "evidence_score": evidence_score,
                            "freshness_score": freshness_score,
                            "authority_score": authority_score,
                            "conflict_risk": conflict_risk,
                        },
                        "published_at": row.published_at,
                    },
                )
            )
        return converted

    def _hit_score_breakdown(
        self,
        *,
        hit: UnifiedSearchHit,
        query_terms: set[str],
        temporal_query: bool,
    ) -> dict[str, float]:
        content = str(hit.content or "").lower()
        if query_terms:
            overlap = sum(1 for token in query_terms if token in content) / len(query_terms)
        else:
            overlap = 0.0

        meta_grading = {}
        if isinstance(hit.meta, dict):
            raw_grading = hit.meta.get("grading", {})
            if isinstance(raw_grading, dict):
                meta_grading = raw_grading
        relevance_score = self._safe_float(meta_grading.get("relevance_score"), -1.0)
        evidence_score = self._safe_float(meta_grading.get("evidence_score"), -1.0)
        freshness_score = self._safe_float(meta_grading.get("freshness_score"), -1.0)
        authority_score = self._safe_float(meta_grading.get("authority_score"), -1.0)
        conflict_risk = self._safe_float(meta_grading.get("conflict_risk"), -1.0)

        if relevance_score < 0:
            relevance_score = self._clamp(0.58 * overlap + 0.42 * max(0.0, float(hit.score)))
        if evidence_score < 0:
            evidence_score = self._fallback_evidence_score(content=content, overlap=overlap)
        if freshness_score < 0:
            freshness_score = self._fallback_freshness_score(hit=hit, temporal_query=temporal_query)
        if authority_score < 0:
            authority_score = self._fallback_authority_score(hit=hit)
        if conflict_risk < 0:
            conflict_risk = 0.16

        if hit.source_type == "web":
            score_web = self._clamp(
                0.36 * relevance_score
                + 0.26 * evidence_score
                + 0.2 * freshness_score
                + 0.14 * authority_score
                + 0.04 * (1.0 - conflict_risk)
            )
            score_kb = self._clamp(0.25 * score_web + 0.15 * overlap)
        else:
            base_kb = self._clamp(float(hit.score))
            quality = self._clamp(
                0.34 * relevance_score
                + 0.34 * evidence_score
                + 0.16 * freshness_score
                + 0.16 * authority_score
                - 0.1 * conflict_risk
            )
            score_kb = self._clamp(0.64 * base_kb + 0.36 * quality)
            score_web = self._clamp(0.2 * base_kb + 0.8 * quality)

        return {
            "relevance_score": relevance_score,
            "evidence_score": evidence_score,
            "freshness_score": freshness_score,
            "authority_score": authority_score,
            "conflict_risk": self._clamp(conflict_risk),
            "score_kb": score_kb,
            "score_web": score_web,
        }

    def _dynamic_alpha(self, *, query_analysis: QueryAnalysis) -> tuple[float, dict[str, Any]]:
        temporal = self._clamp(float(query_analysis.temporal_intent_score))
        oov = self._clamp(float(query_analysis.oov_entity_score))
        kb_gap = self._clamp(1.0 - float(query_analysis.kb_coverage_score))
        route_mode = str(query_analysis.route_mode or "kb_only")

        linear = 1.45 * temporal + 1.15 * oov + 1.2 * kb_gap - 1.3
        alpha = self._sigmoid(linear)
        if route_mode == "kb_only":
            alpha = min(alpha, 0.32)
        elif route_mode == "web_dominant":
            alpha = max(alpha, 0.68)
        else:
            alpha = self._clamp(alpha, 0.35, 0.75)

        reasons = {str(reason) for reason in query_analysis.reasons}
        if {"temporal_intent_high", "policy_temporal_trigger"} & reasons:
            alpha = self._clamp(alpha + 0.06)
        if "kb_coverage_low" in reasons:
            alpha = self._clamp(alpha + 0.05)

        components = {
            "temporal": round(temporal, 6),
            "oov": round(oov, 6),
            "kb_gap": round(kb_gap, 6),
            "linear": round(linear, 6),
            "route_mode": route_mode,
            "reasons": list(query_analysis.reasons),
        }
        return alpha, components

    def _fallback_evidence_score(self, *, content: str, overlap: float) -> float:
        text = str(content or "").strip()
        if not text:
            return 0.0
        length_score = min(1.0, len(text) / 220.0)
        actionable_markers = ("需要", "建议", "必须", "步骤", "should", "must", "required")
        actionable_hits = sum(1 for marker in actionable_markers if marker in text)
        actionable_score = min(1.0, actionable_hits / 3.0)
        token_pool = re.findall(r"[a-z0-9\u4e00-\u9fff]", text)
        unique_density = len(set(token_pool)) / max(len(token_pool), 1)
        return self._clamp(
            0.35 * length_score + 0.2 * actionable_score + 0.25 * unique_density + 0.2 * overlap
        )

    def _fallback_freshness_score(self, *, hit: UnifiedSearchHit, temporal_query: bool) -> float:
        candidates = [
            str(hit.meta.get("published_at", "")).strip() if isinstance(hit.meta, dict) else "",
            str(hit.source_path).strip(),
            str(hit.source).strip(),
            str(hit.section_title).strip(),
        ]
        best: float | None = None
        for value in candidates:
            if not value:
                continue
            parsed = self._extract_date(value)
            if parsed is None:
                continue
            age_days = (datetime.now(timezone.utc).date() - parsed).days
            if age_days <= 30:
                score = 1.0
            elif age_days <= 90:
                score = 0.82
            elif age_days <= 180:
                score = 0.66
            elif age_days <= 365:
                score = 0.48
            else:
                score = 0.28
            best = score if best is None else max(best, score)
        if best is None:
            return 0.35 if temporal_query and hit.source_type == "web" else 0.55
        return self._clamp(best)

    def _fallback_authority_score(self, *, hit: UnifiedSearchHit) -> float:
        source_text = " ".join([str(hit.source).lower(), str(hit.source_path).lower(), str(hit.section_title).lower()])
        if any(marker in source_text for marker in (".gov", ".edu", ".org", "docs.", "official")):
            return 0.88
        if any(marker in source_text for marker in ("forum", "bbs", "weibo", "zhihu", "reddit", "blog")):
            return 0.45
        if str(hit.source_path).startswith(("http://", "https://")):
            return 0.72
        if str(hit.source_path).strip():
            return 0.68
        return 0.58

    def _extract_date(self, text: str) -> date | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        date_patterns = re.findall(
            r"(20\d{2})[-/年](0?[1-9]|1[0-2])[-/月](0?[1-9]|[12]\d|3[01])",
            raw,
        )
        for year, month, day in date_patterns:
            try:
                return date(int(year), int(month), int(day))
            except Exception:
                continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except Exception:
            pass
        if len(raw) >= 10:
            try:
                return date.fromisoformat(raw[:10].replace("/", "-"))
            except Exception:
                return None
        return None

    def _conflict_stance(self, lowered_content: str) -> int:
        restrict_markers = ("禁止", "限制", "下架", "封禁", "ban", "restriction", "penalty")
        relax_markers = ("允许", "放宽", "恢复", "支持", "allow", "approved")
        has_restrict = any(marker in lowered_content for marker in restrict_markers)
        has_relax = any(marker in lowered_content for marker in relax_markers)
        if has_restrict and not has_relax:
            return -1
        if has_relax and not has_restrict:
            return 1
        return 0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-float(value)))
        except OverflowError:
            return 0.0 if value < 0 else 1.0

    def _dedupe_hits(self, rows: list[UnifiedSearchHit], *, limit: int) -> list[UnifiedSearchHit]:
        deduped: list[UnifiedSearchHit] = []
        seen: set[str] = set()
        for row in rows:
            source = str(row.source).strip().lower()
            source_path = str(row.source_path).strip().lower()
            content = str(row.content).strip().lower()
            key = source_path or f"{source}|{content[:120]}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= limit:
                break
        return deduped

    def _build_trace(
        self,
        *,
        query: str,
        planner_output: PlannerOutput,
        query_analysis: QueryAnalysis,
        rag_trace: dict[str, Any],
        hits: list[UnifiedSearchHit],
        rag_executed: bool,
        web_trace: dict[str, Any],
    ) -> dict[str, Any]:
        planner_trace = {
            "plan_id": planner_output.plan_id,
            "need_web_search": planner_output.need_web_search,
            "source_route": planner_output.source_route,
            "route_mode": planner_output.route_mode,
            "fusion_strategy": planner_output.fusion_strategy,
            "domain_relevance_score": planner_output.domain_relevance_score,
            "confidence": planner_output.confidence,
            "reasons": planner_output.reasons,
            "query_expansion": planner_output.query_expansion,
            "retrieval_plan": planner_output.retrieval_plan,
        }
        return build_orchestrator_trace(
            query=query,
            rag_trace=rag_trace,
            analysis=query_analysis.to_dict(),
            planner=planner_trace,
            rag_executed=rag_executed,
            web_trace=web_trace,
            web_search_interface_ready=self.web_searcher is not None,
            final_results=[self._hit_to_trace_row(item) for item in hits],
        )

    def _apply_phase_a_serial_signals(
        self,
        *,
        query_analysis: QueryAnalysis,
        local_hits: list[UnifiedSearchHit],
    ) -> None:
        if not local_hits:
            query_analysis.need_web_search = True
            query_analysis.route_mode = "web_dominant"
            query_analysis.reasons = self._merge_reasons(
                list(query_analysis.reasons),
                ["kb_empty_triggered_web_fallback"],
            )
            return

        kb_confidence = self._phase_a_kb_confidence(local_hits)
        if kb_confidence < self._phase_a_rag_confidence_threshold():
            query_analysis.need_web_search = True
            if query_analysis.route_mode == "kb_only":
                query_analysis.route_mode = "hybrid"
            query_analysis.reasons = self._merge_reasons(
                list(query_analysis.reasons),
                ["phase_a_low_confidence_trigger"],
            )

    @staticmethod
    def _to_unified_hit(item: SearchResult) -> UnifiedSearchHit:
        return UnifiedSearchHit(
            source_type="kb",
            source=str(item.source),
            content=str(item.content),
            score=float(item.score),
            file_uuid=str(item.file_uuid),
            chunk_id=int(item.chunk_id),
            source_path=str(item.source_path),
            section_title=str(item.section_title),
            meta={
                "matched_terms": list(item.matched_terms),
                "retrieval_paths": list(item.retrieval_paths),
                "grading": dict(item.grading),
            },
        )

    @staticmethod
    def _hit_to_trace_row(item: UnifiedSearchHit) -> dict[str, Any]:
        return {
            "source_type": item.source_type,
            "file_uuid": item.file_uuid,
            "chunk_id": item.chunk_id,
            "source": item.source,
            "source_path": item.source_path,
            "section_title": item.section_title,
            "content": item.content,
            "score": item.score,
            "meta": item.meta,
        }

    @staticmethod
    def _compute_confidence(hits: list[UnifiedSearchHit]) -> float:
        if not hits:
            return 0.0
        ordered = sorted(hits, key=lambda row: row.score, reverse=True)
        selected = ordered[:6]
        score_sum = sum(max(0.0, row.score) for row in selected)
        return min(1.0, score_sum / max(len(selected), 1))

    def _phase_a_rag_confidence_threshold(self) -> float:
        default_threshold = 0.58
        search_cfg = getattr(self.config, "search", None)
        raw_value = getattr(search_cfg, "phase_a_rag_confidence_threshold", default_threshold)
        try:
            return max(0.0, min(1.0, float(raw_value)))
        except Exception:
            return default_threshold

    def _phase_a_kb_confidence(self, local_hits: list[UnifiedSearchHit]) -> float:
        if not local_hits:
            return 0.0
        top_hits = sorted(local_hits, key=lambda row: row.score, reverse=True)[:3]
        return self._compute_confidence(top_hits)

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        import re

        terms = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", str(query or "").lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped

    @staticmethod
    def _merge_reasons(left: list[str], right: list[str]) -> list[str]:
        return merge_reason_codes(left, right)

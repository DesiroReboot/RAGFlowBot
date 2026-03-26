from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from src.core.search.planner import Planner, PlannerOutput
from src.core.search.rag_search import SearchResult
from src.core.search.source_utils import build_grouped_citations


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


class WebSearcher(Protocol):
    def search_with_trace(
        self,
        query: str,
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ...


class SearchOrchestrator:
    def __init__(
        self,
        *,
        planner: Planner,
        rag_searcher: Any,
        web_searcher: WebSearcher | None,
        config: Any,
    ) -> None:
        self.planner = planner
        self.rag_searcher = rag_searcher
        self.web_searcher = web_searcher
        self.config = config

    def search_with_trace(self, query: str) -> OrchestratorResult:
        planner_output = self.planner.plan(query)

        rag_executed = bool(planner_output.allow_rag)
        rag_trace: dict[str, Any] = {}
        if rag_executed:
            rag_hits, rag_trace = self.rag_searcher.search_with_trace(query)
            hits = [self._to_unified_hit(item) for item in rag_hits]
            citations = build_grouped_citations(hits)
            retrieval_confidence = self._compute_confidence(hits)
        else:
            hits = []
            citations = []
            retrieval_confidence = 0.0

        trace_search = self._build_trace(
            query=query,
            planner_output=planner_output,
            rag_trace=rag_trace,
            hits=hits,
            rag_executed=rag_executed,
        )
        return OrchestratorResult(
            hits=hits,
            citations=citations,
            retrieval_confidence=retrieval_confidence,
            trace_search=trace_search,
        )

    def _build_trace(
        self,
        *,
        query: str,
        planner_output: PlannerOutput,
        rag_trace: dict[str, Any],
        hits: list[UnifiedSearchHit],
        rag_executed: bool,
    ) -> dict[str, Any]:
        trace = dict(rag_trace or {})
        trace["query"] = {"text": query}
        trace["planner"] = {
            "plan_id": planner_output.plan_id,
            "need_web_search": planner_output.need_web_search,
            "source_route": planner_output.source_route,
            "fusion_strategy": planner_output.fusion_strategy,
            "allow_rag": planner_output.allow_rag,
            "filter_reason": planner_output.filter_reason,
            "domain_relevance_score": planner_output.domain_relevance_score,
            "domain_filter": dict(planner_output.domain_filter),
            "confidence": planner_output.confidence,
            "reasons": planner_output.reasons,
            "query_expansion": planner_output.query_expansion,
            "retrieval_plan": planner_output.retrieval_plan,
        }
        trace["rag"] = {
            "executed": rag_executed,
            "skip_reason": "" if rag_executed else planner_output.filter_reason,
        }
        trace["web"] = {
            "requested": planner_output.allow_rag and planner_output.need_web_search,
            "executed": False,
            "execution_skipped": True,
            "skip_reason": (
                "web_search_reserved_not_enabled"
                if planner_output.allow_rag
                else "blocked_by_domain_filter"
            ),
            "interface": "web_searcher.search_with_trace(query, top_k=...)",
            "fallback_used": False,
            "source_route": planner_output.source_route,
            "fusion_strategy": planner_output.fusion_strategy,
        }
        trace["orchestrator"] = {
            "active_architecture": "planner_orchestrator_rag",
            "web_search_interface_ready": self.web_searcher is not None,
        }
        trace["final_results"] = [self._hit_to_trace_row(item) for item in hits]
        return trace

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

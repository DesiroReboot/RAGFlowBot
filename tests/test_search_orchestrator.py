from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.search.orchestrator import SearchOrchestrator
from src.core.search.planner import PlannerOutput
from src.core.search.rag_search import SearchResult


@dataclass
class _StubPlanner:
    output: PlannerOutput

    def plan(self, query: str, *, trace_context: dict[str, Any] | None = None) -> PlannerOutput:  # noqa: ARG002
        return self.output


class _StubRAGSearcher:
    def __init__(self) -> None:
        self.called = 0

    def search_with_trace(self, query: str) -> tuple[list[SearchResult], dict[str, Any]]:  # noqa: ARG002
        self.called += 1
        return (
            [
                SearchResult(
                    file_uuid="f1",
                    source="kb-a.md",
                    content="内容A",
                    score=0.82,
                    chunk_id=1,
                    source_path="/kb/kb-a.md",
                    section_title="s1",
                ),
                SearchResult(
                    file_uuid="f2",
                    source="kb-b.md",
                    content="内容B",
                    score=0.64,
                    chunk_id=2,
                    source_path="/kb/kb-b.md",
                    section_title="s2",
                ),
            ],
            {"fts_recall": [{"source": "kb-a.md"}], "generation": {"branch_errors": {}}},
        )


class _StubWebSearcher:
    def __init__(self) -> None:
        self.called = 0

    def search_with_trace(
        self,
        query: str,
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # noqa: ARG002
        self.called += 1
        return ([], {"ok": True})


def test_orchestrator_uses_planner_fields_and_skips_web_execution() -> None:
    rag_searcher = _StubRAGSearcher()
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-1",
            need_web_search=True,
            source_route="hybrid",
            fusion_strategy="rag_fusion",
            allow_rag=True,
            filter_reason="score_above_threshold",
            domain_relevance_score=0.87,
            domain_filter={"decision": "allow", "score": 0.87},
            reasons=["temporal_intent_detected"],
            retrieval_plan={"sources": [{"name": "kb_index", "enabled": True}]},
        )
    )
    web_searcher = _StubWebSearcher()
    orchestrator = SearchOrchestrator(
        planner=planner,
        rag_searcher=rag_searcher,
        web_searcher=web_searcher,
        config=None,
    )

    result = orchestrator.search_with_trace("最近平台政策")

    assert rag_searcher.called == 1
    assert web_searcher.called == 0
    assert len(result.hits) == 2
    assert result.trace_search["planner"]["source_route"] == "hybrid"
    assert result.trace_search["planner"]["fusion_strategy"] == "rag_fusion"
    assert result.trace_search["planner"]["allow_rag"] is True
    assert result.trace_search["planner"]["domain_filter"]["decision"] == "allow"
    assert result.trace_search["rag"]["executed"] is True
    assert result.trace_search["web"]["executed"] is False
    assert result.trace_search["web"]["skip_reason"] == "web_search_reserved_not_enabled"


def test_orchestrator_skips_rag_when_blocked_by_domain_filter() -> None:
    rag_searcher = _StubRAGSearcher()
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-blocked",
            need_web_search=False,
            source_route="kb_only",
            fusion_strategy="none",
            allow_rag=False,
            filter_reason="score_below_threshold",
            domain_relevance_score=0.1,
            domain_filter={"decision": "block", "score": 0.1, "negative_hits": ["nba"]},
            reasons=["domain_filter:score_below_threshold"],
            retrieval_plan={"sources": [{"name": "kb_index", "enabled": False}]},
        )
    )
    orchestrator = SearchOrchestrator(
        planner=planner,
        rag_searcher=rag_searcher,
        web_searcher=None,
        config=None,
    )

    result = orchestrator.search_with_trace("今天NBA战况")

    assert rag_searcher.called == 0
    assert result.hits == []
    assert result.citations == []
    assert result.retrieval_confidence == 0.0
    assert result.trace_search["planner"]["allow_rag"] is False
    assert result.trace_search["planner"]["filter_reason"] == "score_below_threshold"
    assert result.trace_search["planner"]["domain_filter"]["decision"] == "block"
    assert result.trace_search["rag"]["executed"] is False
    assert result.trace_search["rag"]["skip_reason"] == "score_below_threshold"
    assert result.trace_search["web"]["requested"] is False
    assert result.trace_search["web"]["skip_reason"] == "blocked_by_domain_filter"

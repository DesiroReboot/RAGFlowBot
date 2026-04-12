from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from src.core.search.orchestrator import SearchOrchestrator
from src.core.search.planner import PlannerOutput
from src.core.search.query_analyzer import QueryAnalysis
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
                    content="content a",
                    score=0.82,
                    chunk_id=1,
                    source_path="/kb/kb-a.md",
                    section_title="s1",
                ),
                SearchResult(
                    file_uuid="f2",
                    source="kb-b.md",
                    content="content b",
                    score=0.64,
                    chunk_id=2,
                    source_path="/kb/kb-b.md",
                    section_title="s2",
                ),
            ],
            {"fts_recall": [{"source": "kb-a.md"}], "generation": {"branch_errors": {}}},
        )


class _StubRAGFlowSearcher:
    def __init__(self) -> None:
        self.called = 0

    def search_with_trace(self, query: str) -> tuple[list[SearchResult], dict[str, Any]]:  # noqa: ARG002
        self.called += 1
        return (
            [
                SearchResult(
                    file_uuid="rf1",
                    source="ragflow-doc",
                    content="ragflow content",
                    score=0.88,
                    chunk_id=10,
                    source_path="https://ragflow/doc/1",
                    section_title="chunk-10",
                )
            ],
            {"provider": "ragflow", "selected_count": 1},
        )


class _LowConfidenceRAGSearcher:
    def __init__(self) -> None:
        self.called = 0

    def search_with_trace(self, query: str) -> tuple[list[SearchResult], dict[str, Any]]:  # noqa: ARG002
        self.called += 1
        return (
            [
                SearchResult(
                    file_uuid="low-1",
                    source="kb-low-a.md",
                    content="short weak hint",
                    score=0.24,
                    chunk_id=1,
                    source_path="/kb/kb-low-a.md",
                    section_title="s1",
                ),
                SearchResult(
                    file_uuid="low-2",
                    source="kb-low-b.md",
                    content="another weak hint",
                    score=0.2,
                    chunk_id=2,
                    source_path="/kb/kb-low-b.md",
                    section_title="s2",
                ),
            ],
            {"fts_recall": [{"source": "kb-low-a.md"}], "generation": {"branch_errors": {}}},
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


class _StubAnalyzer:
    def analyze(
        self,
        *,
        query: str,  # noqa: ARG002
        local_results: list[Any],  # noqa: ARG002
        search_trace: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> QueryAnalysis:
        return QueryAnalysis(
            temporal_intent_score=0.8,
            domain_relevance_score=0.7,
            oov_entity_score=0.6,
            kb_coverage_score=0.3,
            need_web_search=True,
            reasons=["temporal_intent_high", "kb_coverage_low"],
            route_mode="web_dominant",
            query_intent={"temporal_terms": ["latest"]},
        )


class _NoWebAnalyzer:
    def analyze(
        self,
        *,
        query: str,  # noqa: ARG002
        local_results: list[Any],  # noqa: ARG002
        search_trace: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> QueryAnalysis:
        return QueryAnalysis(
            temporal_intent_score=0.0,
            domain_relevance_score=0.1,
            oov_entity_score=0.1,
            kb_coverage_score=0.55,
            need_web_search=False,
            reasons=[],
            route_mode="kb_only",
            query_intent={},
        )


class _StubWebEvaluator:
    def evaluate(self, *, query: str, results: list[Any]) -> Any:  # noqa: ARG002
        return SimpleNamespace(
            result_count=len(results),
            top1_score=0.92,
            top3_mean=0.81,
            score_gap=0.2,
            domain_diversity=0.3,
            freshness_ratio=0.8,
            noise_ratio=0.1,
            conflict_detected=False,
            to_dict=lambda: {
                "result_count": len(results),
                "top1_score": 0.92,
                "top3_mean": 0.81,
                "score_gap": 0.2,
                "domain_diversity": 0.3,
                "freshness_ratio": 0.8,
                "noise_ratio": 0.1,
                "conflict_detected": False,
            },
        )


class _StubWebRouter:
    def route(self, *, query: str, analysis: Any, evaluation: Any) -> Any:  # noqa: ARG002
        return SimpleNamespace(
            fusion_strategy="direct_fusion",
            reasons=["direct_fusion_thresholds_met"],
            metrics=evaluation.to_dict(),
            fallback=False,
        )


class _StubWebClient:
    def __init__(self) -> None:
        self.called = 0

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:  # noqa: ARG002
        self.called += 1
        return [
            {
                "title": "Platform policy weekly update",
                "url": "https://news.example.com/policy",
                "snippet": "Latest policy update with compliance reminders.",
                "score": 0.94,
                "source_domain": "news.example.com",
                "published_at": "2026-03-20",
            }
        ]


class _MemoryStoreRecorder:
    def __init__(self) -> None:
        self.decision_rows: list[dict[str, Any]] = []
        self.io_rows: list[dict[str, Any]] = []

    def safe_call(self, fn_name: str, **kwargs: Any) -> None:
        if fn_name == "append_decision_trace":
            self.decision_rows.append(dict(kwargs))
            return
        if fn_name == "append_io_snapshot":
            self.io_rows.append(dict(kwargs))
            return


def test_orchestrator_uses_planner_fields_and_skips_web_execution() -> None:
    rag_searcher = _StubRAGSearcher()
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-1",
            need_web_search=True,
            source_route="hybrid",
            fusion_strategy="rag_fusion",
            domain_relevance_score=0.87,
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

    result = orchestrator.search_with_trace("latest policy")

    assert rag_searcher.called == 1
    assert web_searcher.called == 0
    assert len(result.hits) == 2
    assert result.trace_search["planner"]["source_route"] == "hybrid"
    assert result.trace_search["planner"]["fusion_strategy"] == "rag_fusion"
    assert result.trace_search["rag"]["executed"] is True
    assert result.trace_search["web"]["executed"] is False
    assert result.trace_search["web"]["skip_reason"] == "web_routing_unavailable"


def test_orchestrator_executes_web_and_fuses_results() -> None:
    rag_searcher = _StubRAGSearcher()
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-web",
            need_web_search=True,
            source_route="hybrid",
            fusion_strategy="none",
            domain_relevance_score=0.8,
            reasons=["temporal_intent_high"],
            retrieval_plan={"sources": [{"name": "kb_index", "enabled": True}]},
        )
    )
    web_client = _StubWebClient()
    orchestrator = SearchOrchestrator(
        planner=planner,
        rag_searcher=rag_searcher,
        web_searcher=web_client,
        config=SimpleNamespace(
            search=SimpleNamespace(
                web_search_enabled=True,
                web_rag_max_docs=8,
                web_search_provider="tavily",
            )
        ),
        query_analyzer=_StubAnalyzer(),
        web_result_evaluator=_StubWebEvaluator(),
        web_router=_StubWebRouter(),
        answer_top_k=3,
    )

    result = orchestrator.search_with_trace("latest platform policy update")

    assert rag_searcher.called == 1
    assert web_client.called == 1
    assert len(result.hits) >= 1
    assert any(hit.source_type == "web" for hit in result.hits)
    assert result.trace_search["web"]["executed"] is True
    assert result.trace_search["web"]["fusion_strategy"] == "direct_fusion"
    assert result.trace_search["web"]["fallback_used"] is False
    dynamic = result.trace_search["web"]["dynamic_fusion"]
    assert dynamic["alpha"] >= 0.6
    assert dynamic["route_mode"] == "web_dominant"
    assert "temporal" in dynamic["alpha_components"]


def test_orchestrator_blocks_l2_when_l1_confidence_is_low() -> None:
    rag_searcher = _LowConfidenceRAGSearcher()
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-phase-a-low-conf",
            need_web_search=False,
            source_route="kb_only",
            fusion_strategy="none",
            domain_relevance_score=0.86,
            reasons=["manual_planner_no_web"],
            retrieval_plan={"sources": [{"name": "kb_index", "enabled": True}]},
        )
    )
    web_client = _StubWebClient()
    orchestrator = SearchOrchestrator(
        planner=planner,
        rag_searcher=rag_searcher,
        web_searcher=web_client,
        config=SimpleNamespace(
            search=SimpleNamespace(
                web_search_enabled=True,
                web_rag_max_docs=8,
                web_search_provider="tavily",
                phase_a_rag_confidence_threshold=0.58,
            )
        ),
        query_analyzer=_NoWebAnalyzer(),
        web_result_evaluator=_StubWebEvaluator(),
        web_router=_StubWebRouter(),
        answer_top_k=3,
    )

    result = orchestrator.search_with_trace("category trend")

    assert rag_searcher.called == 1
    assert web_client.called == 0
    assert result.trace_search["decision"]["trigger_full_rag"] is False
    assert result.trace_search["decision"]["reason_code"] == "LOW_RELEVANCE_GATE"
    assert result.trace_search["l2"]["executed"] is False


def test_orchestrator_exposes_l1_and_route_decision_methods() -> None:
    rag_searcher = _LowConfidenceRAGSearcher()
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-lite",
            need_web_search=False,
            source_route="kb_only",
            fusion_strategy="none",
            domain_relevance_score=0.86,
            reasons=["manual_planner_no_web"],
            retrieval_plan={"sources": [{"name": "kb_index", "enabled": True}]},
        )
    )
    orchestrator = SearchOrchestrator(
        planner=planner,
        rag_searcher=rag_searcher,
        web_searcher=None,
        config=SimpleNamespace(
            search=SimpleNamespace(
                l1_trigger_threshold=0.6,
                l2_max_top_k=4,
                phase_a_rag_confidence_threshold=0.58,
                web_search_enabled=False,
                web_rag_max_docs=4,
                web_search_provider="mock",
            )
        ),
        query_analyzer=_NoWebAnalyzer(),
        web_result_evaluator=None,
        web_router=None,
    )

    l1_result = orchestrator.run_l1_partial("category trend")
    decision = orchestrator.route_by_l1_confidence(l1_result)

    assert rag_searcher.called == 1
    assert l1_result.trace["l1"]["hit_count"] == 2
    assert decision.trigger_full_rag is False
    assert decision.reason_code == "LOW_RELEVANCE_GATE"


def test_orchestrator_accepts_ragflow_searcher_contract() -> None:
    rag_searcher = _StubRAGFlowSearcher()
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-ragflow",
            need_web_search=False,
            source_route="kb_only",
            fusion_strategy="none",
            domain_relevance_score=0.9,
            reasons=[],
            retrieval_plan={"sources": [{"name": "ragflow", "enabled": True}]},
        )
    )
    orchestrator = SearchOrchestrator(
        planner=planner,
        rag_searcher=rag_searcher,
        web_searcher=None,
        config=SimpleNamespace(
            search=SimpleNamespace(
                l1_trigger_threshold=0.2,
                l2_max_top_k=4,
                phase_a_rag_confidence_threshold=0.2,
                web_search_enabled=False,
                web_rag_max_docs=4,
                web_search_provider="mock",
            )
        ),
        query_analyzer=_NoWebAnalyzer(),
        web_result_evaluator=None,
        web_router=None,
    )

    result = orchestrator.search_with_trace("policy")
    assert rag_searcher.called == 1
    assert len(result.hits) == 1
    assert result.hits[0].source == "ragflow-doc"


def test_orchestrator_writes_memory_decision_and_retrieved_snapshots() -> None:
    rag_searcher = _StubRAGSearcher()
    planner = _StubPlanner(
        output=PlannerOutput(
            plan_id="plan-memory",
            need_web_search=True,
            source_route="hybrid",
            fusion_strategy="none",
            domain_relevance_score=0.83,
            reasons=["temporal_intent_high"],
            retrieval_plan={"sources": [{"name": "kb_index", "enabled": True}]},
        )
    )
    web_client = _StubWebClient()
    recorder = _MemoryStoreRecorder()
    orchestrator = SearchOrchestrator(
        planner=planner,
        rag_searcher=rag_searcher,
        web_searcher=web_client,
        config=SimpleNamespace(
            search=SimpleNamespace(
                web_search_enabled=True,
                web_rag_max_docs=8,
                web_search_provider="tavily",
                phase_a_rag_confidence_threshold=0.2,
                l1_trigger_threshold=0.2,
                l2_max_top_k=4,
            )
        ),
        query_analyzer=_StubAnalyzer(),
        web_result_evaluator=_StubWebEvaluator(),
        web_router=_StubWebRouter(),
        answer_top_k=3,
    )

    result = orchestrator.search_with_trace(
        "latest policy update",
        run_id="run_test_memory_001",
        memory_store=recorder,
    )

    assert len(result.hits) >= 1
    assert len(recorder.decision_rows) >= 4
    assert any(row.get("stage") == "_apply_web_routing" for row in recorder.decision_rows)
    assert any(
        row.get("io_type") == "retrieved_results" and row.get("run_id") == "run_test_memory_001"
        for row in recorder.io_rows
    )

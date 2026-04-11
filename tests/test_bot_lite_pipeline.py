from __future__ import annotations

from types import SimpleNamespace

from src.config import GenerationConfig
from src.core.bot_agent import ReActAgent
from src.core.search.rag_search import SearchResult


def _make_agent(*, template_enabled: bool, orchestrator: object) -> ReActAgent:
    agent = ReActAgent.__new__(ReActAgent)
    agent.answer_top_k = 3
    agent.retrieval_provider = "legacy"
    agent.search_orchestrator = orchestrator
    agent.config = SimpleNamespace(
        search=SimpleNamespace(l1_template_enabled=template_enabled),
        generation=GenerationConfig(mode="template"),
    )
    agent.manifest_store = SimpleNamespace(get_manifest=lambda: {"build_version": "rag-v2", "status": "ready"})
    agent.generation_client = SimpleNamespace()
    return agent


def test_run_sync_uses_lite_pipeline_template_gate() -> None:
    orchestrator = SimpleNamespace(
        run_l1_partial=lambda query: SimpleNamespace(  # noqa: ARG005
            confidence=0.2,
            trace={"l1": {"confidence": 0.2, "threshold": 0.58, "hit_count": 0}, "web": {"fallback_used": False}},
        ),
        route_by_l1_confidence=lambda result: SimpleNamespace(
            trigger_full_rag=False,
            reason_code="LOW_RELEVANCE_GATE",
            threshold=0.58,
            l1_confidence=result.confidence,
        ),
        run_l2_full=lambda query, result: (_ for _ in ()).throw(AssertionError("l2 should not run")),  # noqa: ARG005
    )
    agent = _make_agent(template_enabled=True, orchestrator=orchestrator)

    response = agent.run_sync("who wins nba today", include_trace=True)

    assert "LOW_RELEVANCE_GATE" in response.answer
    assert response.trace["strategy_execution"][0]["stage"] == "l1_gate"
    assert response.trace["strategy_execution"][0]["decision"] == "template_response"
    assert response.trace["retrieval_confidence"] == 0.2
    assert response.trace["retrieval_provider"] == "legacy"


def test_run_sync_uses_lite_pipeline_l2_when_triggered() -> None:
    hit = SearchResult(
        file_uuid="f1",
        source="kb-a.md",
        content="content a",
        score=0.82,
        chunk_id=1,
        source_path="/kb/kb-a.md",
        section_title="s1",
    )
    orchestrator = SimpleNamespace(
        run_l1_partial=lambda query: SimpleNamespace(  # noqa: ARG005
            confidence=0.8,
            trace={"l1": {"confidence": 0.8, "threshold": 0.58, "hit_count": 1}, "web": {"fallback_used": False}},
        ),
        route_by_l1_confidence=lambda result: SimpleNamespace(
            trigger_full_rag=True,
            reason_code="L2_FULL_RAG_TRIGGERED",
            threshold=0.58,
            l1_confidence=result.confidence,
        ),
        run_l2_full=lambda query, result: SimpleNamespace(  # noqa: ARG005
            hits=[hit],
            citations=[{"source": "kb-a.md", "aliases": []}],
            retrieval_confidence=0.82,
            trace={"l2": {"executed": True}, "web": {"fallback_used": False}},
        ),
    )
    agent = _make_agent(template_enabled=True, orchestrator=orchestrator)

    response = agent.run_sync("amazon policy", include_trace=True)

    assert response.answer
    assert response.retrieval_confidence >= 0.82
    assert response.trace["strategy_execution"][0]["stage"] == "l1_gate"
    assert response.trace["strategy_execution"][0]["decision"] == "trigger_full_rag"
    assert response.trace["retrieval_provider"] == "legacy"

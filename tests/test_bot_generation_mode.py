from __future__ import annotations

from types import SimpleNamespace

from src.config import GenerationConfig
from src.core.bot_agent import AnswerDraft, ReActAgent


def _build_agent(*, generation: GenerationConfig) -> ReActAgent:
    agent = ReActAgent.__new__(ReActAgent)
    agent.config = SimpleNamespace(generation=generation)
    agent.generation_client = SimpleNamespace(available=True)
    return agent


def _build_draft() -> AnswerDraft:
    return AnswerDraft(
        query="How should we optimize product selection?",
        theme="product_selection",
        steps=[
            "Validate market demand first.",
            "Evaluate competition and margin.",
            "Run a small pilot and review data.",
        ],
        evidence=["Demand should be validated before scaling investment."],
        citations=[{"source": "trade-guide.md", "aliases": []}],
    )


def test_compose_answer_fallbacks_to_template_when_fts_miss() -> None:
    generation = GenerationConfig(mode="hybrid")
    agent = _build_agent(generation=generation)
    draft = _build_draft()
    template_answer = agent._render_template_answer(draft)

    def _should_not_call(**_: object) -> str:
        raise AssertionError("hybrid rewrite should not run when fts misses")

    agent._hybrid_rewrite = _should_not_call  # type: ignore[method-assign]
    answer, meta = agent._compose_answer(
        draft=draft,
        template_answer=template_answer,
        search_trace={"fts_recall": []},
    )

    assert answer == template_answer
    assert meta["final_mode"] == "template"
    assert meta["fallback_reason"] == "fts_no_hit"


def test_compose_answer_fallbacks_when_claim_support_low() -> None:
    generation = GenerationConfig(
        mode="hybrid",
        min_quality_score=0.4,
        min_claim_support_rate=0.8,
        min_citation_coverage=0.0,
    )
    agent = _build_agent(generation=generation)
    draft = _build_draft()
    template_answer = agent._render_template_answer(draft)
    agent._hybrid_rewrite = (  # type: ignore[method-assign]
        lambda **_: (
            "Question: optimize product selection\n"
            "Execution Steps:\n"
            "1. Brand story\n"
            "2. Festival marketing\n"
            "3. Livestreaming\n"
            "Key Info:\n"
            "- all external experience\n"
            "References:\n"
            "- trade-guide.md"
        )
    )

    answer, meta = agent._compose_answer(
        draft=draft,
        template_answer=template_answer,
        search_trace={"fts_recall": [{"source": "trade-guide.md"}], "generation": {"branch_errors": {}}},
    )

    assert answer == template_answer
    assert meta["final_mode"] == "template"
    assert meta["fallback_reason"] == "claim_support_below_threshold"


def test_compose_answer_uses_hybrid_when_quality_checks_pass() -> None:
    generation = GenerationConfig(
        mode="hybrid",
        min_quality_score=0.4,
        min_claim_support_rate=0.2,
        min_citation_coverage=0.5,
    )
    agent = _build_agent(generation=generation)
    draft = _build_draft()
    template_answer = agent._render_template_answer(draft)
    hybrid_answer = (
        "Question: optimize product selection\n"
        "Execution Steps:\n"
        "1. Validate market demand first.\n"
        "2. Evaluate competition and margin.\n"
        "3. Run a small pilot and review data.\n"
        "Key Info:\n"
        "- Demand should be validated before scaling investment.\n"
        "References:\n"
        "- trade-guide.md\n"
    )
    agent._hybrid_rewrite = (  # type: ignore[method-assign]
        lambda **_: hybrid_answer
    )

    answer, meta = agent._compose_answer(
        draft=draft,
        template_answer=template_answer,
        search_trace={"fts_recall": [{"source": "trade-guide.md"}], "generation": {"branch_errors": {}}},
    )

    assert answer == hybrid_answer
    assert meta["final_mode"] == "hybrid"
    assert meta["fallback_reason"] == ""


def test_run_sync_returns_no_retrieval_fallback_when_search_empty() -> None:
    agent = ReActAgent.__new__(ReActAgent)
    agent.search_orchestrator = SimpleNamespace(
        search_with_trace=lambda query: SimpleNamespace(  # noqa: ARG005
            hits=[],
            citations=[],
            retrieval_confidence=0.0,
            trace_search={"planner": {"filter_reason": "rag_all_queries"}},
        )
    )
    agent.manifest_store = SimpleNamespace(get_manifest=lambda: {"build_version": "rag-v2"})

    response = agent.run_sync("who wins nba today", include_trace=True)

    assert response.answer
    assert response.trace["strategy_execution"][0]["reason"] == "no_retrieval_results"
    assert response.trace["strategy_execution"][0]["filter_reason"] == "rag_all_queries"

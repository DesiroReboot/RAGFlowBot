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
        query="如何优化选品策略？",
        theme="product_selection",
        steps=[
            "先做市场需求验证，确认核心需求。",
            "再评估竞争强度和利润空间。",
            "最后小批量测试并复盘。",
        ],
        evidence=["市场需求需要先验证，再评估利润空间，最后进行小批量测试。"],
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
        lambda **_: "问题：如何优化选品策略？\n建议执行步骤：\n1. 做品牌故事\n2. 做节日营销\n3. 做直播带货\n关键信息：\n- 全部使用外部经验\n参考来源：\n- trade-guide.md"
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
        "问题：如何优化选品策略？\n"
        "建议执行步骤：\n"
        "1. 先做市场需求验证，确认核心需求。\n"
        "2. 再评估竞争强度和利润空间。\n"
        "3. 最后小批量测试并复盘。\n"
        "关键信息：\n"
        "- 市场需求需要先验证，再评估利润空间，最后进行小批量测试。\n"
        "参考来源：\n"
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


def test_run_sync_returns_domain_out_of_scope_fallback_when_filter_blocks() -> None:
    agent = ReActAgent.__new__(ReActAgent)
    agent.search_orchestrator = SimpleNamespace(
        search_with_trace=lambda query: SimpleNamespace(  # noqa: ARG005
            hits=[],
            citations=[],
            retrieval_confidence=0.0,
            trace_search={"planner": {"allow_rag": False, "filter_reason": "score_below_threshold"}},
        )
    )
    agent.manifest_store = SimpleNamespace(get_manifest=lambda: {"build_version": "rag-v2"})

    response = agent.run_sync("今天NBA谁会赢", include_trace=True)

    assert "外贸/跨境电商" in response.answer
    assert response.trace["strategy_execution"][0]["reason"] == "domain_out_of_scope"
    assert response.trace["strategy_execution"][0]["filter_reason"] == "score_below_threshold"

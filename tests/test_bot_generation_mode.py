from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import GenerationConfig
from src.core.bot_agent import AnswerDraft, ReActAgent
from src.core.generation.generation_client import GenerationClient


def _build_agent(*, generation: GenerationConfig) -> ReActAgent:
    agent = ReActAgent.__new__(ReActAgent)
    agent.config = SimpleNamespace(
        generation=generation,
        search=SimpleNamespace(paragraph_output_enabled=True),
    )
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
        source_rows=["[S1] trade-guide.md | chunk#1 | reference"],
        citations=[{"source": "trade-guide.md", "aliases": []}],
        fact_units=[
            {
                "statement": "Demand should be validated before scaling investment.",
                "source": "trade-guide.md",
            }
        ],
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
            "优先讲品牌故事、节日营销和直播节奏，这些做法更容易带来短期曝光[S1]。\n\n"
            "来源：\n"
            "- [S1] trade-guide.md | chunk#1 | reference"
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
        "Demand should be validated before scaling investment[S1]. "
        "在执行层面可以先验证需求，再评估竞争与利润，最后做小规模试投并复盘数据[S1]。\n\n"
        "来源：\n"
        "- [S1] trade-guide.md | chunk#1 | reference\n"
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


def test_render_template_answer_prefers_paragraph_and_keeps_sources() -> None:
    generation = GenerationConfig(mode="template")
    agent = _build_agent(generation=generation)
    draft = _build_draft()

    answer = agent._render_template_answer(draft)

    assert "要点：" not in answer
    assert "来源：" in answer
    assert "[S1]" in answer
    assert "trade-guide.md" in answer


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


def test_build_source_rows_uses_chunk_locator_when_section_missing() -> None:
    generation = GenerationConfig(mode="template")
    agent = _build_agent(generation=generation)
    selected = [
        SimpleNamespace(
            source="02-璇㈢洏鎶ヤ环.md",
            source_path="E:/DATA/澶栬锤鐢靛晢鐭ヨ瘑搴?02-璇㈢洏鎶ヤ环.md",
            section_title="",
            chunk_id=17,
            chunk_kind="procedure",
        )
    ]
    citations = [{"source": "02-璇㈢洏鎶ヤ环.md"}]

    rows, tags = agent._build_source_rows(selected=selected, citations=citations)

    assert tags == ["S1"]
    assert rows == ["[S1] 02-璇㈢洏鎶ヤ环.md | chunk#17 | procedure"]


def test_build_source_rows_prefers_real_section_over_fallback() -> None:
    generation = GenerationConfig(mode="template")
    agent = _build_agent(generation=generation)
    selected = [
        SimpleNamespace(
            source="03-璁㈣揣绛剧害.md",
            source_path="E:/DATA/澶栬锤鐢靛晢鐭ヨ瘑搴?03-璁㈣揣绛剧害.md",
            section_title="",
            chunk_id=5,
            chunk_kind="procedure",
        ),
        SimpleNamespace(
            source="03-璁㈣揣绛剧害.md",
            source_path="E:/DATA/澶栬锤鐢靛晢鐭ヨ瘑搴?03-璁㈣揣绛剧害.md",
            section_title="section-a",
            chunk_id=6,
            chunk_kind="procedure",
        ),
    ]
    citations = [{"source": "03-璁㈣揣绛剧害.md"}]

    rows, _ = agent._build_source_rows(selected=selected, citations=citations)

    assert rows == ["[S1] 03-璁㈣揣绛剧害.md | section-a | procedure"]

def test_generation_client_rewrite_accepts_extended_kwargs() -> None:
    client = GenerationClient(GenerationConfig(api_key="", model="qwen3-32b"))

    with pytest.raises(RuntimeError, match="generation client unavailable"):
        client.rewrite(
            query="fob price composition",
            template_answer="Key points:\n- [S1] unit price\nSources:\n- [S1] test.md | chunk#1 | procedure",
            answer_mode="fact_qa",
            key_points=["unit price (FOB/CIF/CFR/EXW)"],
            steps=[],
            evidence=["unit price (FOB/CIF/CFR/EXW incoterms)"],
            citation_sources=["test.md"],
            paragraph_output=True,
        )

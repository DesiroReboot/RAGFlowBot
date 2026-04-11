from __future__ import annotations

from types import SimpleNamespace

from src.core.bot_agent import ReActAgent


def test_run_sync_blocks_early_when_manifest_empty() -> None:
    agent = ReActAgent.__new__(ReActAgent)
    agent.manifest_store = SimpleNamespace(get_manifest=lambda: {"status": "empty"})
    agent.search_orchestrator = SimpleNamespace(
        search_with_trace=lambda query: (_ for _ in ()).throw(AssertionError("search should not run"))  # noqa: ARG005
    )

    response = agent.run_sync("test query", include_trace=True)

    assert response.trace["strategy_execution"][0]["reason"] == "index_not_ready"
    assert response.trace["search"]["manifest_gate"]["blocked"] is True
    assert response.trace["search"]["manifest_gate"]["status"] == "empty"


def test_run_sync_treats_legacy_non_empty_manifest_as_ready() -> None:
    agent = ReActAgent.__new__(ReActAgent)
    agent.manifest_store = SimpleNamespace(get_manifest=lambda: {"build_version": "rag-v2"})
    agent.search_orchestrator = SimpleNamespace(
        search_with_trace=lambda query: SimpleNamespace(  # noqa: ARG005
            hits=[],
            citations=[],
            retrieval_confidence=0.0,
            trace_search={"planner": {"allow_rag": True}},
        )
    )

    response = agent.run_sync("test query", include_trace=True)

    assert response.trace["strategy_execution"][0]["reason"] == "no_retrieval_results"


def test_run_sync_skips_manifest_gate_when_ragflow_provider_enabled() -> None:
    agent = ReActAgent.__new__(ReActAgent)
    agent.rag_provider = "ragflow"
    agent.retrieval_provider = "ragflow"
    agent.manifest_store = SimpleNamespace(get_manifest=lambda: {"status": "empty"})
    agent.search_orchestrator = SimpleNamespace(
        search_with_trace=lambda query: SimpleNamespace(  # noqa: ARG005
            hits=[],
            citations=[],
            retrieval_confidence=0.0,
            trace_search={"planner": {"allow_rag": True}},
        )
    )

    response = agent.run_sync("test query", include_trace=True)

    assert response.trace["search"]["manifest_gate"]["blocked"] is False
    assert response.trace["search"]["manifest_gate"]["status"] == "ragflow_bypass"

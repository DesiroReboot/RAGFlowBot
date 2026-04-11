from __future__ import annotations

from types import SimpleNamespace

from src.core.search.rag_search import SearchResult
from src.core.search.ragflow_searcher import RAGFlowSearcher


def test_ragflow_searcher_maps_chunks_to_search_results() -> None:
    client = SimpleNamespace(
        search=lambda **kwargs: SimpleNamespace(  # noqa: ARG005
            rewritten_question="rewritten",
            chunks=[
                SimpleNamespace(
                    id="11",
                    document_id="doc-1",
                    content="chunk text",
                    score=0.92,
                    title="doc title",
                    source="/kb/doc.md",
                    metadata={},
                )
            ],
        )
    )
    searcher = RAGFlowSearcher(
        client=client,
        dataset_map={"default": "ds_123"},
        top_k=5,
        min_score=0.1,
    )

    results, trace = searcher.search_with_trace("query")

    assert len(results) == 1
    assert results[0].file_uuid == "doc-1"
    assert results[0].chunk_id == 11
    assert results[0].source == "doc title"
    assert results[0].source_path == "/kb/doc.md"
    assert trace["provider"] == "ragflow"
    assert trace["dataset_id"] == "ds_123"
    assert trace["query_rewrite"] == "rewritten"


def test_ragflow_searcher_filters_by_min_score() -> None:
    client = SimpleNamespace(
        search=lambda **kwargs: SimpleNamespace(  # noqa: ARG005
            rewritten_question="",
            chunks=[
                SimpleNamespace(
                    id="1",
                    document_id="doc-1",
                    content="high",
                    score=0.8,
                    title="a",
                    source="",
                    metadata={},
                ),
                SimpleNamespace(
                    id="2",
                    document_id="doc-2",
                    content="low",
                    score=0.15,
                    title="b",
                    source="",
                    metadata={},
                ),
            ],
        )
    )
    searcher = RAGFlowSearcher(
        client=client,
        dataset_map={"default": "ds_123"},
        min_score=0.2,
    )

    results, _trace = searcher.search_with_trace("query")
    assert len(results) == 1
    assert results[0].content == "high"


def test_ragflow_searcher_fallbacks_to_legacy() -> None:
    legacy = SimpleNamespace(
        search_with_trace=lambda query: (  # noqa: ARG005
            [
                SearchResult(
                    file_uuid="legacy-doc",
                    source="legacy-source",
                    content="legacy-content",
                    score=0.6,
                    chunk_id=1,
                )
            ],
            {"legacy": True},
        )
    )
    client = SimpleNamespace(search=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("timeout")))  # noqa: ARG005
    searcher = RAGFlowSearcher(
        client=client,
        dataset_map={"default": "ds_123"},
        fallback_to_legacy=True,
        legacy_searcher=legacy,
    )

    results, trace = searcher.search_with_trace("query")

    assert len(results) == 1
    assert results[0].file_uuid == "legacy-doc"
    assert trace["fallback_used"] is True
    assert trace["fallback_provider"] == "legacy"


def test_ragflow_searcher_returns_empty_without_fallback() -> None:
    client = SimpleNamespace(search=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("timeout")))  # noqa: ARG005
    searcher = RAGFlowSearcher(
        client=client,
        dataset_map={"default": "ds_123"},
        fallback_to_legacy=False,
        legacy_searcher=None,
    )

    results, trace = searcher.search_with_trace("query")
    assert results == []
    assert trace["provider"] == "ragflow"
    assert trace["fallback_used"] is False

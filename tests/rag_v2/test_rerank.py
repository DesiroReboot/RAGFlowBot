from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from src.core.search.rag_search import RAGSearcher
from src.core.search.reranker import Reranker


def _candidate(file_uuid: str, chunk_id: int, score: float) -> dict:
    return {
        "file_uuid": file_uuid,
        "chunk_id": chunk_id,
        "source": "kb.md",
        "source_path": "/kb/kb.md",
        "section_title": f"sec-{chunk_id}",
        "content": f"content-{file_uuid}",
        "score": score,
        "grading": {"final_score": score},
    }


class _DummyReranker(Reranker):
    provider = "dummy"

    def score(self, *, query: str, candidates: list[dict], timeout_ms: int) -> list[float]:
        del query, candidates, timeout_ms
        return [0.1, 0.9]


def test_rerank_disabled_keeps_grader_order() -> None:
    db_path = Path.cwd() / f"ragv2-rerank-off-{uuid4().hex}.db"
    try:
        searcher = RAGSearcher(
            db_path=str(db_path),
            top_k=3,
            context_top_k=3,
            rerank_enabled=False,
        )
        candidates = [
            _candidate("doc-a", 0, 0.90),
            _candidate("doc-b", 1, 0.80),
            _candidate("doc-c", 2, 0.70),
        ]

        with (
            patch.object(searcher.hybrid, "retrieve", return_value=([], [], {"branch_errors": {}, "vector_meta": {}})),
            patch.object(searcher.fusion, "fuse", return_value=[]),
            patch.object(searcher.grader, "grade", return_value=(candidates, [])),
            patch.object(
                searcher.context_selector,
                "select",
                side_effect=lambda *, candidates, source_scores, top_k: (candidates[:top_k], []),
            ),
        ):
            results, trace = searcher.search_with_trace("test query")

        assert [item.file_uuid for item in results] == ["doc-a", "doc-b", "doc-c"]
        assert trace["rerank"]["enabled"] is False
        assert trace["rerank"]["success"] is True
    finally:
        if db_path.exists():
            db_path.unlink()


def test_rerank_enabled_reorders_top_n_by_hybrid_score() -> None:
    db_path = Path.cwd() / f"ragv2-rerank-on-{uuid4().hex}.db"
    try:
        searcher = RAGSearcher(
            db_path=str(db_path),
            top_k=3,
            context_top_k=3,
            rerank_enabled=True,
            rerank_top_n=2,
            rerank_weight=1.0,
        )
        candidates = [
            _candidate("doc-a", 0, 0.90),
            _candidate("doc-b", 1, 0.80),
            _candidate("doc-c", 2, 0.70),
        ]
        searcher.reranker = _DummyReranker()

        with (
            patch.object(searcher.hybrid, "retrieve", return_value=([], [], {"branch_errors": {}, "vector_meta": {}})),
            patch.object(searcher.fusion, "fuse", return_value=[]),
            patch.object(searcher.grader, "grade", return_value=(candidates, [])),
            patch.object(
                searcher.context_selector,
                "select",
                side_effect=lambda *, candidates, source_scores, top_k: (candidates[:top_k], []),
            ),
        ):
            results, trace = searcher.search_with_trace("test query")

        assert [item.file_uuid for item in results] == ["doc-b", "doc-a", "doc-c"]
        assert trace["rerank"]["enabled"] is True
        assert trace["rerank"]["success"] is True
        assert trace["rerank"]["input_count"] == 2
    finally:
        if db_path.exists():
            db_path.unlink()

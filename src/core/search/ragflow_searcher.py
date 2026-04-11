from __future__ import annotations

from typing import Any

from src.core.search.rag_search import SearchResult
from src.core.search.ragflow_client import RAGFlowClient


class RAGFlowSearcher:
    def __init__(
        self,
        *,
        client: RAGFlowClient,
        dataset_map: dict[str, str] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        fallback_to_legacy: bool = True,
        legacy_searcher: Any | None = None,
    ) -> None:
        self.client = client
        self.dataset_map = dict(dataset_map or {})
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)
        self.fallback_to_legacy = bool(fallback_to_legacy)
        self.legacy_searcher = legacy_searcher

    def search(self, query: str) -> list[SearchResult]:
        results, _trace = self.search_with_trace(query)
        return results

    def search_with_trace(self, query: str) -> tuple[list[SearchResult], dict[str, Any]]:
        dataset_id = self._resolve_dataset_id()
        try:
            response = self.client.search(
                dataset_id=dataset_id,
                question=query,
                top_k=self.top_k,
                filters=None,
            )
            filtered = [item for item in response.chunks if float(item.score) >= self.min_score]
            results = [
                SearchResult(
                    file_uuid=item.document_id or item.id or f"ragflow-{idx}",
                    source=item.title or item.source or f"ragflow_doc_{idx}",
                    content=item.content,
                    score=float(item.score),
                    chunk_id=self._coerce_chunk_id(item.id, idx),
                    matched_terms=[],
                    retrieval_paths=[
                        {
                            "source": "ragflow",
                            "rank": idx,
                            "score": float(item.score),
                        }
                    ],
                    grading={},
                    source_path=item.source,
                    section_title=item.title,
                )
                for idx, item in enumerate(filtered, start=1)
            ]
            trace = {
                "provider": "ragflow",
                "dataset_id": dataset_id,
                "query_rewrite": response.rewritten_question,
                "raw_count": len(response.chunks),
                "selected_count": len(results),
                "min_score": self.min_score,
                "errors": [],
            }
            return results, trace
        except Exception as exc:
            if self.fallback_to_legacy and self.legacy_searcher is not None:
                fallback_results, fallback_trace = self.legacy_searcher.search_with_trace(query)
                trace = {
                    "provider": "ragflow",
                    "fallback_provider": "legacy",
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                    "dataset_id": dataset_id,
                    "legacy_trace": dict(fallback_trace or {}),
                    "errors": [str(exc)],
                }
                return fallback_results, trace
            return [], {
                "provider": "ragflow",
                "dataset_id": dataset_id,
                "fallback_used": False,
                "errors": [str(exc)],
            }

    def _resolve_dataset_id(self) -> str:
        for key in ("default", "*"):
            value = str(self.dataset_map.get(key, "")).strip()
            if value:
                return value
        for value in self.dataset_map.values():
            text = str(value).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _coerce_chunk_id(raw: str, default: int) -> int:
        try:
            return int(raw)
        except Exception:
            return int(default)

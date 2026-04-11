from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.search.context_selector import ContextSelector
from src.core.search.fusion import ReciprocalRankFusion
from src.core.search.grader import ResultGrader
from src.core.search.hybrid_retriever import HybridRetriever
from src.core.search.query_preprocessor import QueryPreprocessor
from src.RAG.config.kbase_config import KBaseConfig


@dataclass
class SearchResult:
    file_uuid: str
    source: str
    content: str
    score: float
    chunk_id: int = 0
    matched_terms: list[str] = field(default_factory=list)
    retrieval_paths: list[dict[str, Any]] = field(default_factory=list)
    grading: dict[str, float] = field(default_factory=dict)
    source_path: str = ""
    section_title: str = ""


class LegacyRAGSearcher:
    def __init__(
        self,
        db_path: str,
        top_k: int = 5,
        *,
        fts_top_k: int = 20,
        vec_top_k: int = 20,
        fusion_rrf_k: int = 60,
        vector_dimension: int = 768,
        context_top_k: int = 6,
        embedding_provider: str = "mock",
        embedding_base_url: str = "",
        embedding_api_key: str = "",
        embedding_model: str = "mock-embedding-v1",
        embedding_batch_size: int = 10,
        embedding_timeout: int = 20,
        embedding_max_retries: int = 3,
        source_quota_mode: str = "balanced",
        max_chunks_per_source: int = 0,
        qa_anchor_enabled: bool = True,
        semantic_guard_enabled: bool = True,
    ):
        self.db_path = db_path
        self.top_k = top_k
        self.fts_top_k = fts_top_k
        self.vec_top_k = vec_top_k
        self.context_top_k = context_top_k
        self.preprocessor = QueryPreprocessor()
        self.config = KBaseConfig(
            db_path=db_path,
            vector_dimension=vector_dimension,
            rag_top_k=top_k,
            fts_top_k=fts_top_k,
            vec_top_k=vec_top_k,
            fusion_rrf_k=fusion_rrf_k,
            context_top_k=context_top_k,
            embedding_provider=embedding_provider,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_timeout=embedding_timeout,
            embedding_max_retries=embedding_max_retries,
        )
        self.hybrid = HybridRetriever(db_path, self.config)
        self.fusion = ReciprocalRankFusion(fusion_rrf_k)
        self.grader = ResultGrader(
            qa_anchor_enabled=qa_anchor_enabled,
            semantic_guard_enabled=semantic_guard_enabled,
        )
        self.context_selector = ContextSelector(
            source_quota_mode=source_quota_mode,
            max_chunks_per_source=max_chunks_per_source,
        )

    def search(self, query: str) -> list[SearchResult]:
        results, _ = self.search_with_trace(query)
        return results

    def search_with_trace(self, query: str) -> tuple[list[SearchResult], dict[str, Any]]:
        preprocess = self.preprocessor.process(query)
        fts_results, vec_results, hybrid_meta = self.hybrid.retrieve(
            query=str(preprocess["normalized"]),
            fts_limit=self.fts_top_k,
            vec_limit=self.vec_top_k,
        )
        branch_errors = hybrid_meta.get("branch_errors", {})

        try:
            fused = self.fusion.fuse(fts_results, vec_results)
            candidates, source_scores = self.grader.grade(
                query_tokens=list(preprocess["tokens"]),
                query_theme_hints=list(preprocess.get("theme_hints", [])),
                fused_results=fused,
                query_intent=dict(preprocess.get("query_intent", {})),
            )
            selected, citations = self.context_selector.select(
                candidates=candidates,
                source_scores=source_scores,
                top_k=self.context_top_k,
            )
        except Exception as exc:
            # Keep lexical branch available when downstream fusion/grading fails.
            fallback = fts_results[: self.top_k]
            results = [
                SearchResult(
                    file_uuid=str(item["file_uuid"]),
                    source=str(item.get("source", "")),
                    content=str(item.get("content", "")),
                    score=1.0 / (1 + idx),
                    chunk_id=int(item["chunk_id"]),
                    matched_terms=[
                        token
                        for token in preprocess["tokens"]
                        if token in str(item.get("content", "")).lower()
                    ],
                    retrieval_paths=[
                        {
                            "source": "fts",
                            "rank": item.get("fts_rank"),
                            "score": item.get("fts_raw_score"),
                        }
                    ],
                    grading={},
                    source_path=str(item.get("source_path", "")),
                    section_title=str(item.get("section_title", "")),
                )
                for idx, item in enumerate(fallback, start=1)
            ]
            return results, {
                "query": {"text": query},
                "preprocess": preprocess,
                "fts_recall": fts_results,
                "vector_recall": vec_results,
                "fusion": [],
                "candidate_grading": [],
                "source_grading": [],
                "context_selection": fallback,
                "branch_diagnostics": {
                    "summary": {
                        "fts_hits": len(fts_results),
                        "vector_hits": len(vec_results),
                        "overlap_hits": 0,
                        "selected_total": len(results),
                    },
                    "branch_contribution": {},
                    "eliminated_candidates": [],
                    "hard_filtered_candidates": list(getattr(self.grader, "last_hard_filtered", [])),
                    "conflict_pool_candidates": list(getattr(self.grader, "last_conflict_pool", [])),
                },
                "generation": {
                    "error": str(exc),
                    "vector_meta": hybrid_meta.get("vector_meta", {}),
                    "branch_errors": branch_errors,
                },
                "citations": [],
                "final_results": [
                    {
                        "file_uuid": item.file_uuid,
                        "chunk_id": item.chunk_id,
                        "source": item.source,
                        "source_path": item.source_path,
                        "section_title": item.section_title,
                        "content": item.content,
                        "score": item.score,
                        "grading": item.grading,
                    }
                    for item in results
                ],
                "errors": [str(exc)],
            }

        results = [
            SearchResult(
                file_uuid=str(item["file_uuid"]),
                source=str(item.get("source", "")),
                content=str(item.get("content", "")),
                score=float(item["score"]),
                chunk_id=int(item["chunk_id"]),
                matched_terms=[
                    token
                    for token in preprocess["tokens"]
                    if token in str(item.get("content", "")).lower()
                ],
                retrieval_paths=[
                    {
                        "source": "fts",
                        "rank": item.get("fts_rank"),
                        "score": item.get("fts_raw_score"),
                    }
                    for _ in [0]
                    if item.get("fts_rank")
                ]
                + [
                    {
                        "source": "vec",
                        "rank": item.get("vec_rank"),
                        "score": item.get("vec_similarity"),
                    }
                    for _ in [0]
                    if item.get("vec_rank")
                ],
                grading=dict(item.get("grading", {})),
                source_path=str(item.get("source_path", "")),
                section_title=str(item.get("section_title", "")),
            )
            for item in selected[: self.top_k]
        ]
        trace = {
            "query": {"text": query},
            "preprocess": preprocess,
            "fts_recall": fts_results,
            "vector_recall": vec_results,
            "fusion": fused,
            "candidate_grading": candidates,
            "source_grading": source_scores,
            "context_selection": selected,
            "branch_diagnostics": self._build_branch_diagnostics(
                fts_results=fts_results,
                vec_results=vec_results,
                fused_results=fused,
                candidate_results=candidates,
                selected_results=selected,
                hard_filtered=list(getattr(self.grader, "last_hard_filtered", [])),
                conflict_pool=list(getattr(self.grader, "last_conflict_pool", [])),
            ),
            "generation": {
                "selected_count": len(results),
                "vector_meta": hybrid_meta.get("vector_meta", {}),
                "branch_errors": branch_errors,
            },
            "citations": citations,
            "final_results": [
                {
                    "file_uuid": item.file_uuid,
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "source_path": item.source_path,
                    "section_title": item.section_title,
                    "content": item.content,
                    "score": item.score,
                    "grading": item.grading,
                }
                for item in results
            ],
        }
        if branch_errors:
            trace["errors"] = [f"{name}: {error}" for name, error in branch_errors.items()]
        return results, trace

    def _build_branch_diagnostics(
        self,
        *,
        fts_results: list[dict[str, Any]],
        vec_results: list[dict[str, Any]],
        fused_results: list[dict[str, Any]],
        candidate_results: list[dict[str, Any]],
        selected_results: list[dict[str, Any]],
        hard_filtered: list[dict[str, Any]] | None = None,
        conflict_pool: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        def _key(item: dict[str, Any]) -> tuple[str, int]:
            return (str(item.get("file_uuid", "")), int(item.get("chunk_id", 0)))

        fts_keys = {_key(item) for item in fts_results}
        vec_keys = {_key(item) for item in vec_results}
        selected_keys = {_key(item) for item in selected_results}
        overlap = fts_keys & vec_keys

        branch_selected_counts = {"fts_only": 0, "vector_only": 0, "hybrid": 0}
        branch_selected_scores = {"fts_only": 0.0, "vector_only": 0.0, "hybrid": 0.0}
        for item in selected_results:
            key = _key(item)
            score = float(item.get("score", 0.0))
            in_fts = key in fts_keys
            in_vec = key in vec_keys
            if in_fts and in_vec:
                branch = "hybrid"
            elif in_fts:
                branch = "fts_only"
            else:
                branch = "vector_only"
            branch_selected_counts[branch] += 1
            branch_selected_scores[branch] += score

        total_selected_score = sum(branch_selected_scores.values()) or 1.0
        branch_contribution = {
            branch: {
                "count": branch_selected_counts[branch],
                "score_sum": round(branch_selected_scores[branch], 6),
                "score_share": round(branch_selected_scores[branch] / total_selected_score, 6),
            }
            for branch in branch_selected_counts
        }

        selected_by_source: dict[str, int] = {}
        selected_content: set[str] = set()
        source_quotas = getattr(self.context_selector, "last_source_quotas", {})
        source_quota_mode = str(getattr(self.context_selector, "source_quota_mode", "balanced"))
        for item in selected_results:
            source = str(item.get("source", ""))
            selected_by_source[source] = selected_by_source.get(source, 0) + 1
            selected_content.add(str(item.get("content", "")).strip())

        eliminated_candidates: list[dict[str, Any]] = []
        for item in candidate_results:
            key = _key(item)
            if key in selected_keys:
                continue
            source = str(item.get("source", ""))
            content_key = str(item.get("content", "")).strip()
            reason = "score_or_source_priority"
            if source_quota_mode != "unbounded" and selected_by_source.get(source, 0) >= int(
                source_quotas.get(source, 1)
            ):
                reason = "source_soft_quota_reached"
            elif content_key in selected_content:
                reason = "duplicate_content"
            elif len(selected_results) >= self.context_top_k:
                reason = "top_k_limit"

            eliminated_candidates.append(
                {
                    "file_uuid": str(item.get("file_uuid", "")),
                    "chunk_id": int(item.get("chunk_id", 0)),
                    "source": source,
                    "score": round(float(item.get("score", 0.0)), 6),
                    "in_fts": key in fts_keys,
                    "in_vector": key in vec_keys,
                    "reason": reason,
                }
            )

        eliminated_candidates.sort(key=lambda row: row["score"], reverse=True)
        return {
            "summary": {
                "fts_hits": len(fts_keys),
                "vector_hits": len(vec_keys),
                "overlap_hits": len(overlap),
                "fused_total": len(fused_results),
                "selected_total": len(selected_results),
                "eliminated_total": len(eliminated_candidates),
                "hard_filtered_total": len(hard_filtered or []),
                "conflict_pool_total": len(conflict_pool or []),
                "source_quotas": source_quotas,
                "source_quota_mode": source_quota_mode,
            },
            "branch_contribution": branch_contribution,
            "eliminated_candidates": eliminated_candidates,
            "hard_filtered_candidates": list(hard_filtered or []),
            "conflict_pool_candidates": list(conflict_pool or []),
        }


# Backward-compatible alias.
RAGSearcher = LegacyRAGSearcher

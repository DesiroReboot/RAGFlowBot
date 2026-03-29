from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config
from src.core.generation import GenerationClient
from src.core.search.orchestrator import SearchOrchestrator, UnifiedSearchHit
from src.core.search.planner import RulePlanner
from src.core.search.query_analyzer import QueryAnalysis, QueryAnalyzer
from src.core.search.rag_search import RAGSearcher, SearchResult
from src.core.search.source_utils import build_grouped_citations
from src.core.search.web_result_evaluator import WebResultEvaluator
from src.core.search.web_router import WebRouter
from src.core.search.web_search_client import WebSearchClient, WebSearchResult
from src.core.trace_builder import (
    GenerationFallbackReason,
    TraceFallbackReason,
    build_agent_trace,
    build_strategy_fallback_step,
    build_web_trace,
    merge_reason_codes,
    normalize_web_trace,
)
from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.readiness import is_index_ready
from src.RAG.storage.manifest_store import ManifestStore


@dataclass
class AgentResponse:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieval_confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerDraft:
    query: str
    theme: str
    steps: list[str]
    evidence: list[str]
    citations: list[dict[str, Any]]


SearchHit = SearchResult | UnifiedSearchHit


class ReActAgent:
    def __init__(self, config: Config):
        self.config = config
        self.answer_top_k = max(
            1,
            int(max(config.search.rag_top_k, config.search.context_top_k)),
        )
        self.generation_client = GenerationClient(config.generation)
        self.kbase_config = KBaseConfig(
            db_path=config.database.db_path,
            source_dir=config.knowledge_base.source_dir,
            supported_extensions=config.knowledge_base.supported_extensions,
            auto_sync_on_startup=config.knowledge_base.auto_sync_on_startup,
            ocr_enabled=config.knowledge_base.ocr_enabled,
            ocr_language=config.knowledge_base.ocr_language,
            ocr_dpi_scale=config.knowledge_base.ocr_dpi_scale,
            ocr_trigger_readability=config.knowledge_base.ocr_trigger_readability,
            min_chunk_readability=config.knowledge_base.min_chunk_readability,
            vector_dimension=config.embedding.dimension,
            rag_top_k=self.answer_top_k,
            fts_top_k=config.search.fts_top_k,
            vec_top_k=config.search.vec_top_k,
            fusion_rrf_k=config.search.fusion_rrf_k,
            context_top_k=config.search.context_top_k,
            chunk_size=config.knowledge_base.chunk_size,
            chunk_overlap=config.knowledge_base.chunk_overlap,
            embedding_provider=config.embedding.provider,
            embedding_base_url=config.embedding.base_url,
            embedding_api_key=config.embedding.api_key,
            embedding_model=config.embedding.model,
            embedding_batch_size=config.embedding.batch_size,
            embedding_timeout=config.embedding.timeout,
            embedding_max_retries=config.embedding.max_retries,
            build_version=config.knowledge_base.build_version,
        )
        self.manifest_store = ManifestStore(config.database.db_path, ensure_schema=False)
        self.rag_searcher = RAGSearcher(
            db_path=config.database.db_path,
            top_k=self.answer_top_k,
            fts_top_k=config.search.fts_top_k,
            vec_top_k=config.search.vec_top_k,
            fusion_rrf_k=config.search.fusion_rrf_k,
            vector_dimension=config.embedding.dimension,
            context_top_k=config.search.context_top_k,
            embedding_provider=config.embedding.provider,
            embedding_base_url=config.embedding.base_url,
            embedding_api_key=config.embedding.api_key,
            embedding_model=config.embedding.model,
            embedding_batch_size=config.embedding.batch_size,
            embedding_timeout=config.embedding.timeout,
            embedding_max_retries=config.embedding.max_retries,
        )
        self.planner = RulePlanner()
        self.query_analyzer = QueryAnalyzer()
        self.web_search_client = WebSearchClient(
            provider=config.search.web_search_provider,
            timeout=config.search.web_search_timeout,
            max_retries=config.search.web_search_retries,
            tavily_api_key=config.search.web_search_tavily_api_key,
            tavily_base_url=config.search.web_search_tavily_base_url,
            tavily_search_depth=config.search.web_search_depth,
            max_results=config.search.web_search_max_results,
        )
        self.web_result_evaluator = WebResultEvaluator()
        self.web_router = WebRouter(
            direct_thresholds=dict(config.search.web_direct_fusion_thresholds),
        )
        self.search_orchestrator = SearchOrchestrator(
            planner=self.planner,
            rag_searcher=self.rag_searcher,
            web_searcher=self.web_search_client,
            config=config,
            query_analyzer=self.query_analyzer,
            web_result_evaluator=self.web_result_evaluator,
            web_router=self.web_router,
            answer_top_k=self.answer_top_k,
        )
        # Backward-compatible alias for any legacy direct access.
        self.searcher = self.rag_searcher

    def run_sync(self, query: str, include_trace: bool = False) -> AgentResponse:
        try:
            self.answer_top_k = max(1, int(getattr(self, "answer_top_k", 3)))
        except Exception:
            self.answer_top_k = 3

        manifest_gate = self._manifest_gate_snapshot()
        manifest = manifest_gate.get("manifest", {})
        if bool(manifest_gate.get("blocked", False)):
            search_trace: dict[str, Any] = {"manifest_gate": dict(manifest_gate)}
            self._normalize_web_trace(search_trace)
            blocked_trace = build_agent_trace(
                query=query,
                search_trace=search_trace,
                manifest=manifest or {},
            )
            blocked_trace["strategy_execution"] = [
                build_strategy_fallback_step(reason=TraceFallbackReason.INDEX_NOT_READY)
            ]
            blocked_trace["final_citations"] = []
            blocked_trace["retrieval_confidence"] = 0.0
            answer = "Index is not ready yet. Please initialize or sync the knowledge base first."
            blocked_trace["final_answer_preview"] = answer
            return AgentResponse(
                answer=answer,
                citations=[],
                retrieval_confidence=0.0,
                trace=blocked_trace if include_trace else {},
            )

        citations_from_search: list[dict[str, Any]] | None = None
        confidence_from_search = 0.0
        results: list[SearchHit] = []
        search_trace = {}
        if hasattr(self, "search_orchestrator") and self.search_orchestrator is not None:
            if hasattr(self.search_orchestrator, "answer_top_k"):
                self.search_orchestrator.answer_top_k = self.answer_top_k
            orchestrator_result = self.search_orchestrator.search_with_trace(query)
            results = self._coerce_search_hits(orchestrator_result.hits)
            search_trace = (
                orchestrator_result.trace_search if isinstance(orchestrator_result.trace_search, dict) else {}
            )
            raw_citations = getattr(orchestrator_result, "citations", None)
            if isinstance(raw_citations, list):
                citations_from_search = raw_citations
            confidence_from_search = float(getattr(orchestrator_result, "retrieval_confidence", 0.0) or 0.0)
        else:
            searcher = getattr(self, "searcher", None) or self.rag_searcher
            rag_results, search_trace = searcher.search_with_trace(query)
            results = self._coerce_search_hits(rag_results)

        if not isinstance(search_trace, dict):
            search_trace = {}
        search_trace["manifest_gate"] = dict(manifest_gate)

        if not (hasattr(self, "search_orchestrator") and self.search_orchestrator is not None):
            # Legacy execution path kept for compatibility when orchestrator is not injected.
            query_analysis = QueryAnalysis(
                temporal_intent_score=0.0,
                domain_relevance_score=0.0,
                oov_entity_score=0.0,
                kb_coverage_score=1.0 if results else 0.0,
                need_web_search=False,
                reasons=[],
            )
            query_analyzer = getattr(self, "query_analyzer", None)
            if query_analyzer is not None:
                try:
                    query_analysis = query_analyzer.analyze(
                        query=query,
                        local_results=results,
                        search_trace=search_trace,
                    )
                except Exception:
                    query_analysis.reasons.append("query_analyzer_error")
            else:
                query_analysis.reasons.append("query_analyzer_unavailable")

            web_trace = build_web_trace(
                requested=bool(query_analysis.need_web_search),
                route_mode="legacy",
                need_web_search=bool(query_analysis.need_web_search),
                reasons=list(query_analysis.reasons),
                metrics={
                    "temporal_intent_score": float(query_analysis.temporal_intent_score),
                    "domain_relevance_score": float(query_analysis.domain_relevance_score),
                    "oov_entity_score": float(query_analysis.oov_entity_score),
                    "kb_coverage_score": float(query_analysis.kb_coverage_score),
                    "kb_result_count": len(results),
                },
            )
            web_routing_ready = (
                hasattr(self, "config")
                and hasattr(getattr(self, "config"), "search")
                and hasattr(self, "web_search_client")
                and hasattr(self, "web_result_evaluator")
                and hasattr(self, "web_router")
            )
            if web_routing_ready:
                try:
                    results, web_trace = self._apply_web_routing(
                        query=query,
                        local_results=results,
                        query_analysis=query_analysis,
                    )
                except Exception as exc:
                    web_trace["fallback_used"] = True
                    web_trace["reasons"] = self._merge_reasons(
                        list(web_trace.get("reasons", [])),
                        [TraceFallbackReason.WEB_ROUTING_ERROR.value],
                    )
                    web_trace["error"] = str(exc)
            else:
                web_trace["reasons"] = self._merge_reasons(
                    list(web_trace.get("reasons", [])),
                    [TraceFallbackReason.WEB_ROUTING_UNAVAILABLE.value],
                )
            search_trace["web"] = web_trace

        self._normalize_web_trace(search_trace)
        trace = build_agent_trace(query=query, search_trace=search_trace, manifest=manifest or {})

        if not results:
            planner_trace = search_trace.get("planner", {}) if isinstance(search_trace, dict) else {}
            allow_rag = True
            filter_reason = str(planner_trace.get("filter_reason", "")).strip()

            if not allow_rag:
                reason = TraceFallbackReason.DOMAIN_OUT_OF_SCOPE.value
                answer = (
                    "当前问题不在外贸/跨境电商知识域内，已跳过知识库检索。"
                    "请改问选品、Listing、广告投放、物流、关税、平台运营等相关问题。"
                )
            else:
                reason = (
                    TraceFallbackReason.INDEX_NOT_READY.value
                    if bool(manifest_gate.get("blocked", False))
                    else TraceFallbackReason.NO_RETRIEVAL_RESULTS.value
                )
                answer = (
                    "当前索引尚未就绪，请先执行离线知识库同步。"
                    if reason == TraceFallbackReason.INDEX_NOT_READY.value
                    else "未从知识库检索到足够相关内容，请补充更具体的问题后重试。"
                )
            trace["strategy_execution"].append(
                build_strategy_fallback_step(reason=reason, filter_reason=filter_reason)
            )
            trace["final_answer_preview"] = answer
            trace["final_citations"] = []
            trace["retrieval_confidence"] = 0.0
            return AgentResponse(
                answer=answer,
                citations=[],
                retrieval_confidence=0.0,
                trace=trace if include_trace else {},
            )

        selected = results[: self.answer_top_k]
        citations = citations_from_search or self._build_citations(selected)
        draft = self._build_answer_draft(query=query, selected=selected, citations=citations)
        template_answer = self._render_template_answer(draft)
        answer, generation_meta = self._compose_answer(
            draft=draft,
            template_answer=template_answer,
            search_trace=search_trace,
        )
        confidence = max(
            confidence_from_search,
            min(1.0, sum(max(0.0, item.score) for item in selected) / max(len(selected), 1)),
        )

        trace["strategy_execution"].append(
            {
                "stage": "context_selection",
                "selected_results": [asdict(item) for item in selected],
            }
        )
        trace["strategy_execution"].append(
            {
                "stage": "answer_generation",
                "meta": generation_meta,
            }
        )
        trace["final_answer_preview"] = answer
        trace["final_citations"] = citations
        trace["retrieval_confidence"] = confidence
        trace["retrieved_results"] = [asdict(result) for result in results]

        return AgentResponse(
            answer=answer,
            citations=citations,
            retrieval_confidence=confidence,
            trace=trace if include_trace else {},
        )

    def _manifest_gate_snapshot(self) -> dict[str, Any]:
        manifest_store = getattr(self, "manifest_store", None)
        if manifest_store is None or not hasattr(manifest_store, "get_manifest"):
            return {
                "ready": True,
                "blocked": False,
                "status": "unknown",
                "reason": "manifest_store_unavailable",
                "manifest": {},
            }

        try:
            raw_manifest = manifest_store.get_manifest()
        except Exception:
            return {
                "ready": True,
                "blocked": False,
                "status": "unknown",
                "reason": "manifest_read_error",
                "manifest": {},
            }

        manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
        ready, reason, status = is_index_ready(manifest)
        return {
            "ready": bool(ready),
            "blocked": not bool(ready),
            "status": status,
            "reason": reason,
            "manifest": manifest,
        }

    def _apply_web_routing(
        self,
        *,
        query: str,
        local_results: list[SearchHit],
        query_analysis: QueryAnalysis,
    ) -> tuple[list[SearchHit], dict[str, Any]]:
        enabled = bool(self.config.search.web_search_enabled)
        kb_empty = len(local_results) == 0
        need_web_search = bool(query_analysis.need_web_search) or kb_empty
        reasons = [str(reason) for reason in query_analysis.reasons if str(reason).strip()]
        if kb_empty:
            reasons = self._merge_reasons(reasons, ["kb_empty_triggered_web_fallback"])
        web_trace = build_web_trace(
            requested=need_web_search,
            route_mode="legacy",
            need_web_search=need_web_search,
            reasons=list(reasons),
            metrics={
                "temporal_intent_score": float(query_analysis.temporal_intent_score),
                "domain_relevance_score": float(query_analysis.domain_relevance_score),
                "oov_entity_score": float(query_analysis.oov_entity_score),
                "kb_coverage_score": float(query_analysis.kb_coverage_score),
                "kb_result_count": len(local_results),
            },
        )
        if not enabled:
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.WEB_SEARCH_DISABLED.value]
            )
            return local_results, web_trace
        if not need_web_search:
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.WEB_NOT_REQUIRED.value]
            )
            return local_results, web_trace

        try:
            web_results = self.web_search_client.search(
                query,
                limit=max(int(self.config.search.web_rag_max_docs), self.answer_top_k),
            )
        except Exception as exc:
            web_trace["fallback_used"] = True
            error_text = str(exc)
            error_reasons = [TraceFallbackReason.WEB_SEARCH_ERROR.value]
            if TraceFallbackReason.PROVIDER_MISCONFIGURED.value in error_text:
                error_reasons.append(TraceFallbackReason.PROVIDER_MISCONFIGURED.value)
            web_trace["reasons"] = self._merge_reasons(web_trace["reasons"], error_reasons)
            web_trace["error"] = error_text
            return local_results, web_trace

        web_trace["metrics"]["provider"] = self.config.search.web_search_provider
        web_trace["metrics"]["result_count_raw"] = len(web_results)
        if not web_results:
            web_trace["fallback_used"] = True
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.WEB_NO_RESULTS.value]
            )
            return local_results, web_trace

        evaluation = self.web_result_evaluator.evaluate(query=query, results=web_results)
        decision = self.web_router.route(
            query=query,
            analysis=query_analysis,
            evaluation=evaluation,
        )
        web_trace["fusion_strategy"] = decision.fusion_strategy
        web_trace["reasons"] = self._merge_reasons(web_trace["reasons"], decision.reasons)
        web_trace["metrics"].update(decision.metrics)
        web_trace["fallback_used"] = bool(decision.fallback)

        if decision.fusion_strategy == "direct_fusion":
            fused = self._build_direct_fusion_results(
                query=query,
                local_results=local_results,
                web_results=web_results,
            )
            if fused:
                return fused, web_trace
            web_trace["fallback_used"] = True
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.DIRECT_FUSION_EMPTY.value]
            )
            return local_results, web_trace

        if decision.fusion_strategy == "rag_fusion":
            fused = self._build_rag_fusion_results(
                query=query,
                local_results=local_results,
                web_results=web_results,
            )
            if fused:
                return fused, web_trace
            web_trace["fallback_used"] = True
            web_trace["reasons"] = self._merge_reasons(
                web_trace["reasons"], [TraceFallbackReason.RAG_FUSION_EMPTY.value]
            )
            return local_results, web_trace

        web_trace["fallback_used"] = True
        return local_results, web_trace

    def _build_direct_fusion_results(
        self,
        *,
        query: str,
        local_results: list[SearchHit],
        web_results: list[WebSearchResult],
    ) -> list[SearchHit]:
        limit = max(self.answer_top_k * 3, self.answer_top_k)
        web_limit = min(int(self.config.search.web_rag_max_docs), 8)
        web_rows = self._convert_web_results(query=query, web_results=web_results[:web_limit])
        ordered = sorted(web_rows + list(local_results), key=lambda row: float(getattr(row, "score", 0.0)), reverse=True)
        return self._dedupe_results(ordered, limit=limit)

    def _build_rag_fusion_results(
        self,
        *,
        query: str,
        local_results: list[SearchHit],
        web_results: list[WebSearchResult],
    ) -> list[SearchHit]:
        web_rows = self._convert_web_results(
            query=query,
            web_results=web_results[: int(self.config.search.web_rag_max_docs)],
        )
        query_terms = set(self._query_terms(query))
        scored_rows: list[tuple[float, SearchHit]] = []
        for row in list(local_results) + web_rows:
            base_score = max(0.0, float(getattr(row, "score", 0.0)))
            content = str(getattr(row, "content", "")).lower()
            overlap = 0.0
            if query_terms:
                overlap = sum(1 for token in query_terms if token in content) / len(query_terms)
            source_path = str(getattr(row, "source_path", ""))
            traceable_bonus = 0.06 if source_path.startswith(("http://", "https://")) else 0.0
            final_score = min(1.0, base_score * 0.75 + overlap * 0.19 + traceable_bonus)
            scored_rows.append((final_score, row))

        scored_rows.sort(key=lambda row: row[0], reverse=True)
        reranked = [row for _, row in scored_rows]
        return self._dedupe_results(
            reranked,
            limit=max(self.answer_top_k * 4, int(self.config.search.web_rag_max_docs)),
        )

    def _convert_web_results(
        self,
        *,
        query: str,
        web_results: list[WebSearchResult],
    ) -> list[SearchHit]:
        query_terms = set(self._query_terms(query))
        converted: list[SearchHit] = []
        for index, row in enumerate(web_results, start=1):
            text = f"{row.title} {row.snippet}".strip()
            if query_terms:
                overlap = sum(1 for token in query_terms if token in text.lower()) / len(query_terms)
            else:
                overlap = 0.0
            score = min(1.0, max(float(row.score), 0.45 + overlap * 0.45))
            source = row.title.strip() or row.source_domain or f"web_result_{index}"
            content = row.snippet.strip() or row.title.strip() or row.url.strip()
            if row.url:
                content = f"{content}\nURL: {row.url}"
            converted.append(
                self._build_search_result(
                    file_uuid=f"web-{index}",
                    source=source,
                    content=content,
                    score=score,
                    chunk_id=index,
                    retrieval_paths=[{"source": "web", "rank": index, "score": score}],
                    grading={"web_score": score},
                    source_path=row.url,
                    section_title=row.source_domain,
                )
            )
        return converted

    def _build_search_result(
        self,
        *,
        file_uuid: str,
        source: str,
        content: str,
        score: float,
        chunk_id: int,
        retrieval_paths: list[dict[str, Any]],
        grading: dict[str, float],
        source_path: str,
        section_title: str,
    ) -> SearchResult:
        return SearchResult(
            file_uuid=file_uuid,
            source=source,
            content=content,
            score=score,
            chunk_id=chunk_id,
            matched_terms=[],
            retrieval_paths=retrieval_paths,
            grading=grading,
            source_path=source_path,
            section_title=section_title,
        )

    def _dedupe_results(self, rows: list[SearchHit], *, limit: int) -> list[SearchHit]:
        deduped: list[SearchHit] = []
        seen: set[str] = set()
        for row in rows:
            source = str(getattr(row, "source", "")).strip().lower()
            source_path = str(getattr(row, "source_path", "")).strip().lower()
            content = str(getattr(row, "content", "")).strip().lower()
            key = source_path or f"{source}|{content[:120]}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _coerce_search_hits(rows: Iterable[SearchHit]) -> list[SearchHit]:
        return [row for row in rows]

    def _merge_reasons(self, left: list[str], right: list[str]) -> list[str]:
        return merge_reason_codes(left, right)

    def _normalize_web_trace(self, search_trace: dict[str, Any]) -> None:
        normalize_web_trace(search_trace)

    def _build_citations(self, selected: list[Any]) -> list[dict[str, Any]]:
        return build_grouped_citations(selected)

    def _build_answer_draft(
        self,
        *,
        query: str,
        selected: list[Any],
        citations: list[dict[str, Any]],
    ) -> AnswerDraft:
        query_terms = self._query_terms(query)
        theme = self._detect_theme(query, selected)
        evidence = self._extract_evidence(query=query, selected=selected, limit=6)
        steps = self._build_thematic_steps(theme=theme, query_terms=query_terms)
        return AnswerDraft(
            query=query,
            theme=theme,
            steps=steps,
            evidence=evidence,
            citations=citations,
        )

    def _compose_answer(
        self,
        *,
        draft: AnswerDraft,
        template_answer: str,
        search_trace: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        requested_mode = str(self.config.generation.mode or "hybrid").strip().lower()
        if requested_mode not in {"template", "hybrid", "llm_rewrite"}:
            requested_mode = "hybrid"

        generation_meta: dict[str, Any] = {
            "requested_mode": requested_mode,
            "final_mode": "template",
            "fallback_reason": "",
            "quality_score": 1.0,
            "claim_support_rate": 1.0,
            "citation_coverage": 1.0,
        }
        if requested_mode == "template":
            return template_answer, generation_meta

        abnormal_reason = self._generation_abnormal_reason(search_trace)
        if abnormal_reason:
            generation_meta["fallback_reason"] = abnormal_reason
            return template_answer, generation_meta

        try:
            rewritten = self._hybrid_rewrite(draft=draft, template_answer=template_answer)
        except Exception as exc:
            generation_meta["fallback_reason"] = GenerationFallbackReason.HYBRID_UNAVAILABLE_OR_ERROR.value
            generation_meta["error"] = str(exc)
            return template_answer, generation_meta

        quality_score, quality_issues = self._evaluate_answer_quality(rewritten, draft)
        claim_support_rate = self._estimate_claim_support(rewritten, draft.evidence)
        citation_coverage = self._estimate_citation_coverage(rewritten, draft.citations)

        generation_meta["quality_score"] = round(quality_score, 4)
        generation_meta["claim_support_rate"] = round(claim_support_rate, 4)
        generation_meta["citation_coverage"] = round(citation_coverage, 4)
        if quality_issues:
            generation_meta["quality_issues"] = quality_issues

        if quality_score < float(self.config.generation.min_quality_score):
            generation_meta["fallback_reason"] = GenerationFallbackReason.QUALITY_BELOW_THRESHOLD.value
            return template_answer, generation_meta
        if claim_support_rate < float(self.config.generation.min_claim_support_rate):
            generation_meta["fallback_reason"] = GenerationFallbackReason.CLAIM_SUPPORT_BELOW_THRESHOLD.value
            return template_answer, generation_meta
        if citation_coverage < float(self.config.generation.min_citation_coverage):
            generation_meta["fallback_reason"] = GenerationFallbackReason.CITATION_COVERAGE_BELOW_THRESHOLD.value
            return template_answer, generation_meta

        generation_meta["final_mode"] = "hybrid"
        return rewritten, generation_meta

    def _generation_abnormal_reason(self, search_trace: dict[str, Any]) -> str:
        fts_recall = search_trace.get("fts_recall", [])
        vec_recall = search_trace.get("vector_recall", [])
        has_fts = isinstance(fts_recall, list) and bool(fts_recall)
        has_vec = isinstance(vec_recall, list) and bool(vec_recall)
        if "fts_recall" in search_trace and not has_fts and not has_vec:
            return GenerationFallbackReason.FTS_NO_HIT.value

        generation_trace = search_trace.get("generation", {})
        if isinstance(generation_trace, dict):
            if generation_trace.get("error"):
                return GenerationFallbackReason.SEARCH_GENERATION_ERROR.value
            branch_errors = generation_trace.get("branch_errors", {})
            selected_count = int(generation_trace.get("selected_count", 0) or 0)
            if selected_count <= 0 and not (has_fts or has_vec):
                return GenerationFallbackReason.NO_RETRIEVAL_RESULTS.value
            if isinstance(branch_errors, dict) and branch_errors:
                # Keep generation enabled when at least one retrieval branch is healthy.
                if "vec" in branch_errors and not has_fts:
                    return GenerationFallbackReason.VECTOR_BRANCH_ERROR_NO_LEXICAL_BACKUP.value
                if "fts" in branch_errors and not has_vec:
                    return GenerationFallbackReason.FTS_BRANCH_ERROR_NO_VECTOR_BACKUP.value

        errors = search_trace.get("errors", [])
        if isinstance(errors, list) and errors and not (has_fts or has_vec):
            return GenerationFallbackReason.SEARCH_ERROR.value
        return ""

    def _hybrid_rewrite(self, *, draft: AnswerDraft, template_answer: str) -> str:
        citation_sources = [
            str(citation.get("source", "")).strip()
            for citation in draft.citations
            if str(citation.get("source", "")).strip()
        ]
        rewritten = self.generation_client.rewrite(
            query=draft.query,
            template_answer=template_answer,
            steps=draft.steps,
            evidence=draft.evidence,
            citation_sources=citation_sources,
        )
        return rewritten.strip()

    def _evaluate_answer_quality(self, answer: str, draft: AnswerDraft) -> tuple[float, list[str]]:
        score = 1.0
        issues: list[str] = []
        required_sections = ["问题：", "建议执行步骤：", "参考来源："]
        if draft.evidence:
            required_sections.append("关键信息：")
        missing_sections = [section for section in required_sections if section not in answer]
        if missing_sections:
            score -= 0.45
            issues.append(f"missing_sections:{','.join(missing_sections)}")

        step_count = len(re.findall(r"(?m)^\d+\.\s+", answer))
        if step_count < min(3, len(draft.steps)):
            score -= 0.2
            issues.append("insufficient_step_count")

        if len(answer.strip()) < 80:
            score -= 0.2
            issues.append("answer_too_short")

        readability = self._readability_ratio(answer)
        if readability < 0.65:
            score -= 0.2
            issues.append("low_readability")

        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if lines:
            unique_ratio = len(set(lines)) / len(lines)
            if unique_ratio < 0.65:
                score -= 0.15
                issues.append("high_repetition")

        return max(0.0, min(1.0, score)), issues

    def _estimate_claim_support(self, answer: str, evidence: list[str]) -> float:
        if not evidence:
            return 1.0
        claims = self._split_claims(answer)
        if not claims:
            return 0.0
        evidence_tokens = [self._text_tokens(line) for line in evidence if line.strip()]
        if not evidence_tokens:
            return 0.0

        supported = 0
        for claim in claims:
            claim_tokens = self._text_tokens(claim)
            if not claim_tokens:
                continue
            best = max((self._token_overlap(claim_tokens, row) for row in evidence_tokens), default=0.0)
            if best >= 0.12:
                supported += 1
        return supported / max(len(claims), 1)

    def _estimate_citation_coverage(self, answer: str, citations: list[dict[str, Any]]) -> float:
        expected_sources = [
            str(citation.get("source", "")).strip().lower()
            for citation in citations
            if str(citation.get("source", "")).strip()
        ]
        if not expected_sources:
            return 1.0
        lowered = answer.lower()
        hit = sum(1 for source in expected_sources if source in lowered)
        return hit / len(expected_sources)

    def _readability_ratio(self, text: str) -> float:
        if not text.strip():
            return 0.0
        readable = re.findall(r"[A-Za-z0-9\u4e00-\u9fff，。！？；：、（）()《》“”‘’\- .:\n]", text)
        return len(readable) / max(len(text), 1)

    def _split_claims(self, answer: str) -> list[str]:
        lines = []
        in_reference = False
        for raw in answer.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("参考来源："):
                in_reference = True
                continue
            if in_reference:
                continue
            if line.endswith("：") and line in {"问题：", "建议执行步骤：", "关键信息："}:
                continue
            lines.append(line)

        claims: list[str] = []
        for line in lines:
            parts = re.split(r"[。！？；\n]+", line)
            for part in parts:
                normalized = re.sub(r"\s+", " ", part).strip(" -\t")
                if len(normalized) >= 8:
                    claims.append(normalized)
        return claims

    def _text_tokens(self, text: str) -> set[str]:
        lowered = text.lower()
        latin_tokens = set(re.findall(r"[a-z0-9_]{2,}", lowered))
        cjk_tokens = set(re.findall(r"[\u4e00-\u9fff]", lowered))
        return latin_tokens | cjk_tokens

    def _token_overlap(self, left: set[str], right: set[str]) -> float:
        if not left:
            return 0.0
        return len(left & right) / len(left)

    def _render_template_answer(self, draft: AnswerDraft) -> str:
        lines = [f"问题：{draft.query}", "建议执行步骤："]
        for idx, step in enumerate(draft.steps, start=1):
            lines.append(f"{idx}. {step}")

        if draft.evidence:
            lines.append("关键信息：")
            for sentence in draft.evidence[:3]:
                lines.append(f"- {sentence}")

        lines.append("参考来源：")
        for citation in draft.citations:
            aliases = [str(alias) for alias in citation.get("aliases", []) if str(alias).strip()]
            if aliases:
                lines.append(f"- {citation['source']} (备选版本: {'; '.join(aliases)})")
            else:
                lines.append(f"- {citation['source']}")
        return "\n".join(lines)

    def _build_human_answer(self, *, query: str, selected: list[Any]) -> str:
        citations = self._build_citations(selected)
        draft = self._build_answer_draft(query=query, selected=selected, citations=citations)
        return self._render_template_answer(draft)

    def _detect_theme(self, query: str, selected: list[Any]) -> str:
        query_text = query.lower()
        source_text = " ".join(str(getattr(item, "source", "")) for item in selected).lower()
        theme_keywords = {
            "product_selection": ["选品", "类目", "需求", "利润", "竞争"],
            "listing": ["listing", "标题", "主图", "关键词", "上架", "五点"],
            "advertising": ["acos", "广告", "ppc", "出价", "投放"],
            "logistics": ["物流", "发货", "fba", "货运", "报关", "装船"],
            "customer_service": ["客服", "消息", "差评", "售后", "回复"],
            "inventory": ["库存", "断货", "补货", "周转", "安全库存"],
            "promotion": ["促销", "折扣", "活动", "秒杀", "优惠"],
            "conversion": ["转化", "点击", "加购", "成交", "评价"],
            "account_security": ["关联", "封号", "账号", "安全", "风控"],
        }
        best_theme = "general"
        best_score = 0
        for theme, keywords in theme_keywords.items():
            query_score = sum(1 for keyword in keywords if keyword in query_text)
            source_score = sum(1 for keyword in keywords if keyword in source_text)
            score = 3 * query_score + source_score
            if score > best_score:
                best_score = score
                best_theme = theme
        return best_theme

    def _build_thematic_steps(self, *, theme: str, query_terms: list[str]) -> list[str]:
        focus = "、".join(query_terms[:3]) if query_terms else "当前问题"
        mapping = {
            "product_selection": [
                "先做市场需求验证，确认目标人群、使用场景和销量趋势。",
                "再评估竞争强度，重点看卖家数量、评价密度和价格带。",
                "核算利润空间，至少覆盖采购、物流、平台费用和广告成本。",
                "检查供应链与合规风险，避免侵权、禁运和交期不稳定。",
                "用小批量测试验证结果，再按数据迭代选品策略。",
            ],
            "listing": [
                "先确定核心关键词与主卖点，明确标题和主图表达重点。",
                "优化标题、五点和详情页，确保信息一致且可搜索。",
                "补齐转化要素，如价格策略、评价素材和信任信息。",
                "上线后跟踪曝光、点击和转化，定位弱项逐项优化。",
            ],
            "advertising": [
                "先拆分投放目标与预算，区分引流词和转化词。",
                "按词分组设置出价与匹配方式，避免高耗低转化。",
                "持续清理无效词，保留高相关高转化词。",
                "结合 ACOS 与利润线做调价，维持可持续投放。",
            ],
            "logistics": [
                "先确认发货模式与时效要求，匹配对应物流方案。",
                "核对包装、标签与报关资料，降低异常和退件风险。",
                "按体积重与实重核算运费，控制单位物流成本。",
                "建立节点追踪和异常预案，保证交付稳定。",
            ],
            "customer_service": [
                "先设定回复时效标准，保证关键消息及时处理。",
                "按问题类型准备标准话术，提高首轮解决率。",
                "针对差评先定位根因，再给出可执行补救方案。",
                "沉淀高频问题与处理结果，持续优化客服 SOP。",
            ],
            "inventory": [
                "先建立安全库存阈值，明确补货触发条件。",
                "按销量趋势和补货周期预测需求，减少断货风险。",
                "将库存分层管理，优先保障核心 SKU。",
                "每周复盘周转与缺货率，动态调整补货计划。",
            ],
            "promotion": [
                "先明确活动目标与人群，再选择合适促销机制。",
                "控制折扣力度与毛利底线，避免只增量不增利。",
                "将活动与广告节奏联动，提升流量承接效率。",
                "活动后复盘转化和利润，保留有效方案。",
            ],
            "conversion": [
                "先定位转化漏斗卡点，区分流量问题与页面问题。",
                "优化主图、卖点和价格策略，提升点击后成交率。",
                "补齐评价与问答等信任要素，降低决策阻力。",
                "按周跟踪转化率和客单价，持续做 A/B 优化。",
            ],
            "account_security": [
                "先梳理账号环境与操作规范，避免高风险行为。",
                "隔离设备、网络与权限，降低关联概率。",
                "建立异常监控与告警，及时处理风控信号。",
                "保留合规证据链，便于申诉与风险复盘。",
            ],
            "general": [
                f"先明确“{focus}”的目标、约束和成功标准。",
                "优先使用可核验的数据或规则做判断，避免仅凭单一片段决策。",
                "把关键结论拆成步骤执行，并在执行后复盘结果再迭代。",
            ],
        }
        return mapping.get(theme, mapping["general"])

    def _extract_evidence(self, *, query: str, selected: list[Any], limit: int) -> list[str]:
        query_terms = self._query_terms(query)
        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query))
        action_markers = ("需要", "建议", "应", "可", "先", "再", "避免", "评估", "分析", "优化")
        scored: list[tuple[float, str]] = []
        seen_sentences: set[str] = set()

        for rank, item in enumerate(selected, start=1):
            content = str(getattr(item, "content", ""))
            if not content:
                continue
            for sentence in self._split_sentences(content):
                if query_has_cjk and not re.search(r"[\u4e00-\u9fff]{2,}", sentence):
                    continue
                key = sentence.lower()
                if key in seen_sentences:
                    continue
                seen_sentences.add(key)
                hit_count = sum(1 for term in query_terms if term and term in sentence.lower())
                marker_bonus = 1 if any(marker in sentence for marker in action_markers) else 0
                source_bonus = max(0.0, 0.5 - 0.1 * (rank - 1))
                length_penalty = 0.2 if len(sentence) > 90 else 0.0
                score = hit_count + marker_bonus + source_bonus - length_penalty
                scored.append((score, sentence))

        scored.sort(key=lambda row: row[0], reverse=True)
        evidence = [sentence for _, sentence in scored[:limit]]
        return evidence

    def _split_sentences(self, text: str) -> list[str]:
        raw_parts = re.split(r"[。！？\n]+", text.replace("\r", "\n"))
        sentences: list[str] = []
        for part in raw_parts:
            line = part.strip()
            if not line:
                continue
            line = re.sub(r"[\t ]+", " ", line)
            if len(line) < 10:
                continue
            if line.startswith(("#", "|", "-", "```", "http")):
                continue
            if line.count("|") >= 2:
                continue
            if not self._is_readable_sentence(line):
                continue
            sentences.append(line)
        return sentences

    def _query_terms(self, query: str) -> list[str]:
        terms = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", query.lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped

    def _is_readable_sentence(self, line: str) -> bool:
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", line)
        if not cleaned:
            return False
        lowered = cleaned.lower()
        noise_markers = ("flatedecode", "xref", "obj", "endobj", "stream", "/filter", "/length")
        if any(marker in lowered for marker in noise_markers):
            return False
        readable = re.findall(r"[A-Za-z0-9\u4e00-\u9fff，。！？；：、（）()《》“”‘’\- ]", cleaned)
        ratio = len(readable) / max(len(cleaned), 1)
        if ratio < 0.75:
            return False
        has_cjk_word = bool(re.search(r"[\u4e00-\u9fff]{2,}", cleaned))
        if has_cjk_word:
            return True
        english_words = re.findall(r"[A-Za-z]{2,}", cleaned)
        return len(english_words) >= 3

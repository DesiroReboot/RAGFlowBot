from __future__ import annotations

from typing import Any

from src.core.search.source_utils import build_grouped_citations


class ContextSelector:
    def __init__(
        self,
        *,
        source_quota_mode: str = "balanced",
        max_chunks_per_source: int = 0,
    ) -> None:
        normalized_mode = str(source_quota_mode or "balanced").strip().lower()
        if normalized_mode not in {"balanced", "unbounded"}:
            normalized_mode = "balanced"
        self.source_quota_mode = normalized_mode
        self.max_chunks_per_source = max(0, int(max_chunks_per_source))
        self.last_source_quotas: dict[str, int] = {}

    def select(
        self,
        *,
        candidates: list[dict[str, Any]],
        source_scores: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if top_k <= 0:
            self.last_source_quotas = {}
            return [], []

        source_candidates: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            source = str(candidate.get("source", ""))
            source_candidates.setdefault(source, []).append(candidate)

        quotas = self._build_source_quotas(
            source_scores=source_scores,
            source_candidates=source_candidates,
            top_k=top_k,
        )
        self.last_source_quotas = quotas

        seen_content: set[str] = set()
        seen_keys: set[tuple[str, int]] = set()
        ranked_candidates = sorted(
            candidates,
            key=lambda row: float(row.get("score", 0.0)),
            reverse=True,
        )
        if self.source_quota_mode == "unbounded":
            selected = self._select_unbounded(
                ranked_candidates=ranked_candidates,
                seen_content=seen_content,
                seen_keys=seen_keys,
                top_k=top_k,
            )
            citations = build_grouped_citations(selected)
            return selected, citations

        per_source_count: dict[str, int] = {}
        selected: list[dict[str, Any]] = []
        for candidate in ranked_candidates:
            source = str(candidate.get("source", ""))
            if source not in quotas:
                continue
            if per_source_count.get(source, 0) >= quotas[source]:
                continue
            content_key = str(candidate.get("content", "")).strip()
            key = (str(candidate.get("file_uuid", "")), int(candidate.get("chunk_id", 0)))
            if content_key in seen_content or key in seen_keys:
                continue
            selected.append(candidate)
            per_source_count[source] = per_source_count.get(source, 0) + 1
            seen_content.add(content_key)
            seen_keys.add(key)
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            # First try a balanced fill to avoid a single source dominating tail slots.
            fallback_source_cap = max(1, min(2, self._max_per_source_cap(top_k)))
            for candidate in ranked_candidates:
                source = str(candidate.get("source", ""))
                if per_source_count.get(source, 0) >= fallback_source_cap:
                    continue
                content_key = str(candidate.get("content", "")).strip()
                key = (str(candidate.get("file_uuid", "")), int(candidate.get("chunk_id", 0)))
                if content_key in seen_content or key in seen_keys:
                    continue
                selected.append(candidate)
                per_source_count[source] = per_source_count.get(source, 0) + 1
                seen_content.add(content_key)
                seen_keys.add(key)
                if len(selected) >= top_k:
                    break

        if len(selected) < top_k:
            # When dedup and caps remove too many rows, fill remaining slots by score.
            for candidate in ranked_candidates:
                source = str(candidate.get("source", ""))
                content_key = str(candidate.get("content", "")).strip()
                key = (str(candidate.get("file_uuid", "")), int(candidate.get("chunk_id", 0)))
                if content_key in seen_content or key in seen_keys:
                    continue
                selected.append(candidate)
                per_source_count[source] = per_source_count.get(source, 0) + 1
                seen_content.add(content_key)
                seen_keys.add(key)
                if len(selected) >= top_k:
                    break

        citations = build_grouped_citations(selected)
        return selected, citations

    def _build_source_quotas(
        self,
        *,
        source_scores: list[dict[str, Any]],
        source_candidates: dict[str, list[dict[str, Any]]],
        top_k: int,
    ) -> dict[str, int]:
        if not source_candidates:
            return {}

        ranked_sources = [str(item.get("source", "")) for item in source_scores if item.get("source")]
        for source in source_candidates:
            if source not in ranked_sources:
                ranked_sources.append(source)

        if self.source_quota_mode == "unbounded":
            return {
                source: min(len(source_candidates.get(source, [])), top_k)
                for source in ranked_sources
                if len(source_candidates.get(source, [])) > 0
            }

        quotas: dict[str, int] = {}
        base_quota = 1
        for source in ranked_sources:
            available = len(source_candidates.get(source, []))
            if available <= 0:
                continue
            quotas[source] = min(base_quota, available)

        remaining = max(0, top_k - sum(quotas.values()))
        if remaining <= 0:
            return quotas

        source_weight: dict[str, float] = {}
        for item in source_scores:
            source = str(item.get("source", ""))
            if source not in quotas:
                continue
            source_weight[source] = max(0.0, float(item.get("score", 0.0))) + 0.6 * max(
                0.0,
                float(item.get("doc_evidence_mass_norm", 0.0)),
            )
        for source in quotas:
            source_weight.setdefault(source, 0.0)

        max_per_source = self._max_per_source_cap(top_k)
        extra_allocated: dict[str, int] = {source: 0 for source in quotas}
        while remaining > 0:
            eligible = [
                source
                for source, quota in quotas.items()
                if quota < min(max_per_source, len(source_candidates.get(source, [])))
            ]
            if not eligible:
                break
            pick = max(
                eligible,
                key=lambda source: source_weight.get(source, 0.0) / (1 + extra_allocated[source]),
            )
            quotas[pick] += 1
            extra_allocated[pick] += 1
            remaining -= 1
        return quotas

    def _max_per_source_cap(self, top_k: int) -> int:
        if self.max_chunks_per_source > 0:
            return max(1, min(self.max_chunks_per_source, top_k))
        return max(2, min(3, top_k))

    def _select_unbounded(
        self,
        *,
        ranked_candidates: list[dict[str, Any]],
        seen_content: set[str],
        seen_keys: set[tuple[str, int]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for candidate in ranked_candidates:
            content_key = str(candidate.get("content", "")).strip()
            key = (str(candidate.get("file_uuid", "")), int(candidate.get("chunk_id", 0)))
            if content_key in seen_content or key in seen_keys:
                continue
            selected.append(candidate)
            seen_content.add(content_key)
            seen_keys.add(key)
            if len(selected) >= top_k:
                break
        return selected

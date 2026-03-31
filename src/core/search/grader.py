from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import math
import re
from typing import Any

from src.core.search.source_utils import canonical_source_id


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


class ResultGrader:
    _NOISE_MARKERS = (
        "flatedecode",
        "xref",
        "endobj",
        "stream",
        "/filter",
        "/length",
        "obj",
    )
    _ACTIONABLE_MARKERS = (
        "需要",
        "应当",
        "建议",
        "步骤",
        "要求",
        "不得",
        "必须",
        "应该",
        "should",
        "must",
        "required",
    )
    _CONFLICT_RESTRICT_MARKERS = ("禁止", "限制", "下架", "封禁", "ban", "restriction", "penalty")
    _CONFLICT_RELAX_MARKERS = ("允许", "放宽", "恢复", "支持", "allow", "approved")
    _LOW_AUTHORITY_DOMAIN_MARKERS = ("forum", "bbs", "weibo", "zhihu", "reddit", "blog")
    _HIGH_AUTHORITY_DOMAIN_MARKERS = (
        ".gov",
        ".edu",
        ".org",
        "docs.",
        "official",
        "openai.com",
        "wikipedia.org",
    )
    _RELATION_QUERY_MARKERS = (
        "组成",
        "构成",
        "公式",
        "计算",
        "等于",
        "包含",
        "报价",
        "price composition",
        "formula",
        "compose",
    )
    _RELATION_CONNECTORS = ("=", "+", "组成", "构成", "包括", "由", "等于", "包含", "含")
    _INCOTERM_MARKERS = ("fob", "cfr", "cif", "exw")

    def __init__(
        self,
        *,
        min_evidence_score: float = 0.26,
        min_freshness_temporal: float = 0.32,
        conflict_pool_threshold: float = 0.82,
        qa_anchor_enabled: bool = True,
        semantic_guard_enabled: bool = True,
    ) -> None:
        self.min_evidence_score = _clamp(min_evidence_score)
        self.min_freshness_temporal = _clamp(min_freshness_temporal)
        self.conflict_pool_threshold = _clamp(conflict_pool_threshold)
        self.qa_anchor_enabled = bool(qa_anchor_enabled)
        self.semantic_guard_enabled = bool(semantic_guard_enabled)
        self.last_hard_filtered: list[dict[str, Any]] = []
        self.last_conflict_pool: list[dict[str, Any]] = []

    def grade(
        self,
        *,
        query_tokens: list[str],
        query_theme_hints: list[str] | None = None,
        fused_results: list[dict[str, Any]],
        query_intent: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.last_hard_filtered = []
        self.last_conflict_pool = []
        if not fused_results:
            return [], []

        temporal_query = self._is_temporal_query(query_tokens=query_tokens, query_intent=query_intent)
        max_rrf = max((float(item.get("rrf_score", 0.0)) for item in fused_results), default=1.0)
        max_similarity = max(
            (max(0.0, float(item.get("vec_similarity", 0.0))) for item in fused_results),
            default=1.0,
        )
        max_lexical_rank = max((int(item.get("fts_rank", 0)) for item in fused_results), default=1)
        content_hash_to_top_score: dict[str, float] = {}

        pre_scored: list[dict[str, Any]] = []
        theme_hints = [hint for hint in (query_theme_hints or []) if hint]
        for item in fused_results:
            content = str(item.get("content", ""))
            content_lower = content.lower()
            overlap = 0.0
            if query_tokens:
                overlap = sum(1 for token in query_tokens if token in content_lower) / len(query_tokens)

            lexical_norm = 0.0
            if item.get("fts_rank"):
                lexical_norm = 1.0 - ((int(item["fts_rank"]) - 1) / max(max_lexical_rank, 1))
            semantic_norm = max(0.0, float(item.get("vec_similarity", 0.0))) / max(
                max_similarity,
                1e-9,
            )
            rrf_norm = float(item.get("rrf_score", 0.0)) / max(max_rrf, 1e-9)

            source = str(item.get("source", "")).lower()
            section_title = str(item.get("section_title", "")).lower()
            metadata_boost = min(
                1.0,
                sum(0.5 for token in query_tokens if token in source or token in section_title),
            )

            readability_score = self._readability_score(content)
            relevance_score = _clamp(
                0.42 * overlap + 0.24 * lexical_norm + 0.24 * semantic_norm + 0.10 * metadata_boost
            )
            evidence_score = self._evidence_score(
                content=content,
                readability_score=readability_score,
                overlap_score=overlap,
            )
            freshness_score = self._freshness_score(item)
            source_key = str(item.get("source", ""))
            source_path = str(item.get("source_path", ""))
            doc_type = str(item.get("doc_type", "text")).lower()
            authority_score = self._authority_score(
                source=source_key,
                source_path=source_path,
                section_title=str(item.get("section_title", "")),
            )

            short_penalty = 0.15 if len(content.strip()) < 40 else 0.0
            pdf_noise_penalty = 0.25 * (1.0 - readability_score) if doc_type == "pdf" else 0.0
            general_noise_penalty = 0.08 * (1.0 - readability_score)
            noise_penalty = short_penalty + pdf_noise_penalty + general_noise_penalty

            source_theme_boost = self._source_theme_boost(
                source=source_key,
                source_path=source_path,
                section_title=str(item.get("section_title", "")),
                doc_type=doc_type,
                query_theme_hints=theme_hints,
            )
            formula_boost = self._formula_structure_boost(
                content=content,
                query_tokens=query_tokens,
            )
            qa_anchor_boost = (
                self._qa_anchor_boost(content=content, query_tokens=query_tokens)
                if self.qa_anchor_enabled
                else 0.0
            )
            semantic_guard_penalty = (
                self._semantic_guard_penalty(content=content, query_tokens=query_tokens)
                if self.semantic_guard_enabled
                else 0.0
            )
            content_hash = str(item.get("content_hash") or hash(content))
            prior_score = content_hash_to_top_score.get(content_hash, 0.0)
            redundancy_penalty = 0.15 if prior_score > 0.75 else 0.0

            candidate_score = (
                0.22 * rrf_norm
                + 0.15 * lexical_norm
                + 0.17 * semantic_norm
                + 0.12 * overlap
                + 0.08 * metadata_boost
                + 0.10 * relevance_score
                + 0.12 * evidence_score
                + 0.08 * freshness_score
                + 0.06 * authority_score
                + source_theme_boost
                + formula_boost
                + qa_anchor_boost
                - redundancy_penalty
                - noise_penalty
                - semantic_guard_penalty
            )
            stance = self._conflict_stance(content_lower)
            pre_scored.append(
                {
                    **item,
                    "source": source_key,
                    "source_path": source_path,
                    "canonical_source_id": canonical_source_id(source_key, source_path),
                    "_candidate_score": candidate_score,
                    "_readability_score": readability_score,
                    "_noise_penalty": noise_penalty,
                    "_redundancy_penalty": redundancy_penalty,
                    "_overlap": overlap,
                    "_lexical_norm": lexical_norm,
                    "_semantic_norm": semantic_norm,
                    "_rrf_norm": rrf_norm,
                    "_metadata_boost": metadata_boost,
                    "_source_theme_boost": source_theme_boost,
                    "_formula_boost": formula_boost,
                    "_qa_anchor_boost": qa_anchor_boost,
                    "_semantic_guard_penalty": semantic_guard_penalty,
                    "_relevance_score": relevance_score,
                    "_evidence_score": evidence_score,
                    "_freshness_score": freshness_score,
                    "_authority_score": authority_score,
                    "_conflict_stance": stance,
                    "_conflict_risk": 0.0,
                }
            )
            content_hash_to_top_score[content_hash] = max(prior_score, candidate_score)

        has_restrict = any(int(item.get("_conflict_stance", 0)) < 0 for item in pre_scored)
        has_relax = any(int(item.get("_conflict_stance", 0)) > 0 for item in pre_scored)
        has_conflict = has_restrict and has_relax
        filtered_pre_scored: list[dict[str, Any]] = []
        for item in pre_scored:
            authority_score = float(item.get("_authority_score", 0.0))
            stance = int(item.get("_conflict_stance", 0))
            if has_conflict and stance != 0:
                conflict_risk = _clamp(0.78 + 0.16 * (1.0 - authority_score))
            else:
                conflict_risk = _clamp(0.12 + 0.18 * (1.0 - authority_score))
            item["_conflict_risk"] = conflict_risk
            item["_candidate_score"] = float(item["_candidate_score"]) - 0.10 * conflict_risk

            evidence_score = float(item.get("_evidence_score", 0.0))
            freshness_score = float(item.get("_freshness_score", 0.0))
            reason = ""
            if evidence_score < self.min_evidence_score:
                reason = "evidence_below_threshold"
            elif temporal_query and freshness_score < self.min_freshness_temporal:
                reason = "freshness_below_threshold_for_temporal_query"
            elif conflict_risk >= self.conflict_pool_threshold:
                reason = "high_conflict_risk_pool"

            if not reason:
                filtered_pre_scored.append(item)
                continue

            row = {
                "file_uuid": str(item.get("file_uuid", "")),
                "chunk_id": int(item.get("chunk_id", 0)),
                "source": str(item.get("source", "")),
                "reason": reason,
                "evidence_score": round(evidence_score, 6),
                "freshness_score": round(freshness_score, 6),
                "authority_score": round(authority_score, 6),
                "conflict_risk": round(conflict_risk, 6),
            }
            if reason == "high_conflict_risk_pool":
                self.last_conflict_pool.append(row)
            else:
                self.last_hard_filtered.append(row)

        if not filtered_pre_scored:
            return [], []

        source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in filtered_pre_scored:
            source_groups[str(item.get("source", ""))].append(item)

        doc_evidence_mass_by_source: dict[str, float] = {}
        doc_size_prior_by_source: dict[str, float] = {}
        doc_quality_prior_by_source: dict[str, float] = {}
        for source, items in source_groups.items():
            sorted_items = sorted(items, key=lambda row: float(row["_candidate_score"]), reverse=True)
            doc_evidence_mass = sum(
                max(0.0, float(row["_candidate_score"])) for row in sorted_items[:3]
            )
            doc_chunk_count = max(
                int(row.get("doc_chunk_count", 0) or 0) for row in sorted_items
            )
            if doc_chunk_count <= 0:
                doc_chunk_count = len(sorted_items)
            doc_size_prior = math.log1p(max(1, doc_chunk_count))
            doc_quality_prior = sum(
                float(row.get("_readability_score", 0.0)) for row in sorted_items[:5]
            ) / max(min(len(sorted_items), 5), 1)
            doc_evidence_mass_by_source[source] = doc_evidence_mass
            doc_size_prior_by_source[source] = doc_size_prior
            doc_quality_prior_by_source[source] = max(0.05, doc_quality_prior)

        evidence_mass_norm = self._minmax_normalize(doc_evidence_mass_by_source)
        size_quality = {
            source: doc_size_prior_by_source[source] * doc_quality_prior_by_source[source]
            for source in source_groups
        }
        size_quality_norm = self._minmax_normalize(size_quality)

        candidate_results: list[dict[str, Any]] = []
        for item in filtered_pre_scored:
            source = str(item.get("source", ""))
            candidate_score = float(item["_candidate_score"])
            final_score = (
                0.58 * candidate_score
                + 0.22 * evidence_mass_norm.get(source, 0.0)
                + 0.10 * size_quality_norm.get(source, 0.0)
                + 0.10 * float(item.get("_authority_score", 0.0))
            )
            graded = {
                **item,
                "grading": {
                    "rrf_score": round(float(item.get("rrf_score", 0.0)), 6),
                    "lexical_score": round(float(item["_lexical_norm"]), 6),
                    "semantic_score": round(float(item["_semantic_norm"]), 6),
                    "overlap_score": round(float(item["_overlap"]), 6),
                    "metadata_boost": round(float(item["_metadata_boost"]), 6),
                    "source_theme_boost": round(float(item["_source_theme_boost"]), 6),
                    "formula_boost": round(float(item["_formula_boost"]), 6),
                    "qa_anchor_boost": round(float(item["_qa_anchor_boost"]), 6),
                    "semantic_guard_penalty": round(float(item["_semantic_guard_penalty"]), 6),
                    "readability_score": round(float(item["_readability_score"]), 6),
                    "relevance_score": round(float(item["_relevance_score"]), 6),
                    "evidence_score": round(float(item["_evidence_score"]), 6),
                    "freshness_score": round(float(item["_freshness_score"]), 6),
                    "authority_score": round(float(item["_authority_score"]), 6),
                    "conflict_risk": round(float(item["_conflict_risk"]), 6),
                    "redundancy_penalty": round(float(item["_redundancy_penalty"]), 6),
                    "noise_penalty": round(float(item["_noise_penalty"]), 6),
                    "candidate_score": round(candidate_score, 6),
                    "doc_evidence_mass_norm": round(evidence_mass_norm.get(source, 0.0), 6),
                    "doc_size_quality_norm": round(size_quality_norm.get(source, 0.0), 6),
                    "final_score": round(final_score, 6),
                    "hard_filter_passed": True,
                },
                "score": final_score,
            }
            for key in (
                "_candidate_score",
                "_readability_score",
                "_noise_penalty",
                "_redundancy_penalty",
                "_overlap",
                "_lexical_norm",
                "_semantic_norm",
                "_rrf_norm",
                "_metadata_boost",
                "_source_theme_boost",
                "_formula_boost",
                "_qa_anchor_boost",
                "_semantic_guard_penalty",
                "_relevance_score",
                "_evidence_score",
                "_freshness_score",
                "_authority_score",
                "_conflict_stance",
                "_conflict_risk",
            ):
                graded.pop(key, None)
            candidate_results.append(graded)

        candidate_results.sort(key=lambda item: float(item["score"]), reverse=True)
        source_results: list[dict[str, Any]] = []
        for source, items in source_groups.items():
            ranked_items = sorted(
                (row for row in candidate_results if str(row.get("source", "")) == source),
                key=lambda row: float(row["score"]),
                reverse=True,
            )
            if not ranked_items:
                continue
            top_score = float(ranked_items[0]["score"])
            second_score = float(ranked_items[1]["score"]) if len(ranked_items) > 1 else 0.0
            coverage = min(1.0, len(items) / 3.0)
            citation_readiness = 1.0 if source and ranked_items[0].get("source_path") else 0.7
            authority_mean = sum(
                float(row.get("grading", {}).get("authority_score", 0.0)) for row in ranked_items[:3]
            ) / max(min(len(ranked_items), 3), 1)
            source_score = (
                0.32 * top_score
                + 0.18 * second_score
                + 0.12 * coverage
                + 0.1 * citation_readiness
                + 0.16 * evidence_mass_norm.get(source, 0.0)
                + 0.12 * authority_mean
            )
            source_results.append(
                {
                    "source": source,
                    "source_path": ranked_items[0].get("source_path", ""),
                    "canonical_source_id": ranked_items[0].get("canonical_source_id", ""),
                    "score": round(source_score, 6),
                    "coverage": round(coverage, 6),
                    "citation_readiness": round(citation_readiness, 6),
                    "top_chunk_id": ranked_items[0]["chunk_id"],
                    "chunk_count": len(items),
                    "doc_evidence_mass": round(doc_evidence_mass_by_source.get(source, 0.0), 6),
                    "doc_evidence_mass_norm": round(evidence_mass_norm.get(source, 0.0), 6),
                    "doc_size_prior": round(doc_size_prior_by_source.get(source, 0.0), 6),
                    "doc_quality_prior": round(doc_quality_prior_by_source.get(source, 0.0), 6),
                    "doc_size_quality_prior_norm": round(size_quality_norm.get(source, 0.0), 6),
                }
            )

        source_results.sort(key=lambda item: float(item["score"]), reverse=True)
        source_scores = {item["source"]: item["score"] for item in source_results}
        for item in candidate_results:
            item["grading"]["source_score"] = round(source_scores.get(item.get("source", ""), 0.0), 6)
        return candidate_results, source_results

    def _minmax_normalize(self, values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        minimum = min(values.values())
        maximum = max(values.values())
        if maximum - minimum <= 1e-9:
            return {key: 1.0 for key in values}
        return {
            key: (value - minimum) / (maximum - minimum)
            for key, value in values.items()
        }

    def _readability_score(self, content: str) -> float:
        text = str(content or "")
        if not text.strip():
            return 0.0
        lowered = text.lower()
        noise_hits = sum(lowered.count(marker) for marker in self._NOISE_MARKERS)
        readable_chars = re.findall(
            r"[A-Za-z0-9\u4e00-\u9fff，。！？；：、“”‘’（）()、,.!?;:\-_/ ]",
            text,
        )
        readability_ratio = len(readable_chars) / max(len(text), 1)
        noise_ratio = min(1.0, noise_hits / max(len(text) / 80.0, 1.0))
        score = 0.75 * readability_ratio + 0.25 * (1.0 - noise_ratio)
        return _clamp(score)

    def _evidence_score(self, *, content: str, readability_score: float, overlap_score: float) -> float:
        text = str(content or "").strip()
        if not text:
            return 0.0
        length_score = min(1.0, len(text) / 240.0)
        token_pool = re.findall(r"[a-z0-9\u4e00-\u9fff]", text.lower())
        unique_density = len(set(token_pool)) / max(len(token_pool), 1)
        actionable_hits = sum(1 for marker in self._ACTIONABLE_MARKERS if marker in text.lower())
        actionable_score = min(1.0, actionable_hits / 3.0)
        date_anchor = 1.0 if re.search(r"20\d{2}[-/年](0?[1-9]|1[0-2])", text) else 0.0
        return _clamp(
            0.30 * readability_score
            + 0.20 * length_score
            + 0.20 * unique_density
            + 0.20 * actionable_score
            + 0.05 * overlap_score
            + 0.05 * date_anchor
        )

    def _freshness_score(self, item: dict[str, Any]) -> float:
        candidates = [
            str(item.get("published_at", "")).strip(),
            str(item.get("updated_at", "")).strip(),
            str(item.get("metadata", {}).get("published_at", "")).strip()
            if isinstance(item.get("metadata"), dict)
            else "",
            str(item.get("source_path", "")).strip(),
            str(item.get("source", "")).strip(),
        ]
        dates: list[date] = []
        for text in candidates:
            if not text:
                continue
            parsed = self._extract_date(text)
            if parsed is not None:
                dates.append(parsed)
        if not dates:
            return 0.55

        newest = max(dates)
        age_days = (datetime.now(timezone.utc).date() - newest).days
        if age_days <= 30:
            return 1.0
        if age_days <= 90:
            return 0.82
        if age_days <= 180:
            return 0.66
        if age_days <= 365:
            return 0.48
        return 0.26

    def _extract_date(self, text: str) -> date | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        for candidate in re.findall(r"20\d{2}[-/年](?:0?[1-9]|1[0-2])[-/月](?:0?[1-9]|[12]\d|3[01])", raw):
            normalized = candidate.replace("年", "-").replace("月", "-").replace("日", "")
            normalized = normalized.replace("/", "-")
            try:
                return date.fromisoformat(normalized)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except Exception:
            pass
        if len(raw) >= 10:
            try:
                return date.fromisoformat(raw[:10].replace("/", "-"))
            except Exception:
                return None
        return None

    def _authority_score(self, *, source: str, source_path: str, section_title: str) -> float:
        joined = " ".join([str(source).lower(), str(source_path).lower(), str(section_title).lower()]).strip()
        if not joined:
            return 0.3
        if any(marker in joined for marker in self._HIGH_AUTHORITY_DOMAIN_MARKERS):
            return 0.88
        if any(marker in joined for marker in self._LOW_AUTHORITY_DOMAIN_MARKERS):
            return 0.45
        if source_path.startswith(("http://", "https://")):
            return 0.72
        if source_path:
            return 0.68
        return 0.58

    def _conflict_stance(self, lowered_content: str) -> int:
        has_restrict = any(marker in lowered_content for marker in self._CONFLICT_RESTRICT_MARKERS)
        has_relax = any(marker in lowered_content for marker in self._CONFLICT_RELAX_MARKERS)
        if has_restrict and not has_relax:
            return -1
        if has_relax and not has_restrict:
            return 1
        return 0

    def _is_temporal_query(
        self,
        *,
        query_tokens: list[str],
        query_intent: dict[str, Any] | None,
    ) -> bool:
        if isinstance(query_intent, dict):
            temporal_terms = query_intent.get("temporal_terms", [])
            if isinstance(temporal_terms, list) and any(str(item).strip() for item in temporal_terms):
                return True
        lowered_tokens = [str(token).lower() for token in query_tokens]
        markers = {"最新", "最近", "近期", "本周", "本月", "今年", "latest", "recent", "today"}
        return any(marker in token for token in lowered_tokens for marker in markers)

    def _source_theme_boost(
        self,
        *,
        source: str,
        source_path: str,
        section_title: str,
        doc_type: str,
        query_theme_hints: list[str],
    ) -> float:
        if not query_theme_hints:
            return 0.0
        source_text = " ".join(
            [
                str(source).lower(),
                str(source_path).lower(),
                str(section_title).lower(),
                canonical_source_id(source, source_path),
            ]
        )
        if not source_text.strip():
            return 0.0

        theme_aliases: dict[str, tuple[str, ...]] = {
            "报关物流": ("报关", "清关", "物流", "运输", "fba", "海外仓"),
            "报价合同": ("报价", "合同", "条款", "询盘", "成交"),
            "生产备货": ("生产", "备货", "补货", "库存", "断货"),
            "收汇退税": ("收汇", "结汇", "退税", "税"),
            "客户开发": ("客户", "开发", "广告", "acos", "转化", "关键词", "标题"),
        }
        boost = 0.0
        for hint in query_theme_hints:
            aliases = theme_aliases.get(hint, (hint.lower(),))
            if any(alias and alias in source_text for alias in aliases):
                boost += 0.08
        if boost > 0 and doc_type == "pdf":
            boost += 0.04
        return min(0.28, boost)

    def _formula_structure_boost(self, *, content: str, query_tokens: list[str]) -> float:
        lowered_content = str(content or "").lower()
        if not lowered_content:
            return 0.0

        formula_intent = any(
            ("公式" in str(token))
            or ("组成" in str(token))
            or ("计算" in str(token))
            or ("报价" in str(token))
            for token in query_tokens
        )
        if not formula_intent:
            return 0.0
        if "=" not in content or "+" not in content:
            return 0.0

        domestic_cost_markers = (
            "国内运费",
            "报关费",
            "装船费",
            "港口杂费",
        )
        domestic_hits = sum(1 for marker in domestic_cost_markers if marker in content)
        # Strongly prioritize canonical FOB composition formulas when users ask "组成/公式/计算".
        if "fob" in lowered_content and "exw" in lowered_content:
            if domestic_hits >= 3:
                return 0.36
            if domestic_hits >= 2:
                return 0.28

        component_markers = (
            "fob",
            "exw",
            "国内运费",
            "报关费",
            "装船费",
            "港口杂费",
            "国际海运费",
            "海运保险费",
        )
        hits = sum(1 for marker in component_markers if marker in lowered_content or marker in content)
        if hits >= 6:
            return 0.18
        if hits >= 4:
            return 0.12
        return 0.0

    def _has_relation_intent(self, query_tokens: list[str]) -> bool:
        if not query_tokens:
            return False
        lowered_tokens = [str(token).lower() for token in query_tokens]
        return any(
            any(marker in token for marker in self._RELATION_QUERY_MARKERS)
            for token in lowered_tokens
        )

    def _qa_anchor_boost(self, *, content: str, query_tokens: list[str]) -> float:
        if not self._has_relation_intent(query_tokens):
            return 0.0
        text = str(content or "")
        lowered = text.lower()
        if not lowered:
            return 0.0

        has_equation = bool(re.search(r"(fob|cfr|cif|exw)\s*[:=：]", lowered))
        has_relation_connector = any(marker in text or marker in lowered for marker in self._RELATION_CONNECTORS)

        boost = 0.0
        if has_equation:
            boost += 0.08
        if has_relation_connector:
            boost += 0.06

        if ("fob" in lowered and "exw" in lowered) and (
            ("国内运费" in text)
            or ("报关费" in text)
            or ("装船费" in text)
            or ("港口杂费" in text)
            or ("inland freight" in lowered)
            or ("customs fee" in lowered)
        ):
            boost += 0.20

        if ("cfr" in lowered and "fob" in lowered) and (
            ("国际海运费" in text) or ("sea freight" in lowered)
        ):
            boost += 0.16

        if ("cif" in lowered and ("fob" in lowered or "cfr" in lowered)) and (
            ("海运保险费" in text) or ("insurance" in lowered)
        ):
            boost += 0.14

        return min(0.34, boost)

    def _semantic_guard_penalty(self, *, content: str, query_tokens: list[str]) -> float:
        if not self._has_relation_intent(query_tokens):
            return 0.0
        text = str(content or "")
        lowered = text.lower()
        if not lowered:
            return 0.0

        query_incoterms = {token.lower() for token in query_tokens if token.lower() in self._INCOTERM_MARKERS}
        content_incoterms = {term for term in self._INCOTERM_MARKERS if term in lowered}
        has_relation_connector = any(marker in text or marker in lowered for marker in self._RELATION_CONNECTORS)

        if has_relation_connector and content_incoterms:
            return 0.0

        hit_query_terms = sum(1 for token in query_tokens if token and str(token).lower() in lowered)
        mentions_pricing_words = any(
            marker in lowered
            for marker in ("price", "quote", "报价", "价格", "费用", "成本", "fob", "cfr", "cif", "exw")
        )
        miss_target_relation = bool(query_incoterms & content_incoterms) or (
            bool(query_incoterms) and not content_incoterms
        )

        penalty = 0.0
        if mentions_pricing_words and not has_relation_connector:
            penalty += 0.08
        if hit_query_terms >= 2 and not has_relation_connector:
            penalty += 0.06
        if miss_target_relation and not has_relation_connector:
            penalty += 0.08
        return min(0.24, penalty)

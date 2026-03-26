from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class DomainFilterResult:
    allow_rag: bool
    reason: str
    score: float
    threshold: float
    positive_hits: list[str] = field(default_factory=list)
    action_hits: list[str] = field(default_factory=list)
    negative_hits: list[str] = field(default_factory=list)
    negative_intent: bool = False

    def to_trace_dict(self) -> dict[str, Any]:
        decision = "allow" if self.allow_rag else "block"
        return {
            "score": round(float(self.score), 4),
            "threshold": round(float(self.threshold), 4),
            "positive_hits": list(self.positive_hits),
            "action_hits": list(self.action_hits),
            "negative_hits": list(self.negative_hits),
            "negative_intent": bool(self.negative_intent),
            "decision": decision,
            "reason": self.reason,
        }


class DomainFilter:
    _POSITIVE_KEYWORDS = (
        "外贸",
        "跨境",
        "电商",
        "跨境电商",
        "亚马逊",
        "amazon",
        "shopify",
        "速卖通",
        "aliexpress",
        "ebay",
        "temu",
        "shein",
        "独立站",
        "listing",
        "sku",
        "fba",
        "fbm",
        "fulfillment",
        "选品",
        "铺货",
        "测款",
        "关键词",
        "广告",
        "ppc",
        "acos",
        "roas",
        "物流",
        "货代",
        "报关",
        "清关",
        "海关",
        "customs",
        "关税",
        "tariff",
        "供应链",
        "合规",
        "侵权",
    )
    _ACTION_KEYWORDS = (
        "如何",
        "怎么",
        "方案",
        "策略",
        "优化",
        "提升",
        "降低",
        "排查",
        "诊断",
        "预算",
        "定价",
        "备货",
        "补货",
        "引流",
        "投放",
        "运营",
        "复盘",
        "分析",
        "执行",
        "计划",
        "流程",
        "strategy",
        "optimize",
        "improve",
        "reduce",
        "playbook",
    )
    _NEGATIVE_KEYWORDS = (
        "电影",
        "电视剧",
        "小说",
        "音乐",
        "歌词",
        "星座",
        "占卜",
        "游戏",
        "王者荣耀",
        "英雄联盟",
        "考研",
        "高数",
        "物理",
        "化学",
        "历史",
        "旅游",
        "景点",
        "天气",
        "菜谱",
        "减肥",
        "健身",
        "python",
        "java",
        "javascript",
        "golang",
        "c++",
        "leetcode",
        "算法",
        "前端",
        "后端",
        "数据库",
        "mysql",
        "linux",
        "docker",
        "k8s",
        "kubernetes",
        "machine learning",
        "deep learning",
        "nba",
        "足球",
        "世界杯",
        "股票",
        "基金",
        "比特币",
    )
    _STRONG_NEGATIVE_KEYWORDS = (
        "python",
        "java",
        "javascript",
        "leetcode",
        "算法",
        "前端",
        "后端",
        "数据库",
        "docker",
        "kubernetes",
        "电影",
        "歌词",
        "菜谱",
        "nba",
        "足球",
    )

    def __init__(self, *, threshold: float = 0.45) -> None:
        self.threshold = max(0.0, min(1.0, float(threshold)))

    def check(self, query: str) -> DomainFilterResult:
        normalized = re.sub(r"\s+", " ", str(query or "")).strip().lower()
        if not normalized:
            return DomainFilterResult(
                allow_rag=True,
                reason="empty_query",
                score=1.0,
                threshold=self.threshold,
            )

        compact = re.sub(r"\s+", "", normalized)
        if len(compact) <= 2:
            return DomainFilterResult(
                allow_rag=True,
                reason="short_query_fail_open",
                score=max(self.threshold, 0.5),
                threshold=self.threshold,
            )

        hits = self._hit_keywords(normalized)
        score = self._score_with_hits(hits)
        negative_intent = self._detect_negative_intent(normalized)

        positive_hits = list(hits["positive_hits"])
        action_hits = list(hits["action_hits"])
        negative_hits = list(hits["negative_hits"])

        allow_rag = score >= self.threshold
        reason = "score_above_threshold" if allow_rag else "score_below_threshold"

        if negative_intent and not positive_hits:
            allow_rag = False
            reason = "negative_intent_without_domain_signal"
        elif not allow_rag and len(positive_hits) >= 2:
            # Guardrail to reduce false blocks on domain-specific noun phrases.
            allow_rag = True
            reason = "strong_domain_signal"

        return DomainFilterResult(
            allow_rag=allow_rag,
            reason=reason,
            score=score,
            threshold=self.threshold,
            positive_hits=positive_hits,
            action_hits=action_hits,
            negative_hits=negative_hits,
            negative_intent=negative_intent,
        )

    def _score_with_hits(self, hits: dict[str, list[str]]) -> float:
        positive_count = len(hits.get("positive_hits", []))
        action_count = len(hits.get("action_hits", []))
        negative_count = len(hits.get("negative_hits", []))

        positive_score = min(0.72, 0.24 * positive_count)
        action_score = min(0.24, 0.08 * action_count)
        negative_penalty = min(0.72, 0.18 * negative_count)

        score = positive_score + action_score - negative_penalty
        if positive_count > 0 and action_count > 0:
            score += 0.08
        if positive_count == 0 and negative_count > 0:
            score -= 0.1
        return max(0.0, min(1.0, round(score, 4)))

    def _score_domain_relevance(self, query: str) -> float:
        normalized = re.sub(r"\s+", " ", str(query or "")).strip().lower()
        if not normalized:
            return 1.0
        hits = self._hit_keywords(normalized)
        return self._score_with_hits(hits)

    def _hit_keywords(self, query: str) -> dict[str, list[str]]:
        lowered = str(query or "").lower()
        return {
            "positive_hits": self._collect_hits(lowered, self._POSITIVE_KEYWORDS),
            "action_hits": self._collect_hits(lowered, self._ACTION_KEYWORDS),
            "negative_hits": self._collect_hits(lowered, self._NEGATIVE_KEYWORDS),
        }

    def _detect_negative_intent(self, query: str) -> bool:
        lowered = str(query or "").lower()
        if not lowered:
            return False
        strong_hits = self._collect_hits(lowered, self._STRONG_NEGATIVE_KEYWORDS)
        if not strong_hits:
            return False
        has_positive = any(token in lowered for token in self._POSITIVE_KEYWORDS)
        return not has_positive

    @staticmethod
    def _collect_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
        hits: list[str] = []
        for keyword in keywords:
            if keyword in text:
                hits.append(keyword)
        return hits

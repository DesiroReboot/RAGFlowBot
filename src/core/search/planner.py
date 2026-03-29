from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Literal

SourceRoute = Literal["kb_only", "web_only", "hybrid"]
FusionStrategy = Literal["none", "direct_fusion", "rag_fusion"]
RouteMode = Literal["none", "serial", "parallel"]


@dataclass
class PlannerOutput:
    plan_id: str
    need_web_search: bool
    source_route: SourceRoute
    fusion_strategy: FusionStrategy
    domain_relevance_score: float = 1.0
    confidence: float = 0.0
    route_mode: RouteMode = "none"
    reasons: list[str] = field(default_factory=list)
    query_expansion: dict[str, Any] = field(default_factory=dict)
    retrieval_plan: dict[str, Any] = field(default_factory=dict)


class Planner:
    """Planner interface for search routing decisions."""

    def plan(self, query: str, *, trace_context: dict[str, Any] | None = None) -> PlannerOutput:
        raise NotImplementedError


class RulePlanner(Planner):
    """Default lightweight planner without domain filter blocking."""

    def plan(self, query: str, *, trace_context: dict[str, Any] | None = None) -> PlannerOutput:
        query_text = str(query or "").strip()
        analyzer_view = self._extract_analyzer_view(trace_context)

        need_web_search = bool(analyzer_view["need_web_search"])
        source_route: SourceRoute = "hybrid" if need_web_search else "kb_only"
        route_mode: RouteMode = "serial" if need_web_search else "none"
        # Planner only keeps two-state routing (must/no). Concrete web fusion strategy
        # is decided later in orchestrator by web evaluation.
        fusion_strategy: FusionStrategy = "none"
        reasons = list(analyzer_view["reason_codes"]) or ["analyzer_default_no_web"]
        confidence = 0.78 if need_web_search else 0.9

        if not query_text:
            reasons = ["empty_query"]
            source_route = "kb_only"
            route_mode = "none"
            fusion_strategy = "none"
            need_web_search = False
            confidence = 0.0

        reasons.append("rag_all_queries")

        return PlannerOutput(
            plan_id=self._build_plan_id(query_text=query_text),
            need_web_search=need_web_search,
            source_route=source_route,
            fusion_strategy=fusion_strategy,
            domain_relevance_score=float(analyzer_view["domain_relevance_score"]),
            confidence=confidence,
            route_mode=route_mode,
            reasons=reasons,
            query_expansion={"core_terms": [], "bridge_terms": [], "synonyms": []},
            retrieval_plan={
                "sources": [
                    {"name": "kb_index", "enabled": True, "priority": 1},
                    {
                        "name": "web_search",
                        "enabled": need_web_search,
                        "priority": 2,
                    },
                ],
                "execution_mode": "serial" if need_web_search else "rag_only",
                "fallback_policy": "phase_a_rag_to_web_on_low_confidence",
            },
        )

    def _extract_analyzer_view(self, trace_context: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(trace_context, dict):
            return {"need_web_search": False, "reason_codes": [], "domain_relevance_score": 1.0}

        raw = trace_context.get("query_analysis")
        if not isinstance(raw, dict):
            return {"need_web_search": False, "reason_codes": [], "domain_relevance_score": 1.0}

        reason_codes = raw.get("reason_codes")
        if not isinstance(reason_codes, list):
            reason_codes = raw.get("reasons")
        normalized_reasons = [
            str(item).strip()
            for item in (reason_codes if isinstance(reason_codes, list) else [])
            if str(item).strip()
        ]
        return {
            "need_web_search": bool(raw.get("need_web_search", False)),
            "reason_codes": normalized_reasons,
            "domain_relevance_score": float(raw.get("domain_relevance_score", 1.0)),
        }

    def _build_plan_id(self, *, query_text: str) -> str:
        if not query_text:
            return "plan-empty"
        digest = hashlib.md5(query_text.encode("utf-8")).hexdigest()[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"plan-{stamp}-{digest}"

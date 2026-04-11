from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
import html
import json
import logging
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config  # noqa: E402
from src.core.bot_agent import ReActAgent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


@dataclass
class GoldenItem:
    id: str
    question: str
    scenario: str = "unknown"
    difficulty: str = "medium"
    source_of_question: str = "synthetic"
    expected_keywords: Dict[str, List[str]] = field(default_factory=lambda: {"must": [], "should": [], "optional": []})
    expected_sources: Dict[str, Any] = field(default_factory=lambda: {"must": [], "should": [], "equivalent_sources": [], "source_aliases": {}})
    rubric: Dict[str, Any] = field(default_factory=dict)
    forbidden_claims: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalResult:
    id: str
    question: str
    scenario: str
    difficulty: str
    source_of_question: str
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    must_source_recall: float = 0.0
    source_precision: float = 0.0
    source_precision_applicable: bool = True
    must_keyword_coverage: float = 0.0
    should_keyword_coverage: float = 0.0
    keyword_score: float = 0.0
    strict_citation_hit: bool = False
    relaxed_citation_hit: bool = False
    claim_supported_rate: float = 0.0
    claim_citation_precision: float = 0.0
    hallucination_rate: float = 0.0
    answer_completeness: float = 0.0
    instruction_following_rate: float = 0.0
    actionability_score: float = 0.0
    actionability_applicable: bool = True
    generation_quality_score: float = 0.0
    pass_rate_at_n: float = 1.0
    pass_count: int = 1
    repeat_runs: int = 1
    retrieved_sources: List[str] = field(default_factory=list)
    matched_must_keywords: List[str] = field(default_factory=list)
    matched_should_keywords: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    latency: float = 0.0
    failure_path: Dict[str, Any] = field(default_factory=dict)
    trace_file: str = ""
    html_file: str = ""
    diff_file: str = ""


@dataclass
class EvalSummary:
    total_items: int = 0
    must_source_recall_avg: float = 0.0
    source_precision_avg: float = 0.0
    source_precision_applicable_items: int = 0
    must_keyword_coverage_avg: float = 0.0
    should_keyword_coverage_avg: float = 0.0
    keyword_score_avg: float = 0.0
    strict_citation_rate: float = 0.0
    relaxed_citation_rate: float = 0.0
    claim_supported_rate_avg: float = 0.0
    claim_citation_precision_avg: float = 0.0
    hallucination_rate_avg: float = 0.0
    answer_completeness_avg: float = 0.0
    instruction_following_rate_avg: float = 0.0
    actionability_score_avg: float = 0.0
    actionability_applicable_items: int = 0
    generation_quality_score_avg: float = 0.0
    pass_rate_at_n_avg: float = 1.0
    repeat_runs: int = 1
    overall_score: float = 0.0
    overall_score_v2: float = 0.0
    grade: str = "D"
    latency_avg: float = 0.0
    scenario_buckets: Dict[str, Dict[str, float]] = field(default_factory=dict)


class EvalChecker:
    TARGET_MUST_SOURCE_RECALL = 0.7
    TARGET_MUST_KEYWORD_COVERAGE = 0.75
    TARGET_STRICT_CITATION_RATE = 0.6

    def __init__(self, config: Optional[Config] = None, db_path: Optional[str] = None):
        self.config = config or Config()
        if db_path is not None:
            self.config.database.db_path = db_path
        self.agent = ReActAgent(self.config)

    def load_golden_set(self, path: str) -> List[GoldenItem]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw_items = data.get("items", [])
        elif isinstance(data, list):
            raw_items = data
        else:
            raise ValueError(f"Unsupported golden-set payload type: {type(data).__name__}")
        items: List[GoldenItem] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict golden item at index %s", index)
                continue
            expected_block = item.get("expected", {})
            expected_keywords = item.get(
                "expected_keywords",
                expected_block.get("keywords", expected_block.get("keywords_required", [])),
            )
            expected_sources = item.get(
                "expected_sources",
                expected_block.get("sources", expected_block.get("evidence_set", [])),
            )
            forbidden_claims = expected_block.get("forbidden_claims", item.get("forbidden_claims", []))
            rubric = item.get("rubric", {})
            if isinstance(expected_keywords, list):
                expected_keywords = {"must": expected_keywords, "should": [], "optional": []}
            else:
                expected_keywords = {
                    "must": expected_keywords.get("must", []),
                    "should": expected_keywords.get("should", []),
                    "optional": expected_keywords.get("optional", []),
                }
            if isinstance(expected_sources, list):
                expected_sources = {"must": expected_sources, "should": [], "equivalent_sources": [], "source_aliases": {}}
            else:
                expected_sources = {
                    "must": expected_sources.get("must", []),
                    "should": expected_sources.get("should", []),
                    "equivalent_sources": expected_sources.get("equivalent_sources", []),
                    "source_aliases": expected_sources.get("source_aliases", {}),
                }
            expected_sources["must"] = self._normalize_expected_source_list(expected_sources.get("must", []))
            expected_sources["should"] = self._normalize_expected_source_list(expected_sources.get("should", []))
            items.append(GoldenItem(
                id=str(item.get("id", f"item-{index + 1:04d}")),
                question=str(item.get("question", item.get("query", ""))),
                scenario=item.get("scenario", item.get("task_type", "unknown")),
                difficulty=item.get("difficulty", item.get("slices", {}).get("difficulty", "medium")),
                source_of_question=item.get("source_of_question", item.get("source", "synthetic")),
                expected_keywords=expected_keywords,
                expected_sources=expected_sources,
                rubric=rubric if isinstance(rubric, dict) else {},
                forbidden_claims=[str(claim) for claim in forbidden_claims] if isinstance(forbidden_claims, list) else [],
                notes=item.get("notes", ""),
            ))
        logger.info("Loaded %s golden set items from %s", len(items), path)
        return items

    @staticmethod
    def _normalize_expected_source_list(raw_sources: List[Any]) -> List[str]:
        normalized: List[str] = []
        for value in raw_sources:
            if isinstance(value, str):
                text = value.strip()
                if text:
                    normalized.append(text)
                continue
            if not isinstance(value, dict):
                continue
            for key in ("source", "source_id", "title", "doc_id", "id"):
                token = str(value.get(key, "")).strip()
                if token:
                    normalized.append(token)
                    break
        return normalized

    def validate_expected_sources(self, items: List[GoldenItem]) -> Dict[str, Any]:
        indexed_sources = self._load_indexed_sources()
        normalized_indexed = [self._normalize_text(source) for source in indexed_sources]
        missing_by_item: Dict[str, Dict[str, List[str]]] = {}
        expected_total = 0
        missing_total = 0

        for item in items:
            expanded = self._expand_sources(item.expected_sources)
            item_missing_must: List[str] = []
            item_missing_should: List[str] = []

            for source in expanded["must"]:
                expected_total += 1
                if not any(self._source_match(actual, source) for actual in normalized_indexed):
                    missing_total += 1
                    item_missing_must.append(source)

            for source in expanded["should"]:
                expected_total += 1
                if not any(self._source_match(actual, source) for actual in normalized_indexed):
                    missing_total += 1
                    item_missing_should.append(source)

            if item_missing_must or item_missing_should:
                missing_by_item[item.id] = {
                    "must": item_missing_must,
                    "should": item_missing_should,
                }

        return {
            "indexed_source_count": len(indexed_sources),
            "expected_source_count": expected_total,
            "missing_count": missing_total,
            "missing_rate": round(missing_total / expected_total, 4) if expected_total else 0.0,
            "missing_by_item": missing_by_item,
            "indexed_sources": indexed_sources,
        }

    def evaluate_single(self, item: GoldenItem) -> Tuple[EvalResult, Dict[str, Any]]:
        started = time.time()
        exception_type = ""
        exception_message = ""
        try:
            response = self.agent.run_sync(item.question, include_trace=True)
            answer = response.answer
            citations = response.citations
            agent_trace = response.trace
        except Exception as exc:
            logger.exception("Agent execution failed for [%s]", item.id)
            exception_type = type(exc).__name__
            exception_message = str(exc)
            answer = f"Execution failed: {exc}"
            citations = []
            agent_trace = {
                "query": item.question,
                "search": {"final_results": [], "errors": [str(exc)]},
                "strategy_execution": [
                    {
                        "stage": "agent_exception",
                        "error_type": exception_type,
                        "error": str(exc),
                    }
                ],
            }

        latency = time.time() - started
        retrieved_sources = [row.get("source", "") for row in agent_trace.get("search", {}).get("final_results", []) if row.get("source")]
        must_source_recall, source_precision, source_precision_applicable = self.calculate_source_metrics(retrieved_sources, item.expected_sources)
        must_cov, should_cov, keyword_score, matched_must, matched_should = self.calculate_keyword_coverage(answer, item.expected_keywords)
        strict_hit, relaxed_hit = self.check_citation_validity(citations, item.expected_sources)
        claim_supported_rate, claim_citation_precision = self.calculate_claim_metrics(
            answer=answer,
            citations=citations,
            rag_trace=agent_trace,
        )
        hallucination_rate = self.calculate_hallucination_rate(claim_supported_rate)
        answer_completeness = self.calculate_answer_completeness(
            answer=answer,
            must_keyword_coverage=must_cov,
            rubric=item.rubric,
        )
        instruction_following_rate = self.calculate_instruction_following_rate(
            answer=answer,
            citations=citations,
            rubric=item.rubric,
            forbidden_claims=item.forbidden_claims,
        )
        actionability_applicable = self._is_actionability_applicable(item)
        actionability_score = self.calculate_actionability_score(answer) if actionability_applicable else 0.0
        generation_quality_score = self.calculate_generation_quality_score(
            answer_completeness=answer_completeness,
            instruction_following_rate=instruction_following_rate,
            actionability_score=actionability_score if actionability_applicable else None,
        )

        issues = []
        if must_source_recall < self.TARGET_MUST_SOURCE_RECALL:
            issues.append(f"Must Source Recall {must_source_recall:.2f} below target {self.TARGET_MUST_SOURCE_RECALL:.2f}")
        if must_cov < self.TARGET_MUST_KEYWORD_COVERAGE:
            issues.append(f"Must Keyword Coverage {must_cov:.2f} below target {self.TARGET_MUST_KEYWORD_COVERAGE:.2f}")
        if not strict_hit:
            issues.append("Strict Citation did not cover every must source.")
        if not relaxed_hit:
            issues.append("Relaxed Citation did not hit any must/equivalent source.")

        quality_gate_fail_reasons: List[str] = []
        if must_source_recall < self.TARGET_MUST_SOURCE_RECALL:
            quality_gate_fail_reasons.append("must_source_recall_below_target")
        if must_cov < self.TARGET_MUST_KEYWORD_COVERAGE:
            quality_gate_fail_reasons.append("must_keyword_coverage_below_target")
        if not strict_hit:
            quality_gate_fail_reasons.append("strict_citation_not_hit")

        empty_recall_reason = ""
        strategy_stages = agent_trace.get("strategy_execution", [])
        if isinstance(strategy_stages, list):
            for stage in strategy_stages:
                if str(stage.get("stage")) == "fallback_answer":
                    empty_recall_reason = str(stage.get("reason", ""))
                    break
            if not empty_recall_reason and not retrieved_sources:
                for stage in strategy_stages:
                    if str(stage.get("stage")) == "agent_exception":
                        empty_recall_reason = "agent_exception"
                        break
        if not empty_recall_reason and not retrieved_sources:
            empty_recall_reason = "no_retrieval_results"

        stage_errors: List[Dict[str, str]] = []
        for stage in strategy_stages if isinstance(strategy_stages, list) else []:
            if stage.get("error"):
                stage_errors.append(
                    {
                        "stage": str(stage.get("stage", "unknown")),
                        "error_type": str(stage.get("error_type", "")),
                        "error": str(stage.get("error", "")),
                    }
                )
        search_errors = agent_trace.get("search", {}).get("errors", [])
        if isinstance(search_errors, list):
            for err in search_errors:
                stage_errors.append({"stage": "search", "error_type": "", "error": str(err)})

        failure_path = {
            "status": "failed" if exception_type else ("degraded" if issues else "passed"),
            "exception_type": exception_type,
            "exception_message": exception_message,
            "retrieval_empty": len(retrieved_sources) == 0,
            "empty_recall_reason": empty_recall_reason,
            "quality_gate_fail_reasons": quality_gate_fail_reasons,
            "stage_errors": stage_errors,
        }

        result = EvalResult(
            id=item.id,
            question=item.question,
            scenario=item.scenario,
            difficulty=item.difficulty,
            source_of_question=item.source_of_question,
            answer=answer,
            citations=citations,
            must_source_recall=must_source_recall,
            source_precision=source_precision,
            source_precision_applicable=source_precision_applicable,
            must_keyword_coverage=must_cov,
            should_keyword_coverage=should_cov,
            keyword_score=keyword_score,
            strict_citation_hit=strict_hit,
            relaxed_citation_hit=relaxed_hit,
            claim_supported_rate=claim_supported_rate,
            claim_citation_precision=claim_citation_precision,
            hallucination_rate=hallucination_rate,
            answer_completeness=answer_completeness,
            instruction_following_rate=instruction_following_rate,
            actionability_score=actionability_score,
            actionability_applicable=actionability_applicable,
            generation_quality_score=generation_quality_score,
            retrieved_sources=retrieved_sources,
            matched_must_keywords=matched_must,
            matched_should_keywords=matched_should,
            issues=issues,
            latency=latency,
            failure_path=failure_path,
        )
        trace_payload = {
            "id": item.id,
            "question": item.question,
            "scenario": item.scenario,
            "difficulty": item.difficulty,
            "source_of_question": item.source_of_question,
            "expected": {
                "keywords": item.expected_keywords,
                "sources": item.expected_sources,
                "rubric": item.rubric,
                "forbidden_claims": item.forbidden_claims,
                "notes": item.notes,
            },
            "metrics": {
                "must_source_recall": result.must_source_recall,
                "source_precision": result.source_precision,
                "source_precision_applicable": result.source_precision_applicable,
                "must_keyword_coverage": result.must_keyword_coverage,
                "should_keyword_coverage": result.should_keyword_coverage,
                "keyword_score": result.keyword_score,
                "strict_citation_hit": result.strict_citation_hit,
                "relaxed_citation_hit": result.relaxed_citation_hit,
                "claim_supported_rate": result.claim_supported_rate,
                "claim_citation_precision": result.claim_citation_precision,
                "hallucination_rate": result.hallucination_rate,
                "answer_completeness": result.answer_completeness,
                "instruction_following_rate": result.instruction_following_rate,
                "actionability_score": result.actionability_score,
                "actionability_applicable": result.actionability_applicable,
                "generation_quality_score": result.generation_quality_score,
                "latency": result.latency,
            },
            "failure_path": failure_path,
            "rag_trace": agent_trace,
            "artifacts": {
                "issues": result.issues,
                "retrieved_sources": result.retrieved_sources,
                "citations": result.citations,
                "matched_must_keywords": result.matched_must_keywords,
                "matched_should_keywords": result.matched_should_keywords,
                "answer": result.answer,
            },
        }
        return result, trace_payload

    def evaluate_all(
        self,
        items: List[GoldenItem],
        repeat: int = 1,
    ) -> Tuple[EvalSummary, List[EvalResult], Dict[str, Dict[str, Any]]]:
        repeat = max(1, int(repeat))
        results: List[EvalResult] = []
        traces: Dict[str, Dict[str, Any]] = {}
        for index, item in enumerate(items, start=1):
            logger.info("Evaluating [%s/%s] %s (repeat=%s)", index, len(items), item.id, repeat)
            result, trace_payload = self.evaluate_single(item)
            run_metrics: List[Dict[str, Any]] = [self._compact_run_metrics(result, run_index=1)]
            pass_count = 1 if self._is_result_pass(result) else 0
            for run_index in range(2, repeat + 1):
                rerun_result, _ = self.evaluate_single(item)
                run_metrics.append(self._compact_run_metrics(rerun_result, run_index=run_index))
                if self._is_result_pass(rerun_result):
                    pass_count += 1
            result.repeat_runs = repeat
            result.pass_count = pass_count
            result.pass_rate_at_n = round(pass_count / repeat, 4)
            trace_payload["multi_run"] = {
                "repeat": repeat,
                "pass_count": pass_count,
                "pass_rate_at_n": result.pass_rate_at_n,
                "runs": run_metrics,
            }
            results.append(result)
            traces[item.id] = trace_payload
        return self._calculate_summary(results, repeat_runs=repeat), results, traces

    def write_trace_artifacts(self, dataset_name: str, run_tag: str, traces: Dict[str, Dict[str, Any]], trace_root: Path) -> Dict[str, Any]:
        dataset_dir = trace_root / dataset_name / run_tag
        dataset_dir.mkdir(parents=True, exist_ok=True)
        items_mapping: Dict[str, Dict[str, str]] = {}
        for item_id, payload in traces.items():
            trace_file = dataset_dir / f"{item_id}.json"
            trace_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            items_mapping[item_id] = {"trace_file": str(trace_file.resolve())}
        run_manifest = {"run_tag": run_tag, "dir": str(dataset_dir.resolve()), "items": items_mapping}
        self._merge_dataset_run_manifest(trace_root / "index.json", dataset_name, run_tag, run_manifest)
        (dataset_dir / "index.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_manifest

    def write_html_artifacts(self, dataset_name: str, run_tag: str, results: List[EvalResult], traces: Dict[str, Dict[str, Any]], html_root: Path) -> Dict[str, Any]:
        dataset_dir = html_root / dataset_name / run_tag
        dataset_dir.mkdir(parents=True, exist_ok=True)
        items_mapping: Dict[str, Dict[str, str]] = {}
        for result in results:
            payload = traces[result.id]
            html_file = dataset_dir / f"{result.id}.html"
            html_file.write_text(self._render_trace_html(dataset_name, payload), encoding="utf-8")
            result.html_file = str(html_file.resolve())
            items_mapping[result.id] = {"html_file": str(html_file.resolve())}
        index_file = dataset_dir / "index.html"
        index_file.write_text(self._render_dataset_index_html(dataset_name, run_tag, results), encoding="utf-8")
        run_manifest = {
            "run_tag": run_tag,
            "dir": str(dataset_dir.resolve()),
            "index_file": str(index_file.resolve()),
            "items": items_mapping,
        }
        self._merge_dataset_run_manifest(html_root / "index.json", dataset_name, run_tag, run_manifest)
        (dataset_dir / "index.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_manifest

    def write_diff_artifacts(
        self,
        dataset_name: str,
        run_tag: str,
        traces: Dict[str, Dict[str, Any]],
        trace_root: Path,
        html_root: Path,
    ) -> Dict[str, Any]:
        previous_run_tag, previous_items = self._find_previous_run(dataset_name, run_tag, trace_root)
        if not previous_run_tag or not previous_items:
            return {
                "enabled": True,
                "current_run_tag": run_tag,
                "previous_run_tag": "",
                "compare_index_file": "",
                "item_count": 0,
                "note": "No previous run found for comparison.",
            }

        compare_dir = html_root / dataset_name / run_tag / f"compare_vs_{previous_run_tag}"
        compare_dir.mkdir(parents=True, exist_ok=True)
        rows: List[str] = []
        item_count = 0
        for item_id, current_payload in traces.items():
            previous_trace_file = previous_items.get(item_id, {}).get("trace_file", "")
            if not previous_trace_file or not Path(previous_trace_file).exists():
                continue
            previous_payload = json.loads(Path(previous_trace_file).read_text(encoding="utf-8"))
            diff_file = compare_dir / f"{item_id}.html"
            diff_file.write_text(
                self._render_item_diff_html(
                    dataset_name=dataset_name,
                    item_id=item_id,
                    current_run_tag=run_tag,
                    previous_run_tag=previous_run_tag,
                    current_payload=current_payload,
                    previous_payload=previous_payload,
                ),
                encoding="utf-8",
            )
            item_count += 1
            rows.append(
                "<tr>"
                f"<td>{html.escape(item_id)}</td>"
                f"<td><a href=\"{html.escape(diff_file.name)}\">open</a></td>"
                "</tr>"
            )

        index_file = compare_dir / "index.html"
        index_file.write_text(
            self._render_diff_index_html(
                dataset_name=dataset_name,
                current_run_tag=run_tag,
                previous_run_tag=previous_run_tag,
                rows="\n".join(rows),
            ),
            encoding="utf-8",
        )
        return {
            "enabled": True,
            "current_run_tag": run_tag,
            "previous_run_tag": previous_run_tag,
            "compare_index_file": str(index_file.resolve()),
            "item_count": item_count,
            "dir": str(compare_dir.resolve()),
        }

    def attach_artifact_paths(
        self,
        results: List[EvalResult],
        trace_manifest: Dict[str, Any],
        html_manifest: Dict[str, Any],
        diff_manifest: Dict[str, Any],
    ) -> None:
        trace_items = trace_manifest.get("items", {})
        html_items = html_manifest.get("items", {})
        diff_file_by_id: Dict[str, str] = {}
        compare_dir = diff_manifest.get("dir", "")
        if compare_dir and Path(compare_dir).exists():
            for file_path in Path(compare_dir).glob("*.html"):
                if file_path.name.lower() == "index.html":
                    continue
                diff_file_by_id[file_path.stem] = f"{file_path.parent.name}/{file_path.name}"
        for result in results:
            result.trace_file = trace_items.get(result.id, {}).get("trace_file", "")
            result.html_file = html_items.get(result.id, {}).get("html_file", "")
            result.diff_file = diff_file_by_id.get(result.id, "")

    def generate_report(
        self,
        summary: EvalSummary,
        results: List[EvalResult],
        output_path: str,
        quality_gate_enabled: bool,
        dataset_name: str,
        run_tag: str,
        trace_manifest: Dict[str, Any],
        html_manifest: Dict[str, Any],
        diff_manifest: Dict[str, Any],
        report_manifest: Dict[str, Any],
        source_validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        quality_gate_failed = self._quality_gate_failed(summary)
        failure_aggregation = self._aggregate_failure_paths(results)
        report = {
            "generated_at": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "run_tag": run_tag,
            "summary": asdict(summary),
            "details": [asdict(result) for result in results],
            "targets": {
                "must_source_recall": self.TARGET_MUST_SOURCE_RECALL,
                "must_keyword_coverage": self.TARGET_MUST_KEYWORD_COVERAGE,
                "strict_citation_rate": self.TARGET_STRICT_CITATION_RATE,
            },
            "quality_gate": {
                "enabled": quality_gate_enabled,
                "failed": quality_gate_failed if quality_gate_enabled else False,
                "would_fail_if_enabled": quality_gate_failed,
            },
            "source_validation": source_validation or {},
            "failure_aggregation": failure_aggregation,
            "artifacts": {"trace": trace_manifest, "html": html_manifest, "diff": diff_manifest, "report": report_manifest},
        }
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved evaluation report to %s", output_file)
        return report

    def write_report_artifacts(
        self,
        dataset_name: str,
        run_tag: str,
        output_path: Path,
        report_root: Path,
    ) -> Dict[str, Any]:
        output_file = output_path.resolve()
        run_manifest = {
            "run_tag": run_tag,
            "dir": str(output_file.parent),
            "report_file": str(output_file),
        }
        self._merge_dataset_run_manifest(report_root / "index.json", dataset_name, run_tag, run_manifest)
        dataset_dir = report_root / dataset_name / run_tag
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "index.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_manifest

    def print_summary(self, summary: EvalSummary, quality_gate_enabled: bool) -> None:
        print("\n" + "=" * 72)
        print("ECBot evaluation report")
        print("=" * 72)
        print(f"Items: {summary.total_items}")
        print(f"Average latency: {summary.latency_avg:.2f}s")
        print("-" * 72)
        print(f"Must Source Recall:      {summary.must_source_recall_avg:.2%}")
        print(f"Source Precision:        {summary.source_precision_avg:.2%}")
        print(f"Source Precision Scope:  {summary.source_precision_applicable_items}/{summary.total_items}")
        print(f"Must Keyword Coverage:   {summary.must_keyword_coverage_avg:.2%}")
        print(f"Should Keyword Coverage: {summary.should_keyword_coverage_avg:.2%}")
        print(f"Keyword Score:           {summary.keyword_score_avg:.2%}")
        print(f"Strict Citation Rate:    {summary.strict_citation_rate:.2%}")
        print(f"Relaxed Citation Rate:   {summary.relaxed_citation_rate:.2%}")
        print(f"Claim Supported Rate:    {summary.claim_supported_rate_avg:.2%}")
        print(f"Claim Citation Precision:{summary.claim_citation_precision_avg:.2%}")
        print(f"Hallucination Rate:      {summary.hallucination_rate_avg:.2%}")
        print(f"Answer Completeness:     {summary.answer_completeness_avg:.2%}")
        print(f"Instruction Following:   {summary.instruction_following_rate_avg:.2%}")
        print(f"Actionability Score:     {summary.actionability_score_avg:.2%}")
        print(f"Actionability Scope:     {summary.actionability_applicable_items}/{summary.total_items}")
        print(f"Generation Quality:      {summary.generation_quality_score_avg:.2%}")
        print(f"Pass Rate@N (N={summary.repeat_runs}): {summary.pass_rate_at_n_avg:.2%}")
        print("-" * 72)
        print(f"Overall Score: {summary.overall_score:.2%}")
        print(f"Overall Score v2: {summary.overall_score_v2:.2%}")
        print(f"Grade: {summary.grade}")
        print(f"Quality gate enabled: {quality_gate_enabled}")
        print("=" * 72)

    def calculate_source_metrics(self, retrieved_sources: List[str], expected_sources: Dict[str, Any]) -> Tuple[float, float, bool]:
        expanded = self._expand_sources(expected_sources)
        must_sources = expanded["must"]
        all_expected = expanded["all"]
        source_precision_applicable = bool(all_expected)
        normalized_retrieved = [self._normalize_text(source) for source in retrieved_sources if str(source).strip()]
        if must_sources:
            matched_must = 0
            for expected in must_sources:
                if any(self._source_match(actual, expected) for actual in normalized_retrieved):
                    matched_must += 1
            must_source_recall = matched_must / len(must_sources)
        else:
            must_source_recall = 1.0
        # No expected source labels means precision is not applicable for this item.
        # Use a neutral value to avoid false-low averages caused by label absence.
        if not source_precision_applicable:
            source_precision = 1.0
        elif normalized_retrieved:
            matched_retrieved = 0
            for actual in normalized_retrieved:
                if any(self._source_match(actual, expected) for expected in all_expected):
                    matched_retrieved += 1
            source_precision = matched_retrieved / len(normalized_retrieved)
        else:
            source_precision = 0.0
        return must_source_recall, source_precision, source_precision_applicable

    def calculate_keyword_coverage(self, answer: str, expected_keywords: Dict[str, List[str]]) -> Tuple[float, float, float, List[str], List[str]]:
        answer_lower = answer.lower()
        must_keywords = expected_keywords.get("must", [])
        should_keywords = expected_keywords.get("should", [])
        matched_must = [keyword for keyword in must_keywords if keyword.lower() in answer_lower]
        matched_should = [keyword for keyword in should_keywords if keyword.lower() in answer_lower]
        must_coverage = len(matched_must) / len(must_keywords) if must_keywords else 1.0
        should_coverage = len(matched_should) / len(should_keywords) if should_keywords else 1.0
        keyword_score = 0.7 * must_coverage + 0.3 * should_coverage if should_keywords else must_coverage
        return must_coverage, should_coverage, keyword_score, matched_must, matched_should

    def check_citation_validity(self, citations: List[Dict[str, Any]], expected_sources: Dict[str, Any]) -> Tuple[bool, bool]:
        expanded = self._expand_sources(expected_sources)
        must_sources = expanded["must"]
        equivalent_groups = expanded["equivalent_groups"]
        if not must_sources:
            return True, True
        if not citations:
            return False, False
        cited_sources = self._extract_cited_sources(citations)
        strict_hit = all(any(self._source_match(cited, must) for cited in cited_sources) for must in must_sources)
        relaxed_hit = any(any(self._source_match(cited, must) for cited in cited_sources) for must in must_sources)
        if not relaxed_hit:
            for group in equivalent_groups:
                if any(any(self._source_match(cited, candidate) for cited in cited_sources) for candidate in group):
                    relaxed_hit = True
                    break
        return strict_hit, relaxed_hit

    def calculate_claim_metrics(
        self,
        *,
        answer: str,
        citations: List[Dict[str, Any]],
        rag_trace: Dict[str, Any],
    ) -> Tuple[float, float]:
        claims = self._split_claims(answer)
        if not claims:
            return 0.0, 0.0

        evidence_rows = self._collect_evidence_rows(rag_trace)
        evidence_by_source: Dict[str, List[str]] = {}
        if isinstance(evidence_rows, list):
            for row in evidence_rows:
                if not isinstance(row, dict):
                    continue
                source = self._normalize_text(
                    str(row.get("source", "")).strip() or str(row.get("source_path", "")).strip()
                )
                content = str(row.get("content", "")).strip()
                if not source or not content:
                    continue
                evidence_by_source.setdefault(source, []).append(content)

        cited_sources = self._extract_cited_sources(citations)
        supported_count = 0
        cited_claim_count = 0
        cited_claim_correct_count = 0

        for claim in claims:
            best_source, support_score = self._best_claim_support(claim, evidence_by_source)
            is_supported = support_score >= 0.15
            if is_supported:
                supported_count += 1

            has_citation = bool(cited_sources)
            if has_citation:
                cited_claim_count += 1
                if best_source and any(
                    self._source_match(best_source, cited) for cited in cited_sources
                ):
                    cited_claim_correct_count += 1

        claim_supported_rate = supported_count / len(claims)
        if cited_claim_count == 0:
            claim_citation_precision = 0.0
        else:
            claim_citation_precision = cited_claim_correct_count / cited_claim_count

        return round(claim_supported_rate, 4), round(claim_citation_precision, 4)

    def _collect_evidence_rows(self, rag_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        search_trace = rag_trace.get("search", {}) if isinstance(rag_trace, dict) else {}
        final_rows = search_trace.get("final_results", [])
        if not isinstance(final_rows, list):
            final_rows = []
        has_content = any(
            isinstance(row, dict) and str(row.get("content", "")).strip()
            for row in final_rows
        )
        if has_content:
            return final_rows
        # Backward compatibility: older traces may only include content under context_selection.
        fallback_rows = search_trace.get("context_selection", [])
        if isinstance(fallback_rows, list):
            return fallback_rows
        return final_rows

    def calculate_hallucination_rate(self, claim_supported_rate: float) -> float:
        bounded_support = max(0.0, min(1.0, float(claim_supported_rate)))
        return round(1.0 - bounded_support, 4)

    def calculate_answer_completeness(
        self,
        *,
        answer: str,
        must_keyword_coverage: float,
        rubric: Dict[str, Any],
    ) -> float:
        section_coverage = self._section_coverage(answer, rubric.get("must_have_sections", []))
        if section_coverage is None:
            score = float(must_keyword_coverage)
        else:
            score = 0.6 * section_coverage + 0.4 * float(must_keyword_coverage)
        return round(max(0.0, min(1.0, score)), 4)

    def calculate_instruction_following_rate(
        self,
        *,
        answer: str,
        citations: List[Dict[str, Any]],
        rubric: Dict[str, Any],
        forbidden_claims: List[str],
    ) -> float:
        checks: List[bool] = []
        if bool(rubric.get("citation_required", False)):
            checks.append(bool(citations))

        section_coverage = self._section_coverage(answer, rubric.get("must_have_sections", []))
        if section_coverage is not None:
            checks.append(section_coverage >= 1.0)

        forbidden = [str(term).strip().lower() for term in forbidden_claims if str(term).strip()]
        if forbidden:
            lowered = answer.lower()
            checks.append(not any(term in lowered for term in forbidden))

        if not checks:
            return 1.0
        return round(sum(1 for ok in checks if ok) / len(checks), 4)

    def calculate_actionability_score(self, answer: str) -> float:
        text = answer.strip()
        if not text:
            return 0.0
        has_step_pattern = bool(re.search(r"(^|\n)\s*(\d+\.\s+|[一二三四五六七八九十]+[、.])", text))
        has_action_terms = bool(re.search(r"(建议|步骤|执行|落实|优先|先|再|然后|最后|step)", text, flags=re.IGNORECASE))
        has_constraint_terms = bool(re.search(r"(风险|注意|前提|如果|条件|限制|成本)", text, flags=re.IGNORECASE))
        score = (float(has_step_pattern or has_action_terms) + float(has_action_terms) + float(has_constraint_terms)) / 3.0
        return round(score, 4)

    def calculate_generation_quality_score(
        self,
        *,
        answer_completeness: float,
        instruction_following_rate: float,
        actionability_score: Optional[float],
    ) -> float:
        metrics = [float(answer_completeness), float(instruction_following_rate)]
        if actionability_score is not None:
            metrics.append(float(actionability_score))
        score = sum(metrics) / max(len(metrics), 1)
        return round(max(0.0, min(1.0, score)), 4)

    def _is_actionability_applicable(self, item: GoldenItem) -> bool:
        scenario = str(item.scenario or "").strip().lower()
        if scenario in {"type_a_fact_qa", "fact_qa"}:
            return False
        rubric = item.rubric if isinstance(item.rubric, dict) else {}
        sections = [
            str(section).strip().lower()
            for section in rubric.get("must_have_sections", [])
            if str(section).strip()
        ]
        if any(token in {"步骤", "执行建议", "step", "steps", "procedure"} for token in sections):
            return True
        if any(token in scenario for token in ("procedure", "workflow", "plan")):
            return True
        return False

    def _section_coverage(self, answer: str, sections: Any) -> Optional[float]:
        if not isinstance(sections, list):
            return None
        normalized_sections = [str(section).strip().lower() for section in sections if str(section).strip()]
        if not normalized_sections:
            return None
        lowered = answer.lower()
        matched = sum(1 for section in normalized_sections if section in lowered)
        return matched / len(normalized_sections)

    def _extract_cited_sources(self, citations: List[Dict[str, Any]]) -> set[str]:
        cited_sources: set[str] = set()
        for citation in citations:
            source = str(citation.get("source", "")).strip()
            title = str(citation.get("title", "")).strip()
            canonical = str(citation.get("canonical_source_id", "")).strip()
            if source:
                cited_sources.add(self._normalize_text(source))
            if title:
                cited_sources.add(self._normalize_text(title))
            if canonical:
                cited_sources.add(self._normalize_text(canonical))
            aliases = citation.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    alias_text = str(alias).strip()
                    if alias_text:
                        cited_sources.add(self._normalize_text(alias_text))
            versions = citation.get("versions", [])
            if isinstance(versions, list):
                for version in versions:
                    if not isinstance(version, dict):
                        continue
                    version_source = str(version.get("source", "")).strip()
                    if version_source:
                        cited_sources.add(self._normalize_text(version_source))
        return cited_sources

    def _split_claims(self, answer: str) -> List[str]:
        if not answer.strip():
            return []
        claims: List[str] = []
        in_reference = False
        for raw in answer.replace("\r", "\n").split("\n"):
            line = str(raw).strip()
            if not line:
                continue
            if line.startswith("来源：") or line.lower().startswith("sources:"):
                in_reference = True
                continue
            if in_reference:
                continue
            if re.match(r"^[-*]?\s*\[S\d+\]", line):
                continue
            lowered_line = line.lower()
            if "| chunk#" in lowered_line or "| procedure" in lowered_line or "| reference" in lowered_line:
                continue
            for part in re.split(r"[。！？!?；;]+", line):
                claim = re.sub(r"\s+", " ", part).strip(" -\t")
                if len(claim) < 8:
                    continue
                claims.append(claim)
        return claims

    def _best_claim_support(
        self,
        claim: str,
        evidence_by_source: Dict[str, List[str]],
    ) -> Tuple[str, float]:
        claim_tokens = self._text_tokens(claim)
        if not claim_tokens:
            return "", 0.0
        best_source = ""
        best_score = 0.0
        for source, snippets in evidence_by_source.items():
            for snippet in snippets:
                overlap = self._token_overlap(claim_tokens, self._text_tokens(snippet))
                if overlap > best_score:
                    best_score = overlap
                    best_source = source
        return best_source, best_score

    def _text_tokens(self, text: str) -> set[str]:
        lowered = text.lower()
        latin_tokens = set(re.findall(r"[a-z0-9_]{2,}", lowered))
        cjk_tokens = set(re.findall(r"[\u4e00-\u9fff]", lowered))
        return latin_tokens | cjk_tokens

    def _token_overlap(self, left: set[str], right: set[str]) -> float:
        if not left:
            return 0.0
        return len(left & right) / len(left)

    def _calculate_summary(self, results: List[EvalResult], repeat_runs: int = 1) -> EvalSummary:
        if not results:
            return EvalSummary()
        total = len(results)
        must_source_recall_avg = sum(r.must_source_recall for r in results) / total
        source_precision_items = [r for r in results if r.source_precision_applicable]
        source_precision_applicable_items = len(source_precision_items)
        source_precision_avg = (
            sum(r.source_precision for r in source_precision_items) / source_precision_applicable_items
            if source_precision_applicable_items
            else 1.0
        )
        must_keyword_coverage_avg = sum(r.must_keyword_coverage for r in results) / total
        should_keyword_coverage_avg = sum(r.should_keyword_coverage for r in results) / total
        keyword_score_avg = sum(r.keyword_score for r in results) / total
        strict_citation_rate = sum(1 for r in results if r.strict_citation_hit) / total
        relaxed_citation_rate = sum(1 for r in results if r.relaxed_citation_hit) / total
        claim_supported_rate_avg = sum(r.claim_supported_rate for r in results) / total
        claim_citation_precision_avg = sum(r.claim_citation_precision for r in results) / total
        hallucination_rate_avg = sum(r.hallucination_rate for r in results) / total
        answer_completeness_avg = sum(r.answer_completeness for r in results) / total
        instruction_following_rate_avg = sum(r.instruction_following_rate for r in results) / total
        actionability_items = [r for r in results if r.actionability_applicable]
        actionability_applicable_items = len(actionability_items)
        actionability_score_avg = (
            sum(r.actionability_score for r in actionability_items) / actionability_applicable_items
            if actionability_applicable_items
            else 1.0
        )
        generation_quality_score_avg = sum(r.generation_quality_score for r in results) / total
        pass_rate_at_n_avg = sum(r.pass_rate_at_n for r in results) / total
        latency_avg = sum(r.latency for r in results) / total
        overall_score = (
            must_source_recall_avg
            + must_keyword_coverage_avg
            + strict_citation_rate
            + claim_supported_rate_avg
            + claim_citation_precision_avg
        ) / 5
        factual_grounding_score = (
            must_source_recall_avg
            + strict_citation_rate
            + claim_supported_rate_avg
            + claim_citation_precision_avg
        ) / 4
        overall_score_v2 = 0.55 * factual_grounding_score + 0.45 * generation_quality_score_avg
        return EvalSummary(
            total_items=total,
            must_source_recall_avg=round(must_source_recall_avg, 4),
            source_precision_avg=round(source_precision_avg, 4),
            source_precision_applicable_items=source_precision_applicable_items,
            must_keyword_coverage_avg=round(must_keyword_coverage_avg, 4),
            should_keyword_coverage_avg=round(should_keyword_coverage_avg, 4),
            keyword_score_avg=round(keyword_score_avg, 4),
            strict_citation_rate=round(strict_citation_rate, 4),
            relaxed_citation_rate=round(relaxed_citation_rate, 4),
            claim_supported_rate_avg=round(claim_supported_rate_avg, 4),
            claim_citation_precision_avg=round(claim_citation_precision_avg, 4),
            hallucination_rate_avg=round(hallucination_rate_avg, 4),
            answer_completeness_avg=round(answer_completeness_avg, 4),
            instruction_following_rate_avg=round(instruction_following_rate_avg, 4),
            actionability_score_avg=round(actionability_score_avg, 4),
            actionability_applicable_items=actionability_applicable_items,
            generation_quality_score_avg=round(generation_quality_score_avg, 4),
            pass_rate_at_n_avg=round(pass_rate_at_n_avg, 4),
            repeat_runs=max(1, int(repeat_runs)),
            overall_score=round(overall_score, 4),
            overall_score_v2=round(overall_score_v2, 4),
            grade=self._determine_grade(must_source_recall_avg, must_keyword_coverage_avg, strict_citation_rate),
            latency_avg=round(latency_avg, 2),
            scenario_buckets=self._calculate_scenario_buckets(results),
        )

    def _calculate_scenario_buckets(self, results: List[EvalResult]) -> Dict[str, Dict[str, float]]:
        bucket_results: Dict[str, List[EvalResult]] = {}
        for result in results:
            bucket_results.setdefault(result.scenario, []).append(result)
        scenario_buckets: Dict[str, Dict[str, float]] = {}
        for scenario, items in bucket_results.items():
            count = len(items)
            source_precision_items = [i for i in items if i.source_precision_applicable]
            source_precision_avg = (
                sum(i.source_precision for i in source_precision_items) / len(source_precision_items)
                if source_precision_items
                else 1.0
            )
            actionability_items = [i for i in items if i.actionability_applicable]
            actionability_score_avg = (
                sum(i.actionability_score for i in actionability_items) / len(actionability_items)
                if actionability_items
                else 1.0
            )
            scenario_buckets[scenario] = {
                "count": count,
                "must_source_recall_avg": round(sum(i.must_source_recall for i in items) / count, 4),
                "source_precision_avg": round(source_precision_avg, 4),
                "source_precision_applicable_items": len(source_precision_items),
                "must_keyword_coverage_avg": round(sum(i.must_keyword_coverage for i in items) / count, 4),
                "strict_citation_rate": round(sum(1 for i in items if i.strict_citation_hit) / count, 4),
                "relaxed_citation_rate": round(sum(1 for i in items if i.relaxed_citation_hit) / count, 4),
                "claim_supported_rate_avg": round(sum(i.claim_supported_rate for i in items) / count, 4),
                "claim_citation_precision_avg": round(sum(i.claim_citation_precision for i in items) / count, 4),
                "hallucination_rate_avg": round(sum(i.hallucination_rate for i in items) / count, 4),
                "answer_completeness_avg": round(sum(i.answer_completeness for i in items) / count, 4),
                "instruction_following_rate_avg": round(sum(i.instruction_following_rate for i in items) / count, 4),
                "actionability_score_avg": round(actionability_score_avg, 4),
                "actionability_applicable_items": len(actionability_items),
                "generation_quality_score_avg": round(sum(i.generation_quality_score for i in items) / count, 4),
                "pass_rate_at_n_avg": round(sum(i.pass_rate_at_n for i in items) / count, 4),
            }
        return scenario_buckets

    def _determine_grade(self, must_source_recall: float, must_keyword_coverage: float, strict_citation_rate: float) -> str:
        if must_source_recall >= 0.85 and must_keyword_coverage >= 0.9 and strict_citation_rate >= 0.85:
            return "S"
        if must_source_recall >= 0.8 and must_keyword_coverage >= 0.85 and strict_citation_rate >= 0.75:
            return "A"
        if must_source_recall >= self.TARGET_MUST_SOURCE_RECALL and must_keyword_coverage >= self.TARGET_MUST_KEYWORD_COVERAGE and strict_citation_rate >= self.TARGET_STRICT_CITATION_RATE:
            return "B"
        if must_source_recall >= 0.5 and must_keyword_coverage >= 0.6 and strict_citation_rate >= 0.4:
            return "C"
        return "D"

    def _quality_gate_failed(self, summary: EvalSummary) -> bool:
        return summary.must_source_recall_avg < self.TARGET_MUST_SOURCE_RECALL or summary.strict_citation_rate < self.TARGET_STRICT_CITATION_RATE or summary.must_keyword_coverage_avg < self.TARGET_MUST_KEYWORD_COVERAGE

    def _is_result_pass(self, result: EvalResult) -> bool:
        return (
            result.must_source_recall >= self.TARGET_MUST_SOURCE_RECALL
            and result.must_keyword_coverage >= self.TARGET_MUST_KEYWORD_COVERAGE
            and result.strict_citation_hit
        )

    def _compact_run_metrics(self, result: EvalResult, run_index: int) -> Dict[str, Any]:
        return {
            "run_index": run_index,
            "must_source_recall": round(result.must_source_recall, 4),
            "must_keyword_coverage": round(result.must_keyword_coverage, 4),
            "strict_citation_hit": bool(result.strict_citation_hit),
            "source_precision": round(result.source_precision, 4),
            "source_precision_applicable": bool(result.source_precision_applicable),
            "claim_supported_rate": round(result.claim_supported_rate, 4),
            "hallucination_rate": round(result.hallucination_rate, 4),
            "answer_completeness": round(result.answer_completeness, 4),
            "instruction_following_rate": round(result.instruction_following_rate, 4),
            "actionability_score": round(result.actionability_score, 4),
            "actionability_applicable": bool(result.actionability_applicable),
            "generation_quality_score": round(result.generation_quality_score, 4),
            "latency": round(result.latency, 4),
            "status": str(result.failure_path.get("status", "unknown")),
            "pass": self._is_result_pass(result),
        }

    def _load_indexed_sources(self) -> List[str]:
        sources: List[str] = []
        try:
            with sqlite3.connect(f"file:{self.config.database.db_path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                    )
                }
                if "chunks" in tables:
                    sources.extend(
                        [
                            str(row["source_filename"]).strip()
                            for row in conn.execute(
                                "SELECT DISTINCT source_filename FROM chunks WHERE source_filename IS NOT NULL"
                            ).fetchall()
                            if str(row["source_filename"]).strip()
                        ]
                    )
                if "files" in tables:
                    sources.extend(
                        [
                            str(row["filename"]).strip()
                            for row in conn.execute(
                                "SELECT DISTINCT filename FROM files WHERE filename IS NOT NULL"
                            ).fetchall()
                            if str(row["filename"]).strip()
                        ]
                    )
        except sqlite3.Error as exc:
            logger.warning(
                "Failed to open indexed source DB '%s': %s. Continue with empty source inventory.",
                self.config.database.db_path,
                exc,
            )
            return []
        deduped: List[str] = []
        seen: set[str] = set()
        for source in sources:
            normalized = self._normalize_text(source)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(source)
        return deduped

    def _normalize_text(self, text: str) -> str:
        return str(text).strip().lower()

    def _expand_sources(self, expected_sources: Dict[str, Any]) -> Dict[str, Any]:
        must_sources = [self._normalize_text(source) for source in expected_sources.get("must", [])]
        should_sources = [self._normalize_text(source) for source in expected_sources.get("should", [])]
        equivalent_groups: List[List[str]] = []
        for group in expected_sources.get("equivalent_sources", []):
            if isinstance(group, list):
                normalized_group = [self._normalize_text(source) for source in group if str(source).strip()]
                if normalized_group:
                    equivalent_groups.append(normalized_group)
        alias_map = expected_sources.get("source_aliases", {})
        alias_pairs: List[Tuple[str, str]] = []
        if isinstance(alias_map, dict):
            for key, value in alias_map.items():
                key_norm = self._normalize_text(key)
                if isinstance(value, list):
                    for alias in value:
                        alias_pairs.append((key_norm, self._normalize_text(alias)))
                else:
                    alias_pairs.append((key_norm, self._normalize_text(value)))
        for key_norm, alias_norm in alias_pairs:
            if not alias_norm:
                continue
            if key_norm in must_sources and alias_norm not in must_sources:
                must_sources.append(alias_norm)
            if key_norm in should_sources and alias_norm not in should_sources:
                should_sources.append(alias_norm)
        all_expected = set(must_sources + should_sources)
        for group in equivalent_groups:
            all_expected.update(group)
        return {"must": must_sources, "should": should_sources, "all": list(all_expected), "equivalent_groups": equivalent_groups}

    def _source_match(self, actual: str, expected: str) -> bool:
        actual_norm = self._normalize_text(actual)
        expected_norm = self._normalize_text(expected)
        if not actual_norm or not expected_norm:
            return False
        return expected_norm in actual_norm or actual_norm in expected_norm

    def _aggregate_failure_paths(self, results: List[EvalResult]) -> Dict[str, Any]:
        status_counter: Counter[str] = Counter()
        exception_counter: Counter[str] = Counter()
        empty_recall_counter: Counter[str] = Counter()
        quality_gate_reason_counter: Counter[str] = Counter()
        stage_error_counter: Counter[str] = Counter()

        for result in results:
            failure_path = result.failure_path or {}
            status_counter[str(failure_path.get("status", "unknown"))] += 1
            exception_type = str(failure_path.get("exception_type", ""))
            if exception_type:
                exception_counter[exception_type] += 1
            empty_recall_reason = str(failure_path.get("empty_recall_reason", ""))
            if empty_recall_reason:
                empty_recall_counter[empty_recall_reason] += 1
            for reason in failure_path.get("quality_gate_fail_reasons", []):
                quality_gate_reason_counter[str(reason)] += 1
            for stage_error in failure_path.get("stage_errors", []):
                stage = str(stage_error.get("stage", "unknown"))
                stage_error_counter[stage] += 1

        return {
            "status_counts": dict(status_counter),
            "exception_type_counts": dict(exception_counter),
            "empty_recall_reason_counts": dict(empty_recall_counter),
            "quality_gate_reason_counts": dict(quality_gate_reason_counter),
            "stage_error_counts": dict(stage_error_counter),
        }

    def _merge_dataset_run_manifest(
        self,
        manifest_path: Path,
        dataset_name: str,
        run_tag: str,
        run_fragment: Dict[str, Any],
    ) -> None:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {}
        dataset_manifest = manifest.get(dataset_name, {"latest_run": "", "runs": {}})
        if not isinstance(dataset_manifest, dict):
            dataset_manifest = {"latest_run": "", "runs": {}}
        if "runs" not in dataset_manifest or not isinstance(dataset_manifest.get("runs"), dict):
            # Migrate legacy shape: {"dir": "...", "items": {...}, ...}
            legacy_run: Dict[str, Any] = {}
            if dataset_manifest.get("dir") or dataset_manifest.get("items"):
                legacy_run = {
                    "run_tag": "legacy_flat",
                    "dir": dataset_manifest.get("dir", ""),
                    "index_file": dataset_manifest.get("index_file", ""),
                    "items": dataset_manifest.get("items", {}),
                }
            dataset_manifest = {"latest_run": "legacy_flat" if legacy_run else "", "runs": {}}
            if legacy_run:
                dataset_manifest["runs"]["legacy_flat"] = legacy_run
        dataset_manifest["runs"][run_tag] = run_fragment
        dataset_manifest["latest_run"] = run_tag
        manifest[dataset_name] = dataset_manifest
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _find_previous_run(
        self,
        dataset_name: str,
        current_run_tag: str,
        trace_root: Path,
    ) -> Tuple[str, Dict[str, Dict[str, str]]]:
        manifest_path = trace_root / "index.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_manifest = manifest.get(dataset_name, {})
            runs = dataset_manifest.get("runs", {})
            run_tags = sorted(tag for tag in runs.keys() if tag != current_run_tag)
            if run_tags:
                previous_run_tag = run_tags[-1]
                previous_items = runs.get(previous_run_tag, {}).get("items", {})
                if isinstance(previous_items, dict):
                    return previous_run_tag, previous_items

        # Backward compatibility with flat legacy layout: Eval/trace/<dataset>/<id>.json
        legacy_dir = trace_root / dataset_name
        if legacy_dir.exists():
            legacy_items: Dict[str, Dict[str, str]] = {}
            for trace_file in legacy_dir.glob("*.json"):
                if trace_file.name.lower() == "index.json":
                    continue
                legacy_items[trace_file.stem] = {"trace_file": str(trace_file.resolve())}
            if legacy_items:
                return "legacy_flat", legacy_items
        return "", {}

    def _render_trace_html(self, dataset_name: str, payload: Dict[str, Any]) -> str:
        branch_diagnostics = payload.get("rag_trace", {}).get("search", {}).get("branch_diagnostics", {})
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(dataset_name)} / {html.escape(payload['id'])}</title>
<style>
:root{{--bg:#f6f2ea;--panel:#fffdf8;--ink:#1e1a17;--muted:#6b6259;--line:#d8cfc3;--accent:#a54b1a;--accent-soft:#f4dfd2}}
body{{margin:0;padding:32px;font-family:"Segoe UI","PingFang SC",sans-serif;color:var(--ink);background:radial-gradient(circle at top left,#f3d7c0 0,transparent 28%),linear-gradient(180deg,#f7f3ec 0%,#ece3d6 100%)}}
.wrap{{max-width:1180px;margin:0 auto;display:grid;gap:18px}} .hero,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:22px 24px;box-shadow:0 14px 40px rgba(68,46,28,.08)}}
.hero h1{{margin:0 0 8px;font-size:28px}} .meta{{color:var(--muted);display:flex;gap:14px;flex-wrap:wrap;font-size:14px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}} .grid-3{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}
h2{{margin:0 0 12px;font-size:18px}} pre{{margin:0;white-space:pre-wrap;word-break:break-word;background:#fcf8f2;border:1px solid #eee1d2;border-radius:12px;padding:14px;line-height:1.5;font-size:13px}}
.question{{padding:14px;border-radius:12px;background:var(--accent-soft);color:#6b2f10;font-weight:600}} .answer{{border-left:4px solid var(--accent)}}
@media (max-width: 1024px) {{ .grid-3{{grid-template-columns:1fr}} }}
</style></head><body><div class="wrap">
<section class="hero"><h1>{html.escape(payload['id'])} trace</h1><div class="meta"><span>dataset: {html.escape(dataset_name)}</span><span>scenario: {html.escape(payload['scenario'])}</span><span>difficulty: {html.escape(payload['difficulty'])}</span><span>source: {html.escape(payload['source_of_question'])}</span></div><div class="question">{html.escape(payload['question'])}</div></section>
<section class="grid-3"><article class="panel"><h2>Metrics</h2><pre>{self._json_block(payload['metrics'])}</pre></article><article class="panel answer"><h2>Answer</h2><pre>{html.escape(payload['artifacts']['answer'])}</pre></article><article class="panel"><h2>Expected</h2><pre>{self._json_block(payload['expected'])}</pre></article></section>
<section class="grid"><article class="panel"><h2>Branch Diagnostics</h2><pre>{self._json_block(branch_diagnostics)}</pre></article><article class="panel"><h2>Failure Path</h2><pre>{self._json_block(payload.get('failure_path', {}))}</pre></article></section>
<section class="grid"><article class="panel"><h2>Search Trace</h2><pre>{self._json_block(payload['rag_trace'].get('search', {}))}</pre></article><article class="panel"><h2>Strategy Execution</h2><pre>{self._json_block(payload['rag_trace'].get('strategy_execution', []))}</pre></article></section>
<section class="panel"><h2>Multi Run</h2><pre>{self._json_block(payload.get('multi_run', {}))}</pre></section>
<section class="panel"><h2>Artifacts</h2><pre>{self._json_block(payload['artifacts'])}</pre></section>
</div></body></html>"""

    def _render_dataset_index_html(self, dataset_name: str, run_tag: str, results: List[EvalResult]) -> str:
        repeat_runs = results[0].repeat_runs if results else 1
        rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(result.id)}</td>"
            f"<td>{html.escape(result.scenario)}</td>"
            f"<td>{result.must_source_recall:.2%}</td>"
            f"<td>{result.must_keyword_coverage:.2%}</td>"
            f"<td>{'yes' if result.strict_citation_hit else 'no'}</td>"
            f"<td>{result.claim_supported_rate:.2%}</td>"
            f"<td>{result.claim_citation_precision:.2%}</td>"
            f"<td>{result.hallucination_rate:.2%}</td>"
            f"<td>{result.answer_completeness:.2%}</td>"
            f"<td>{result.instruction_following_rate:.2%}</td>"
            f"<td>{result.actionability_score:.2%}</td>"
            f"<td>{result.generation_quality_score:.2%}</td>"
            f"<td>{result.pass_rate_at_n:.2%}</td>"
            f"<td>{html.escape(str(result.failure_path.get('empty_recall_reason', '')))}</td>"
            f"<td><a href=\"{html.escape(Path(result.html_file).name)}\">open</a></td>"
            f"<td>{('<a href=\"' + html.escape(result.diff_file) + '\">open</a>') if result.diff_file else '-'}</td>"
            "</tr>"
            for result in results
        )
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(dataset_name)} trace index</title>
<style>body{{margin:0;padding:32px;background:#f5f1e8;color:#211b16;font-family:"Segoe UI","PingFang SC",sans-serif}} .panel{{max-width:1080px;margin:0 auto;background:#fffdf8;border:1px solid #d8cfc3;border-radius:18px;padding:24px;box-shadow:0 14px 40px rgba(68,46,28,.08)}} table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:12px 10px;border-bottom:1px solid #eadfce}} th{{color:#6b6259;font-size:12px;text-transform:uppercase;letter-spacing:.06em}}</style>
</head><body><section class="panel"><h1>{html.escape(dataset_name)} trace index</h1><p>run_tag: {html.escape(run_tag)} | repeat: {repeat_runs}</p><table><thead><tr><th>ID</th><th>Scenario</th><th>Must Source Recall</th><th>Must Keyword Coverage</th><th>Strict Citation</th><th>Claim Supported</th><th>Claim Citation Precision</th><th>Hallucination</th><th>Answer Completeness</th><th>Instruction Following</th><th>Actionability</th><th>Generation Quality</th><th>Pass Rate@N</th><th>Empty Recall Reason</th><th>HTML</th><th>Diff</th></tr></thead><tbody>{rows}</tbody></table></section></body></html>"""

    def _render_item_diff_html(
        self,
        *,
        dataset_name: str,
        item_id: str,
        current_run_tag: str,
        previous_run_tag: str,
        current_payload: Dict[str, Any],
        previous_payload: Dict[str, Any],
    ) -> str:
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(dataset_name)} {html.escape(item_id)} diff</title>
<style>
body{{margin:0;padding:28px;background:#f5f1e8;color:#211b16;font-family:"Segoe UI","PingFang SC",sans-serif}}
.wrap{{max-width:1320px;margin:0 auto;display:grid;gap:16px}}
.panel{{background:#fffdf8;border:1px solid #d8cfc3;border-radius:16px;padding:18px;box-shadow:0 12px 36px rgba(68,46,28,.08)}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
pre{{margin:0;white-space:pre-wrap;word-break:break-word;background:#fcf8f2;border:1px solid #eee1d2;border-radius:12px;padding:12px;font-size:12px;line-height:1.45}}
h1,h2{{margin:0 0 10px}}
</style></head><body><div class="wrap">
<section class="panel"><h1>{html.escape(item_id)} diff</h1><div>dataset: {html.escape(dataset_name)} | current: {html.escape(current_run_tag)} | previous: {html.escape(previous_run_tag)}</div></section>
<section class="grid">
<article class="panel"><h2>Current Metrics</h2><pre>{self._json_block(current_payload.get("metrics", {}))}</pre></article>
<article class="panel"><h2>Previous Metrics</h2><pre>{self._json_block(previous_payload.get("metrics", {}))}</pre></article>
</section>
<section class="grid">
<article class="panel"><h2>Current Branch Diagnostics</h2><pre>{self._json_block(current_payload.get("rag_trace", {}).get("search", {}).get("branch_diagnostics", {}))}</pre></article>
<article class="panel"><h2>Previous Branch Diagnostics</h2><pre>{self._json_block(previous_payload.get("rag_trace", {}).get("search", {}).get("branch_diagnostics", {}))}</pre></article>
</section>
<section class="grid">
<article class="panel"><h2>Current Failure Path</h2><pre>{self._json_block(current_payload.get("failure_path", {}))}</pre></article>
<article class="panel"><h2>Previous Failure Path</h2><pre>{self._json_block(previous_payload.get("failure_path", {}))}</pre></article>
</section>
</div></body></html>"""

    def _render_diff_index_html(
        self,
        *,
        dataset_name: str,
        current_run_tag: str,
        previous_run_tag: str,
        rows: str,
    ) -> str:
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(dataset_name)} compare index</title>
<style>body{{margin:0;padding:32px;background:#f5f1e8;color:#211b16;font-family:"Segoe UI","PingFang SC",sans-serif}} .panel{{max-width:980px;margin:0 auto;background:#fffdf8;border:1px solid #d8cfc3;border-radius:18px;padding:24px;box-shadow:0 14px 40px rgba(68,46,28,.08)}} table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:12px 10px;border-bottom:1px solid #eadfce}} th{{color:#6b6259;font-size:12px;text-transform:uppercase;letter-spacing:.06em}}</style>
</head><body><section class="panel"><h1>{html.escape(dataset_name)} compare index</h1><p>current: {html.escape(current_run_tag)} | previous: {html.escape(previous_run_tag)}</p><table><thead><tr><th>ID</th><th>Diff</th></tr></thead><tbody>{rows}</tbody></table></section></body></html>"""

    def _json_block(self, payload: Any) -> str:
        return html.escape(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ECBot golden set evaluation.")
    parser.add_argument("--dataset", default="Eval/golden_set.json", help="Path to the golden set dataset.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional report path. Defaults to <report_dir>/<dataset_name>/<run_tag>/eval_report.json.",
    )
    parser.add_argument("--db-path", default=None, help="Override database path.")
    parser.add_argument("--report-dir", default=None, help="Directory for evaluation reports.")
    parser.add_argument("--trace-dir", default=None, help="Directory for raw trace JSON files.")
    parser.add_argument("--html-dir", default=None, help="Directory for per-item HTML trace views.")
    parser.add_argument("--run-tag", default=None, help="Run tag for versioned artifacts. Default: YYYYMMDD-HHMMSS.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat count for each item to compute pass_rate@N. Default: 1.",
    )
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument("--enable-quality-gate", action="store_true", help="Enable quality gate exit blocking for this run.")
    gate_group.add_argument("--disable-quality-gate", action="store_true", help="Disable quality gate exit blocking for this run.")
    parser.add_argument(
        "--fail-on-missing-expected-source",
        action="store_true",
        help="Fail early when expected sources in golden set are missing from indexed KB sources.",
    )
    return parser


def main() -> None:
    _configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()

    checker = EvalChecker(config=Config(), db_path=args.db_path)
    dataset_name = Path(args.dataset).stem
    report_dir = Path(args.report_dir or checker.config.evaluation.report_dir)
    trace_dir = Path(args.trace_dir or checker.config.evaluation.trace_dir)
    html_dir = Path(args.html_dir or checker.config.evaluation.html_dir)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = args.output or str(report_dir / dataset_name / run_tag / "eval_report.json")
    if args.enable_quality_gate:
        quality_gate_enabled = True
    elif args.disable_quality_gate:
        quality_gate_enabled = False
    else:
        quality_gate_enabled = checker.config.evaluation.quality_gate_enabled

    items = checker.load_golden_set(args.dataset)
    source_validation = checker.validate_expected_sources(items)
    if source_validation.get("missing_count", 0) > 0:
        logger.warning(
            "Detected %s missing expected sources (rate=%s).",
            source_validation["missing_count"],
            f"{source_validation['missing_rate']:.2%}",
        )
        if args.fail_on_missing_expected_source:
            logger.error("Aborting due to missing expected sources. Use source normalization first.")
            sys.exit(2)

    summary, results, traces = checker.evaluate_all(items, repeat=args.repeat)
    trace_manifest = checker.write_trace_artifacts(dataset_name, run_tag, traces, trace_dir)
    diff_manifest = checker.write_diff_artifacts(dataset_name, run_tag, traces, trace_dir, html_dir)
    checker.attach_artifact_paths(results, trace_manifest, {"items": {}}, diff_manifest)
    html_manifest = checker.write_html_artifacts(dataset_name, run_tag, results, traces, html_dir)
    checker.attach_artifact_paths(results, trace_manifest, html_manifest, diff_manifest)
    report_manifest = checker.write_report_artifacts(dataset_name, run_tag, Path(output_path), report_dir)
    checker.generate_report(
        summary,
        results,
        output_path,
        quality_gate_enabled,
        dataset_name,
        run_tag,
        trace_manifest,
        html_manifest,
        diff_manifest,
        report_manifest,
        source_validation=source_validation,
    )
    checker.print_summary(summary, quality_gate_enabled)

    quality_gate_failed = checker._quality_gate_failed(summary)
    sys.exit(1 if quality_gate_enabled and quality_gate_failed else 0)


if __name__ == "__main__":
    main()


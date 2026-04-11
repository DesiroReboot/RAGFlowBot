from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_REGISTRY = PROJECT_ROOT / "Eval" / "pipeline" / "golden-set.pipeline.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run golden-set evaluation pipeline (single or multi-set).")
    parser.add_argument("--dataset", default="Eval/golden-set.json", help="Single dataset path (backward-compatible).")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset paths. If provided, overrides --dataset.",
    )
    parser.add_argument(
        "--set-ids",
        default="",
        help="Comma-separated set ids, e.g. 0,1,2. Will resolve to <set-prefix><id><set-suffix>.",
    )
    parser.add_argument("--set-prefix", default="Eval/golden-set-", help="Prefix for --set-ids.")
    parser.add_argument("--set-suffix", default=".json", help="Suffix for --set-ids.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat count per item.")
    parser.add_argument("--run-tag", default=None, help="Optional batch run tag (default: YYYYMMDD-HHMMSS).")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining sets if one set fails.")
    parser.add_argument("--dry-run", action="store_true", help="Only print resolved sets, do not execute eval.")
    parser.add_argument(
        "--enable-quality-gate",
        action="store_true",
        help="Enable quality-gate exit behavior (disabled by default in this pipeline).",
    )
    return parser


def _print_summary(summary: dict[str, Any], dataset_name: str) -> None:
    print(f"Golden pipeline summary [{dataset_name}]")
    print(f"total_items={summary.get('total_items', 0)}")
    print(f"overall_score={summary.get('overall_score', 0.0):.4f}")
    print(f"overall_score_v2={summary.get('overall_score_v2', 0.0):.4f}")
    print(f"grade={summary.get('grade', 'N/A')}")
    print(f"must_source_recall_avg={summary.get('must_source_recall_avg', 0.0):.4f}")
    print(f"strict_citation_rate={summary.get('strict_citation_rate', 0.0):.4f}")
    print(f"must_keyword_coverage_avg={summary.get('must_keyword_coverage_avg', 0.0):.4f}")
    print(f"generation_quality_score_avg={summary.get('generation_quality_score_avg', 0.0):.4f}")
    print(f"latency_avg={summary.get('latency_avg', 0.0):.4f}")


def _parse_csv_tokens(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return _parse_csv_tokens(args.datasets)
    if args.set_ids:
        ids = _parse_csv_tokens(args.set_ids)
        return [f"{args.set_prefix}{set_id}{args.set_suffix}" for set_id in ids]
    return [args.dataset]


def _extract_artifact_indexes(report: dict[str, Any]) -> dict[str, str]:
    artifacts = report.get("artifacts", {})
    trace_artifacts = artifacts.get("trace", {}) if isinstance(artifacts.get("trace"), dict) else {}
    html_artifacts = artifacts.get("html", {}) if isinstance(artifacts.get("html"), dict) else {}
    diff_artifacts = artifacts.get("diff", {}) if isinstance(artifacts.get("diff"), dict) else {}

    trace_index_file = str(trace_artifacts.get("index_file", "")).strip()
    if not trace_index_file:
        trace_dir = str(trace_artifacts.get("dir", "")).strip()
        if trace_dir:
            trace_index_file = str((Path(trace_dir) / "index.json").resolve())

    html_index_file = str(html_artifacts.get("index_file", "")).strip()
    if not html_index_file:
        html_dir = str(html_artifacts.get("dir", "")).strip()
        if html_dir:
            html_index_file = str((Path(html_dir) / "index.html").resolve())

    diff_index_file = str(diff_artifacts.get("index_file", "")).strip()
    if not diff_index_file:
        diff_index_file = str(diff_artifacts.get("compare_index_file", "")).strip()

    return {
        "trace_index_file": trace_index_file,
        "html_index_file": html_index_file,
        "diff_index_file": diff_index_file,
    }


def _append_registry_history(batch_tag: str, set_runs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    registry = _load_json(PIPELINE_REGISTRY)
    history = registry.get("history", [])
    for run in set_runs:
        history.append(
            {
                "run_tag": run["set_run_tag"],
                "batch_run_tag": batch_tag,
                "dataset": run["dataset"],
                "repeat": max(1, args.repeat),
                "quality_gate_enabled": bool(args.enable_quality_gate),
                "exit_code": int(run["exit_code"]),
                "report_file": run["report_file"],
                "trace_index_file": run["trace_index_file"],
                "html_index_file": run["html_index_file"],
                "diff_index_file": run["diff_index_file"],
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    batch_history = registry.get("batch_history", [])
    batch_history.append(
        {
            "batch_run_tag": batch_tag,
            "datasets": [run["dataset"] for run in set_runs],
            "repeat": max(1, args.repeat),
            "quality_gate_enabled": bool(args.enable_quality_gate),
            "set_count": len(set_runs),
            "failed_sets": [run["dataset"] for run in set_runs if int(run.get("exit_code", 1)) != 0],
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    registry.update(
        {
            "name": "golden-set",
            "entrypoint": "python scripts/run_golden_pipeline.py",
            "dataset": "Eval/golden-set.json",
            "default_repeat": 1,
            "default_quality_gate_enabled": False,
            "latest_run_tag": batch_tag,
            "latest_report_file": set_runs[-1]["report_file"] if set_runs else "",
            "history": history[-100:],
            "batch_history": batch_history[-30:],
        }
    )
    _save_json(PIPELINE_REGISTRY, registry)


def _build_stage_summary_payload(batch_tag: str, set_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": "golden-set",
        "batch_run_tag": batch_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "set_count": len(set_runs),
        "sets": set_runs,
    }


def _run_one_set(
    dataset_path: str,
    set_run_tag: str,
    repeat: int,
    enable_quality_gate: bool,
) -> dict[str, Any]:
    dataset_name = Path(dataset_path).stem
    report_path = PROJECT_ROOT / "Eval" / "report" / dataset_name / set_run_tag / "eval_report.json"

    cmd = [
        sys.executable,
        "scripts/eval_runner.py",
        "--dataset",
        dataset_path,
        "--repeat",
        str(max(1, repeat)),
        "--run-tag",
        set_run_tag,
        "--output",
        str(report_path),
    ]
    if enable_quality_gate:
        cmd.append("--enable-quality-gate")
    else:
        cmd.append("--disable-quality-gate")

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, env=env)

    if not report_path.exists():
        return {
            "dataset": dataset_path,
            "dataset_name": dataset_name,
            "set_run_tag": set_run_tag,
            "exit_code": int(completed.returncode or 1),
            "report_file": str(report_path.resolve()),
            "trace_index_file": "",
            "html_index_file": "",
            "diff_index_file": "",
            "summary": {},
            "status": "missing_report",
        }

    report = _load_json(report_path)
    summary = report.get("summary", {})
    artifact_indexes = _extract_artifact_indexes(report)

    return {
        "dataset": dataset_path,
        "dataset_name": dataset_name,
        "set_run_tag": set_run_tag,
        "exit_code": int(completed.returncode),
        "report_file": str(report_path.resolve()),
        "trace_index_file": artifact_indexes["trace_index_file"],
        "html_index_file": artifact_indexes["html_index_file"],
        "diff_index_file": artifact_indexes["diff_index_file"],
        "summary": summary,
        "status": "ok" if int(completed.returncode) == 0 else "failed",
    }


def main() -> int:
    args = build_parser().parse_args()
    batch_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    datasets = _resolve_datasets(args)
    if not datasets:
        print("No datasets resolved.")
        return 1

    if args.dry_run:
        print("Dry run: resolved datasets")
        for idx, dataset in enumerate(datasets):
            print(f"  [{idx}] {dataset}")
        return 0

    set_runs: list[dict[str, Any]] = []
    has_failure = False

    for idx, dataset_path in enumerate(datasets):
        set_run_tag = f"{batch_tag}-s{idx}"
        print(f"\n=== Running set {idx + 1}/{len(datasets)}: {dataset_path} (run_tag={set_run_tag}) ===")
        set_result = _run_one_set(
            dataset_path=dataset_path,
            set_run_tag=set_run_tag,
            repeat=max(1, args.repeat),
            enable_quality_gate=bool(args.enable_quality_gate),
        )
        set_runs.append(set_result)

        _print_summary(set_result.get("summary", {}), set_result.get("dataset_name", Path(dataset_path).stem))
        print("Stage artifacts")
        print(f"report={set_result.get('report_file', '')}")
        print(f"trace_index={set_result.get('trace_index_file', '')}")
        print(f"html_index={set_result.get('html_index_file', '')}")
        print(f"diff_index={set_result.get('diff_index_file', '')}")
        stage_summary = _build_stage_summary_payload(batch_tag, set_runs)
        print("stage_summary=" + json.dumps(stage_summary, ensure_ascii=False))

        if int(set_result.get("exit_code", 1)) != 0:
            has_failure = True
            if not args.continue_on_error:
                print("Stopping due to set failure. Use --continue-on-error to run remaining sets.")
                break

    _append_registry_history(batch_tag, set_runs, args)
    final_stage_summary = _build_stage_summary_payload(batch_tag, set_runs)
    print("\nBatch stage summary (dialog only)")
    print(json.dumps(final_stage_summary, ensure_ascii=False, indent=2))
    print(f"sets_finished={len(set_runs)} sets_failed={sum(1 for run in set_runs if int(run.get('exit_code', 1)) != 0)}")
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

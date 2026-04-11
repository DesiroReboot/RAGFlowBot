from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("Eval/golden-set.json")
DEFAULT_OUTPUT_DIR = Path("Eval")
DEFAULT_PREFIX = "golden-set"
DEFAULT_SHARDS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stratified split for golden-set dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to source golden-set json.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for shard json files.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Output file prefix.")
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS, help="Number of shards.")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level JSON object.")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Expected 'items' field as list.")
    return payload


def _stratify(items: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        task_type = str(item.get("task_type", "unknown"))
        answerability = str(item.get("answerability", "unknown"))
        key = (task_type, answerability)
        buckets.setdefault(key, []).append(item)
    return buckets


def _split_evenly(values: list[dict[str, Any]], shards: int) -> list[list[dict[str, Any]]]:
    size = len(values)
    base = size // shards
    remainder = size % shards
    parts: list[list[dict[str, Any]]] = []
    cursor = 0
    for i in range(shards):
        take = base + (1 if i < remainder else 0)
        parts.append(values[cursor: cursor + take])
        cursor += take
    return parts


def _format_distribution(items: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        key = f"{item.get('task_type', 'unknown')}|{item.get('answerability', 'unknown')}"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    shards = max(1, int(args.shards))

    payload = _load_json(input_path)
    items = payload.get("items", [])
    if not items:
        raise ValueError("Source dataset has no items.")

    typed_items: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not an object.")
        typed_items.append(item)

    buckets = _stratify(typed_items)
    shard_items: list[list[dict[str, Any]]] = [[] for _ in range(shards)]
    for _, bucket_items in sorted(buckets.items(), key=lambda x: x[0]):
        parts = _split_evenly(bucket_items, shards)
        for shard_idx, part in enumerate(parts):
            shard_items[shard_idx].extend(part)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_meta = {k: v for k, v in payload.items() if k != "items"}
    for shard_idx in range(shards):
        shard_payload = dict(base_meta)
        shard_payload["dataset_name"] = f"{base_meta.get('dataset_name', 'golden-set')} [shard {shard_idx}]"
        shard_payload["items"] = shard_items[shard_idx]
        shard_payload["shard"] = {
            "index": shard_idx,
            "total_shards": shards,
            "source_dataset": str(input_path.as_posix()),
            "distribution": _format_distribution(shard_items[shard_idx]),
        }
        out_file = output_dir / f"{args.prefix}-{shard_idx}.json"
        out_file.write_text(json.dumps(shard_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for shard_idx in range(shards):
        distribution = _format_distribution(shard_items[shard_idx])
        print(f"shard={shard_idx} size={len(shard_items[shard_idx])} distribution={distribution}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

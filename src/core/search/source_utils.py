from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_HASH_SUFFIX_RE = re.compile(r"[-_][0-9a-f]{8,}$", re.IGNORECASE)
_TAIL_VERSION_RE = re.compile(r"[-_ ](?:v(?:er(?:sion)?)?\s*)?\d+$", re.IGNORECASE)
_SEPARATORS_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)


def source_label(source: str, source_path: str = "") -> str:
    if str(source).strip():
        return str(source).strip()
    if str(source_path).strip():
        return Path(str(source_path).strip()).name
    return ""


def canonical_source_id(source: str, source_path: str = "") -> str:
    label = source_label(source, source_path)
    if not label:
        return ""
    stem = Path(label).stem.strip().lower()
    stem = _HASH_SUFFIX_RE.sub("", stem)
    stem = _TAIL_VERSION_RE.sub("", stem)

    # Keep the primary CJK topic before conjunctions: 与/和/及.
    stem = re.sub(r"([\u4e00-\u9fff]{2,})[\u4e0e\u548c\u53ca][\u4e00-\u9fff]{1,}", r"\1", stem)

    stem = _SEPARATORS_RE.sub(" ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or Path(label).stem.strip().lower()


def build_grouped_citations(items: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if isinstance(item, dict):
            source = str(item.get("source", "")).strip()
            path = str(item.get("source_path", "")).strip()
            score = float(item.get("score", 0.0))
            doc_id = str(item.get("file_uuid", "")).strip()
            chunk_id = int(item.get("chunk_id", 0) or 0)
            text = str(item.get("content", "")).strip()
            canonical = str(item.get("canonical_source_id", "")).strip() or canonical_source_id(
                source,
                path,
            )
        else:
            source = str(getattr(item, "source", "")).strip()
            path = str(getattr(item, "source_path", "")).strip()
            score = float(getattr(item, "score", 0.0))
            doc_id = str(getattr(item, "file_uuid", "")).strip()
            chunk_id = int(getattr(item, "chunk_id", 0) or 0)
            text = str(getattr(item, "content", "")).strip()
            canonical = canonical_source_id(source, path)
        if not source:
            continue
        canonical = canonical or source.lower()
        grouped.setdefault(canonical, [])
        if not any(row["source"] == source and row["path"] == path for row in grouped[canonical]):
            grouped[canonical].append(
                {
                    "source": source,
                    "path": path,
                    "score": score,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )

    citations: list[dict[str, Any]] = []
    for canonical, versions in grouped.items():
        versions.sort(key=lambda row: row["score"], reverse=True)
        primary = versions[0]
        aliases = [row["source"] for row in versions[1:]]
        citations.append(
            {
                "source": primary["source"],
                "title": primary["source"],
                "path": primary["path"],
                "doc_id": primary.get("doc_id", ""),
                "chunk_id": primary.get("chunk_id", 0),
                "score": primary.get("score", 0.0),
                "text": primary.get("text", ""),
                "canonical_source_id": canonical,
                "aliases": aliases,
                "versions": [
                    {
                        "source": row["source"],
                        "path": row["path"],
                        "doc_id": row.get("doc_id", ""),
                        "chunk_id": row.get("chunk_id", 0),
                    }
                    for row in versions
                ],
                "_primary_score": primary["score"],
            }
        )

    citations.sort(
        key=lambda row: (float(row.get("_primary_score", 0.0)), len(row.get("versions", []))),
        reverse=True,
    )
    for row in citations:
        row.pop("_primary_score", None)
    return citations

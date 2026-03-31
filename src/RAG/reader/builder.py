from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, TypedDict, cast

from src.RAG.classification.classifier import Classifier
from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.indexing.indexer import Indexer
from src.RAG.preprocessing.parser import DocumentParser
from src.RAG.reader.scanner import FileScanner
from src.RAG.storage.file_mapper import FileMapper
from src.RAG.storage.manifest_store import ManifestStore


@dataclass(frozen=True)
class ParsedDocument:
    path: Path
    content: str
    doc_type: str
    parse_metadata: dict[str, Any]
    file_hash: str
    file_size: int


class BuilderSummary(TypedDict):
    processed: int
    updated: int
    skipped: int
    failed: int
    errors: list[str]
    indexed_files: int
    indexed_chunks: int
    partial_files: int
    chunks_dropped_total: int
    pdf_observability: list[dict[str, Any]]


class IndexResult(TypedDict):
    chunks_written: int
    vectors_written: int
    chunks_dropped: int
    index_status: str
    last_error: str | None


class KnowledgeBaseBuilder:
    def __init__(self, config: KBaseConfig):
        self.config = config
        self.scanner = FileScanner(config.supported_extensions)
        self.parser = DocumentParser(config)
        self.classifier = Classifier(config)
        self.indexer = Indexer(config.db_path, config)
        self.file_mapper = FileMapper(config.db_path, ensure_db_schema=False)
        self.manifest_store = ManifestStore(config.db_path, ensure_schema=False)

    def sync(self, source_dir: str | None = None, *, force_reindex: bool = False) -> dict[str, Any]:
        source_root = source_dir or self.config.source_dir
        files = self.scanner.scan(source_root)
        if not files:
            return self._finalize(
                {
                    "processed": 0,
                    "updated": 0,
                    "skipped": 0,
                    "failed": 0,
                    "errors": [f"source_dir_not_found_or_empty: {source_root}"],
                    "indexed_files": 0,
                    "indexed_chunks": 0,
                    "partial_files": 0,
                    "chunks_dropped_total": 0,
                    "pdf_observability": [],
                }
            )

        summary: BuilderSummary = {
            "processed": 0,
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
            "indexed_files": 0,
            "indexed_chunks": 0,
            "partial_files": 0,
            "chunks_dropped_total": 0,
            "pdf_observability": [],
        }
        for file_path in files:
            summary["processed"] += 1
            try:
                parsed = self._parse_file(file_path)
                file_uuid = self._stable_doc_uuid(str(file_path))
                existing = self.file_mapper.get_file_by_path(str(file_path))
                has_chunks = self.file_mapper.count_ready_chunks(file_uuid) > 0
                if (
                    not force_reindex
                    and existing
                    and str(existing.get("file_hash", "")) == parsed.file_hash
                    and has_chunks
                    and str(existing.get("index_status", "ready")) in {"ready", "partial"}
                ):
                    summary["skipped"] += 1
                    continue

                category = "uncategorized"
                confidence = 0.0
                if self.config.auto_classification:
                    category, confidence = self.classifier.classify(parsed.content[:6000])
                summary_text = self._summarize(parsed.content)
                parse_method = str(parsed.parse_metadata.get("parse_method", "ready"))
                parse_readability = float(parsed.parse_metadata.get("readability_score", 1.0))
                parse_noise = float(parsed.parse_metadata.get("noise_ratio", 0.0))
                summary_with_quality = (
                    f"{summary_text} (cls_conf={confidence:.2f}; parse={parse_method}; "
                    f"readability={parse_readability:.2f}; noise={parse_noise:.2f})"
                )

                self.file_mapper.save_file(
                    uuid=file_uuid,
                    filename=file_path.name,
                    filepath=str(file_path),
                    category=category,
                    summary=summary_with_quality.strip(),
                    file_hash=parsed.file_hash,
                    file_size=parsed.file_size,
                    doc_type=parsed.doc_type,
                    parse_status=parse_method,
                    index_status="indexing",
                )

                index_result = cast(
                    IndexResult,
                    self.indexer.index_document(
                        file_uuid=file_uuid,
                        content=parsed.content,
                        source=file_path.name,
                        source_path=str(file_path),
                        section_title="",
                        doc_type=parsed.doc_type,
                    ),
                )
                summary["updated"] += 1
                summary["indexed_chunks"] += index_result["chunks_written"]
                summary["chunks_dropped_total"] += int(index_result.get("chunks_dropped", 0))
                if parsed.doc_type == "pdf":
                    summary["pdf_observability"].append(
                        {
                            "source": file_path.name,
                            "parse_method": parse_method,
                            "readability_score": round(parse_readability, 6),
                            "noise_ratio": round(parse_noise, 6),
                            "chunks_written": int(index_result.get("chunks_written", 0)),
                            "chunks_dropped": int(index_result.get("chunks_dropped", 0)),
                        }
                    )
                if str(index_result["index_status"]) == "partial":
                    summary["partial_files"] += 1
                self.file_mapper.update_index_status(
                    file_uuid,
                    str(index_result["index_status"]),
                    index_result.get("last_error"),
                )
            except Exception as exc:
                summary["failed"] += 1
                summary["errors"].append(f"{file_path}: {exc}")

        return self._finalize(summary)

    def _parse_file(self, file_path: Path) -> ParsedDocument:
        content, metadata = self.parser.parse(file_path)
        if not content.strip():
            content = file_path.name
        raw = file_path.read_bytes() if file_path.exists() else content.encode("utf-8")
        file_hash = hashlib.sha256(raw).hexdigest()
        file_size = len(raw)
        return ParsedDocument(
            path=file_path,
            content=content,
            doc_type=str(metadata.get("type", "text")),
            parse_metadata=metadata,
            file_hash=file_hash,
            file_size=file_size,
        )

    def _stable_doc_uuid(self, file_path: str) -> str:
        normalized = file_path.replace("\\", "/").strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _summarize(self, content: str, max_chars: int = 140) -> str:
        compact = " ".join(content.strip().split())
        if len(compact) <= max_chars:
            return compact
        return f"{compact[:max_chars].rstrip()}..."

    def _finalize(self, summary: BuilderSummary) -> dict[str, Any]:
        stats = self.indexer.get_index_stats()
        summary["indexed_files"] = stats["indexed_files"]
        summary["indexed_chunks"] = stats["indexed_chunks"]
        has_retrieval_rows = any(
            int(stats.get(key, 0) or 0) > 0
            for key in ("indexed_chunks", "fts_documents", "vec_rows")
        )
        if not has_retrieval_rows:
            # Prevent false "ready" when the source dir is missing or all chunks are dropped.
            attempted = (
                int(summary.get("processed", 0) or 0) > 0
                or int(summary.get("updated", 0) or 0) > 0
                or int(summary.get("failed", 0) or 0) > 0
                or bool(summary.get("errors", []))
            )
            status = "failed" if attempted else "empty"
        else:
            status = "ready"
            if summary["failed"] > 0:
                status = "failed" if summary["updated"] == 0 else "partial"
            elif summary["partial_files"] > 0:
                status = "partial"

        last_error = summary["errors"][-1] if summary["errors"] else None
        self.manifest_store.upsert_manifest(
            status=status,
            embedding_provider=self.indexer.embedding.provider,
            embedding_model=self.indexer.embedding.model_name,
            embedding_dimension=self.indexer.embedding.dimension,
            build_version=self.config.build_version,
            indexed_files=summary["indexed_files"],
            indexed_chunks=summary["indexed_chunks"],
            partial_files=summary["partial_files"],
            last_error=last_error,
        )
        return dict(summary)

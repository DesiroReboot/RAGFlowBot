from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.RAG.classification.classifier import Classifier
from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.indexing.indexer import Indexer
from src.RAG.preprocessing.parser import DocumentParser
from src.RAG.reader.builder import KnowledgeBaseBuilder
from src.RAG.storage.conflict_resolver import ConflictResolver
from src.RAG.storage.file_mapper import FileMapper
from src.RAG.storage.manifest_store import ManifestStore


class KBaseManager:
    def __init__(self, config: KBaseConfig | None = None):
        self.config = config or KBaseConfig()
        self.file_mapper = FileMapper(self.config.db_path)
        self.manifest_store = ManifestStore(self.config.db_path, ensure_schema=False)
        self.conflict_resolver = ConflictResolver(self.config.db_path, ensure_db_schema=False)
        self.classifier = Classifier(self.config)
        self.parser = DocumentParser(self.config)
        self.indexer = Indexer(self.config.db_path, self.config)
        self.builder = KnowledgeBaseBuilder(self.config)

        if self.config.auto_sync_on_startup:
            self.sync_configured_source()

    @staticmethod
    def generate_file_uuid(filepath: str, file_hash: str) -> str:
        normalized = filepath.replace("\\", "/").strip().lower()
        raw = f"{normalized}|{file_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def sync_configured_source(self, *, force_reindex: bool = False) -> dict[str, Any]:
        return self.builder.sync(self.config.source_dir, force_reindex=force_reindex)

    def scan_and_process(self, source_dir: str, *, force_reindex: bool = False) -> dict[str, Any]:
        if not Path(source_dir).exists():
            return {
                "processed": 0,
                "updated": 0,
                "skipped": 0,
                "failed": 0,
                "errors": [f"directory_not_found: {source_dir}"],
                "indexed_files": self.indexer.get_index_stats()["indexed_files"],
                "indexed_chunks": self.indexer.get_index_stats()["indexed_chunks"],
                "partial_files": 0,
            }
        return self.builder.sync(source_dir, force_reindex=force_reindex)

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.indexer.search(query, limit)

    def extract_content(self, file_uuid: str) -> dict[str, Any]:
        file_info = self.file_mapper.get_file(file_uuid)
        if not file_info:
            return {"file_uuid": file_uuid, "content": "", "chunks": []}
        rows = self.file_mapper.execute(
            """
            SELECT chunk_id, content, section_title
            FROM chunks
            WHERE file_uuid = ?
            ORDER BY chunk_id
            """,
            (file_uuid,),
        )
        content = "\n".join(str(row["content"]) for row in rows)
        return {
            "file_uuid": file_uuid,
            "source": file_info.get("filename", ""),
            "source_path": file_info.get("filepath", ""),
            "content": content,
            "chunks": rows,
        }

    def classify_document(self, file_uuid: str) -> tuple[str, float]:
        extracted = self.extract_content(file_uuid)
        category, confidence = self.classifier.classify(str(extracted.get("content", "")))
        self.file_mapper.update_category(file_uuid, category)
        return category, confidence

    def get_statistics(self) -> dict[str, Any]:
        index_stats = self.indexer.get_index_stats()
        conflict_stats = self.conflict_resolver.get_conflict_stats()
        category_distribution = self.file_mapper.count_by_category()
        manifest = self.manifest_store.get_manifest() or {}
        return {
            "total_files": index_stats["indexed_files"],
            "total_chunks": index_stats["indexed_chunks"],
            "fts_documents": index_stats["fts_documents"],
            "vec_rows": index_stats["vec_rows"],
            "category_distribution": category_distribution,
            "conflicts": conflict_stats,
            "manifest": manifest,
        }

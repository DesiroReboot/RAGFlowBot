from __future__ import annotations

import sqlite3
from typing import Any

from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.reader.embedding_client import EmbeddingClient
from src.KB.manifest_store import ManifestStore


class VecRetriever:
    def __init__(self, db_path: str, config: KBaseConfig):
        self.db_path = db_path
        self.config = config
        self.embedding_client = EmbeddingClient(config)
        self.manifest_store = ManifestStore(db_path, ensure_schema=False)

    def retrieve(self, query: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        manifest = self.manifest_store.get_manifest()
        manifest_warning = None
        if manifest:
            manifest_model = str(manifest.get("embedding_model", ""))
            manifest_dimension = int(manifest.get("embedding_dimension", 0))
            if manifest_model and manifest_model != self.embedding_client.model_name:
                raise RuntimeError(
                    f"embedding model mismatch: query={self.embedding_client.model_name}, index={manifest_model}"
                )
            if manifest_dimension and manifest_dimension != self.embedding_client.dimension:
                raise RuntimeError(
                    f"embedding dimension mismatch: query={self.embedding_client.dimension}, index={manifest_dimension}"
                )
        else:
            manifest_warning = "manifest_missing"

        query_vector = self.embedding_client.embed_text(query)
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            try:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                    )
                }
                if "vec_index" not in tables:
                    return [], {"manifest_warning": manifest_warning, "candidate_pool": 0}
                vec_columns = {row[1] for row in conn.execute("PRAGMA table_info(vec_index)")}
                file_columns = (
                    {row[1] for row in conn.execute("PRAGMA table_info(files)")}
                    if "files" in tables
                    else set()
                )
                has_chunks = "chunks" in tables
                joins = []
                source_expr = "NULL"
                source_path_expr = "NULL"
                section_expr = "NULL"
                doc_type_expr = "NULL"
                content_expr = "NULL"
                doc_chunk_count_expr = "1"
                if has_chunks:
                    joins.append(
                        """
                        LEFT JOIN chunks
                            ON chunks.file_uuid = vec_index.file_uuid
                           AND chunks.chunk_id = vec_index.chunk_id
                        """
                    )
                    joins.append(
                        """
                        LEFT JOIN (
                            SELECT file_uuid, COUNT(*) AS doc_chunk_count
                            FROM chunks
                            GROUP BY file_uuid
                        ) chunk_stats ON chunk_stats.file_uuid = vec_index.file_uuid
                        """
                    )
                    source_expr = "chunks.source_filename"
                    source_path_expr = "chunks.source_path"
                    section_expr = "chunks.section_title"
                    doc_type_expr = "chunks.doc_type"
                    content_expr = "chunks.content"
                    doc_chunk_count_expr = "COALESCE(chunk_stats.doc_chunk_count, 1)"
                if "source" in vec_columns:
                    source_expr = f"COALESCE({source_expr}, vec_index.source)"
                if "files" in tables:
                    joins.append("LEFT JOIN files ON files.uuid = vec_index.file_uuid")
                    if "filename" in file_columns:
                        source_expr = f"COALESCE({source_expr}, files.filename, '')"
                    if "filepath" in file_columns:
                        source_path_expr = f"COALESCE({source_path_expr}, files.filepath, '')"
                    if "doc_type" in file_columns:
                        doc_type_expr = f"COALESCE({doc_type_expr}, files.doc_type, 'text')"
                if "filename" not in file_columns:
                    source_expr = f"COALESCE({source_expr}, '')"
                if "filepath" not in file_columns:
                    source_path_expr = f"COALESCE({source_path_expr}, '')"
                if "doc_type" not in file_columns:
                    doc_type_expr = f"COALESCE({doc_type_expr}, 'text')"
                section_expr = f"COALESCE({section_expr}, '')"
                content_expr = f"COALESCE({content_expr}, '')"

                joins_clause = " ".join(joins)
                sql_template = """
                    SELECT
                        vec_index.file_uuid AS file_uuid,
                        vec_index.chunk_id AS chunk_id,
                        __SOURCE_EXPR__ AS source,
                        __SOURCE_PATH_EXPR__ AS source_path,
                        __SECTION_EXPR__ AS section_title,
                        __DOC_TYPE_EXPR__ AS doc_type,
                        __DOC_CHUNK_COUNT_EXPR__ AS doc_chunk_count,
                        __CONTENT_EXPR__ AS content,
                        vec_index.embedding AS embedding
                    FROM vec_index
                    __JOINS__
                """
                sql = (
                    sql_template.replace("__SOURCE_EXPR__", source_expr)
                    .replace("__SOURCE_PATH_EXPR__", source_path_expr)
                    .replace("__SECTION_EXPR__", section_expr)
                    .replace("__DOC_TYPE_EXPR__", doc_type_expr)
                    .replace("__DOC_CHUNK_COUNT_EXPR__", doc_chunk_count_expr)
                    .replace("__CONTENT_EXPR__", content_expr)
                    .replace("__JOINS__", joins_clause)
                )
                rows = conn.execute(
                    sql
                ).fetchall()
            except sqlite3.OperationalError:
                return [], {"manifest_warning": manifest_warning, "candidate_pool": 0}

        scored: list[dict[str, Any]] = []
        for row in rows:
            vector = self.embedding_client.deserialize(row["embedding"])
            similarity = self.embedding_client.cosine_similarity(query_vector, vector)
            scored.append(
                {
                    "file_uuid": row["file_uuid"],
                    "chunk_id": int(row["chunk_id"]),
                    "source": row["source"],
                    "source_path": row["source_path"],
                    "section_title": row["section_title"],
                    "doc_type": row["doc_type"],
                    "doc_chunk_count": int(row["doc_chunk_count"]),
                    "content": row["content"],
                    "vec_similarity": float(similarity),
                }
            )

        scored.sort(key=lambda item: item["vec_similarity"], reverse=True)
        results = []
        for rank, item in enumerate(scored[:limit], start=1):
            results.append({**item, "vec_rank": rank})
        return results, {"manifest_warning": manifest_warning, "candidate_pool": len(scored)}

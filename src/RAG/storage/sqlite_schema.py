from __future__ import annotations

import sqlite3


DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS files (
        uuid TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL UNIQUE,
        category TEXT DEFAULT 'uncategorized',
        summary TEXT DEFAULT '',
        file_hash TEXT NOT NULL,
        file_size INTEGER NOT NULL DEFAULT 0,
        doc_type TEXT NOT NULL DEFAULT 'text',
        parse_status TEXT NOT NULL DEFAULT 'ready',
        index_status TEXT NOT NULL DEFAULT 'ready',
        last_error TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_scanned_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_uuid TEXT NOT NULL,
        chunk_id INTEGER NOT NULL,
        source_filename TEXT NOT NULL,
        source_path TEXT NOT NULL,
        section_title TEXT DEFAULT '',
        doc_type TEXT NOT NULL DEFAULT 'text',
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(file_uuid) REFERENCES files(uuid) ON DELETE CASCADE,
        UNIQUE(file_uuid, chunk_id)
    );
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
        content,
        source UNINDEXED,
        section_title UNINDEXED,
        file_uuid UNINDEXED,
        chunk_id UNINDEXED,
        tokenize='unicode61'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS vec_index (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_uuid TEXT NOT NULL,
        chunk_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        source TEXT DEFAULT '',
        source_path TEXT DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(file_uuid) REFERENCES files(uuid) ON DELETE CASCADE,
        UNIQUE(file_uuid, chunk_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS index_manifest (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        status TEXT NOT NULL DEFAULT 'empty',
        embedding_provider TEXT DEFAULT '',
        embedding_model TEXT DEFAULT '',
        embedding_dimension INTEGER DEFAULT 0,
        build_version TEXT DEFAULT '',
        indexed_files INTEGER NOT NULL DEFAULT 0,
        indexed_chunks INTEGER NOT NULL DEFAULT 0,
        partial_files INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        source_file_count INTEGER DEFAULT 0,
        source_max_mtime TEXT DEFAULT '',
        source_scanned_at TEXT DEFAULT '',
        last_index_files INTEGER DEFAULT 0,
        last_index_run_id TEXT DEFAULT ''
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS conflicts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        conflicting_sources TEXT,
        resolution_status TEXT NOT NULL DEFAULT 'detected',
        resolution_note TEXT,
        priority TEXT NOT NULL DEFAULT 'medium',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        resolved_at TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_files_filepath ON files(filepath);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_file_uuid ON chunks(file_uuid);",
    "CREATE INDEX IF NOT EXISTS idx_vec_file_uuid ON vec_index(file_uuid);",
]


def ensure_schema(conn: sqlite3.Connection) -> None:
    for sql in DDL:
        conn.execute(sql)
    conn.execute(
        """
        INSERT INTO index_manifest (id, status)
        VALUES (1, 'empty')
        ON CONFLICT(id) DO NOTHING
        """
    )
    _migrate_index_manifest(conn)
    conn.commit()


def _migrate_index_manifest(conn: sqlite3.Connection) -> None:
    """Migrate existing index_manifest table to add new columns."""
    try:
        cursor = conn.execute("PRAGMA table_info(index_manifest)")
        columns = {row[1] for row in cursor.fetchall()}

        new_columns = {
            "source_file_count": "INTEGER DEFAULT 0",
            "source_max_mtime": "TEXT DEFAULT ''",
            "source_scanned_at": "TEXT DEFAULT ''",
            "last_index_files": "INTEGER DEFAULT 0",
            "last_index_run_id": "TEXT DEFAULT ''",
        }

        for col_name, col_def in new_columns.items():
            if col_name not in columns:
                conn.execute(f"ALTER TABLE index_manifest ADD COLUMN {col_name} {col_def}")
    except sqlite3.OperationalError:
        # Table might not exist yet, will be created by DDL
        pass

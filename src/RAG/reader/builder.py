from __future__ import annotations

# Backward compatibility shim - re-export from KB module
# NOTE: This file is kept for backward compatibility only.
# New code should import from src.KB.builder directly.

def __getattr__(name: str):
    if name in ("KnowledgeBaseBuilder", "ParsedDocument", "BuilderSummary", "IndexResult"):
        from src.KB.builder import (
            KnowledgeBaseBuilder,
            ParsedDocument,
            BuilderSummary,
            IndexResult,
        )
        if name == "KnowledgeBaseBuilder":
            return KnowledgeBaseBuilder
        elif name == "ParsedDocument":
            return ParsedDocument
        elif name == "BuilderSummary":
            return BuilderSummary
        elif name == "IndexResult":
            return IndexResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["KnowledgeBaseBuilder", "ParsedDocument", "BuilderSummary", "IndexResult"]

from __future__ import annotations

# Backward compatibility shim - re-export from KB module
# NOTE: This file is kept for backward compatibility only.
# New code should import from src.KB.manifest_store directly.

def __getattr__(name: str):
    if name == "ManifestStore":
        from src.KB.manifest_store import ManifestStore
        return ManifestStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ManifestStore"]

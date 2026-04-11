from __future__ import annotations

from dataclasses import dataclass, field
import json
from time import sleep
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError


@dataclass
class RAGFlowChunk:
    id: str
    document_id: str
    content: str
    score: float
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGFlowSearchResponse:
    chunks: list[RAGFlowChunk]
    rewritten_question: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class RAGFlowClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_ms: int = 2500,
        max_retries: int = 1,
        session: Any | None = None,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.timeout = max(0.2, int(timeout_ms) / 1000.0)
        self.max_retries = max(0, int(max_retries))
        self._session = session

    def search(
        self,
        *,
        dataset_id: str,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> RAGFlowSearchResponse:
        if not self.base_url:
            raise RuntimeError("provider_misconfigured:ragflow_base_url_missing")
        if not self.api_key or self.api_key.upper().startswith("YOUR_"):
            raise RuntimeError("provider_misconfigured:ragflow_api_key_missing")
        if not str(dataset_id or "").strip():
            raise RuntimeError("provider_misconfigured:ragflow_dataset_id_missing")

        url = self._safe_http_url(f"{self.base_url}/api/v1/retrieval/search")
        payload = {
            "dataset_id": str(dataset_id).strip(),
            "question": str(question or "").strip(),
            "top_k": max(1, int(top_k)),
        }
        if isinstance(filters, dict) and filters:
            payload["filters"] = dict(filters)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = self._post_json(url=url, payload=payload, headers=headers)
                chunks = self._extract_chunks(data)
                return RAGFlowSearchResponse(
                    chunks=chunks,
                    rewritten_question=str(
                        data.get("rewritten_question", "")
                        or data.get("query_rewrite", "")
                        or ""
                    ).strip(),
                    raw=data,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep(min(0.6 * (attempt + 1), 1.5))
                    continue
                break
        raise RuntimeError(f"ragflow_search_failed:{last_error}")

    def _extract_chunks(self, data: dict[str, Any]) -> list[RAGFlowChunk]:
        raw_chunks = data.get("chunks", [])
        if not isinstance(raw_chunks, list):
            raw_chunks = []

        rows: list[RAGFlowChunk] = []
        for item in raw_chunks:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("id", "") or item.get("chunk_id", "")).strip()
            doc_id = str(item.get("document_id", "") or item.get("doc_id", "")).strip()
            text = str(item.get("content", "") or item.get("text", "")).strip()
            if not text:
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            meta = item.get("metadata", {})
            rows.append(
                RAGFlowChunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content=text,
                    score=score,
                    title=str(item.get("title", "")).strip(),
                    source=str(item.get("source", "")).strip(),
                    metadata=dict(meta) if isinstance(meta, dict) else {},
                )
            )
        return rows

    def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        session = self._session
        if session is not None and hasattr(session, "post"):
            response = session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            status_code = int(getattr(response, "status_code", 200))
            if status_code >= 400:
                text = str(getattr(response, "text", "")).strip()
                raise RuntimeError(f"http_{status_code}:{text[:200]}")
            loaded = response.json() if callable(getattr(response, "json", None)) else {}
            return loaded if isinstance(loaded, dict) else {}

        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:  # nosec B310
                body = response.read().decode("utf-8", errors="ignore")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"http_{exc.code}:{body[:200]}") from exc

        if not body.strip():
            return {}
        loaded = json.loads(body)
        return loaded if isinstance(loaded, dict) else {}

    def _safe_http_url(self, raw_url: str) -> str:
        parsed = parse.urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("missing url host")
        return raw_url

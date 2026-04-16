from __future__ import annotations

from abc import ABC, abstractmethod
import json
from time import sleep
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError


class Reranker(ABC):
    provider: str = "noop"

    @abstractmethod
    def score(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        timeout_ms: int,
    ) -> list[float]:
        """Return per-candidate rerank raw scores aligned with candidates."""


class NoopReranker(Reranker):
    provider = "noop"

    def score(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        timeout_ms: int,
    ) -> list[float]:
        del query, timeout_ms
        return [float(item.get("score", 0.0)) for item in candidates]


class APIReranker(Reranker):
    provider = "api"

    def __init__(
        self,
        *,
        model: str = "gte-rerank-v2",
        base_url: str = "",
        api_key: str = "",
        timeout_ms: int = 800,
        max_retries: int = 1,
        session: Any | None = None,
    ) -> None:
        self.model = str(model or "gte-rerank-v2").strip() or "gte-rerank-v2"
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.timeout_ms = max(100, int(timeout_ms))
        self.max_retries = max(0, int(max_retries))
        self._session = session

    def score(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        timeout_ms: int,
    ) -> list[float]:
        if not candidates:
            return []
        if not self.base_url:
            raise RuntimeError("provider_misconfigured:rerank_base_url_missing")
        if not self.api_key or self.api_key.upper().startswith("YOUR_"):
            raise RuntimeError("provider_misconfigured:rerank_api_key_missing")

        url = self._safe_http_url(f"{self.base_url}/rerank")
        documents = [self._to_rerank_document(item) for item in candidates]
        payload = {
            "model": self.model,
            "query": str(query or "").strip(),
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        timeout_sec = max(1, int(timeout_ms / 1000))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post_json(
                    url=url,
                    payload=payload,
                    headers=headers,
                    timeout=timeout_sec,
                )
                return self._parse_scores(response=response, expected_count=len(documents))
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep(min(0.8 * (attempt + 1), 2.0))
                    continue
                break
        raise RuntimeError(f"api rerank failed: {last_error}")

    def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        session = self._session
        if session is not None and hasattr(session, "post"):
            response = session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            status_code = int(getattr(response, "status_code", 200))
            if status_code >= 400:
                text = str(getattr(response, "text", "")).strip()
                raise RuntimeError(f"http_{status_code}:{text[:200]}")
            body = ""
            json_method = getattr(response, "json", None)
            if callable(json_method):
                loaded = json_method()
                return loaded if isinstance(loaded, dict) else {}
            body = str(getattr(response, "text", "")).strip()
            if not body:
                return {}
            loaded = json.loads(body)
            return loaded if isinstance(loaded, dict) else {}

        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:  # nosec B310
                body = response.read().decode("utf-8", errors="ignore")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"http_{exc.code}:{body[:200]}") from exc
        if not body.strip():
            return {}
        loaded = json.loads(body)
        return loaded if isinstance(loaded, dict) else {}

    def _parse_scores(self, *, response: dict[str, Any], expected_count: int) -> list[float]:
        rows: Any = response.get("results")
        if not isinstance(rows, list):
            rows = response.get("data")
        if not isinstance(rows, list):
            output = response.get("output")
            if isinstance(output, dict):
                rows = output.get("results") or output.get("data")
        if not isinstance(rows, list):
            rows = []

        scores = [0.0 for _ in range(expected_count)]
        cursor = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            index = row.get("index", row.get("document_index"))
            score_raw = row.get("relevance_score", row.get("score", 0.0))
            try:
                value = float(score_raw or 0.0)
            except Exception:
                value = 0.0

            if isinstance(index, int) and 0 <= index < expected_count:
                scores[index] = value
                continue

            if cursor < expected_count:
                scores[cursor] = value
                cursor += 1

        if any(score != 0.0 for score in scores):
            return scores
        raise RuntimeError("api rerank response missing scores")

    def _to_rerank_document(self, item: dict[str, Any]) -> str:
        source = str(item.get("source", "")).strip()
        section_title = str(item.get("section_title", "")).strip()
        content = str(item.get("content", "")).strip()
        prefix = " | ".join(part for part in (source, section_title) if part)
        return f"{prefix}\n{content}".strip()

    def _safe_http_url(self, raw_url: str) -> str:
        parsed = parse.urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("missing url host")
        return raw_url


def build_reranker(
    provider: str,
    *,
    model: str = "gte-rerank-v2",
    base_url: str = "",
    api_key: str = "",
    timeout_ms: int = 800,
    max_retries: int = 1,
) -> Reranker:
    normalized = str(provider or "noop").strip().lower()
    if normalized == "noop":
        return NoopReranker()
    if normalized in {"api", "dashscope"}:
        return APIReranker(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_ms=timeout_ms,
            max_retries=max_retries,
        )
    # Unknown provider degrades to noop to preserve stability.
    return NoopReranker()

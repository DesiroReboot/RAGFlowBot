from __future__ import annotations

import pytest

from src.core.search.reranker import APIReranker


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict):
        self.payload = payload

    def post(self, url: str, json: dict, headers: dict, timeout: int):  # noqa: A002
        assert url.endswith("/rerank")
        assert "query" in json
        assert isinstance(json.get("documents"), list)
        assert headers.get("Authorization", "").startswith("Bearer ")
        assert timeout >= 1
        return _FakeResponse(payload=self.payload)


def test_api_reranker_parses_scores_by_index() -> None:
    reranker = APIReranker(
        model="gte-rerank-v2",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="test-key",
        session=_FakeSession(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            }
        ),
    )
    scores = reranker.score(
        query="test",
        candidates=[
            {"source": "a.md", "section_title": "s1", "content": "A"},
            {"source": "b.md", "section_title": "s2", "content": "B"},
        ],
        timeout_ms=800,
    )

    assert scores == [0.42, 0.91]


def test_api_reranker_requires_api_key() -> None:
    reranker = APIReranker(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="",
    )
    with pytest.raises(RuntimeError, match="rerank_api_key_missing"):
        reranker.score(query="q", candidates=[{"content": "x"}], timeout_ms=800)

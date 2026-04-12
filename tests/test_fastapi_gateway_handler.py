from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.config import Config
from src.fastapi_gateway.app import create_app
from src.fastapi_gateway.services import event_service as event_service_module


class _DummyAgent:
    def run_sync(
        self,
        query: str,
        include_trace: bool = False,  # noqa: ARG002
        *,
        run_id: str | None = None,  # noqa: ARG002
        memory_store: object | None = None,  # noqa: ARG002
    ):
        class _Resp:
            answer = "ok"
            retrieval_confidence = 1.0
            trace = {
                "search": {"fts_hits": 1, "vec_hits": 1},
                "retrieval_provider": "legacy",
                "final_citations": [{"source": "kb-a.md", "section": "s1"}],
            }

        if include_trace:
            return _Resp()
        return _Resp()


class _DummyAPIClient:
    def __init__(self) -> None:
        self._dialog = {"ok": True, "missing_tokens": [], "message": ""}
        self.reply_calls = 0
        self.message_calls = 0
        self.reply_texts: list[str] = []

    def token_dialog_payload(self):
        return self._dialog

    def openapi_base_url_ok(self):
        return True

    def validate_credentials(self):
        return {
            "ok": True,
            "error": "",
            "feishu_code": 0,
            "feishu_msg": "",
            "expire": 7200,
            "tenant_access_token_ready": True,
        }

    def send_reply_text(self, *, message_id: str, text: str):  # noqa: ARG002
        self.reply_calls += 1
        self.reply_texts.append(text)
        class _Resp:
            ok = True
            error = ""
            data = {}

        return _Resp()


class _CountingTokenDialogAPIClient(_DummyAPIClient):
    def __init__(self) -> None:
        super().__init__()
        self.token_dialog_calls = 0

    def token_dialog_payload(self):
        self.token_dialog_calls += 1
        return super().token_dialog_payload()

    def send_message_text(self, *, receive_id: str, text: str, receive_id_type: str = "chat_id"):  # noqa: ARG002
        self.message_calls += 1
        class _Resp:
            ok = True
            error = ""
            data = {}

        return _Resp()


def _build_config(tmp_path: Path, extra: dict | None = None) -> Config:
    payload = {
        "database": {
            "db_path": str(tmp_path / "gateway_test.db"),
        },
        "search": {
            "web_search_enabled": False,
            "search_progress_enabled": True,
            "search_progress_keyword_top_k": 4,
        },
        "gateway": {
            "feishu": {
                "enabled": True,
                "receive_mode": "webhook",
                "openapi_base_url": "https://open.feishu.cn/open-apis",
                "app_id": "app-id",
                "app_secret": "app-secret",
                "verification_token": "verify-token",
                "webhook_path": "/webhook/feishu",
            }
        }
    }
    if extra:
        if "gateway" in extra and "feishu" in extra["gateway"]:
            payload["gateway"]["feishu"].update(extra["gateway"]["feishu"])
        if "search" in extra and isinstance(extra["search"], dict):
            payload["search"].update(extra["search"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return Config(str(path))


def test_create_app_rejects_long_connection_mode(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path, extra={"gateway": {"feishu": {"receive_mode": "long_connection"}}})
    with pytest.raises(RuntimeError, match="receive_mode=long_connection"):
        create_app(cfg)


def test_health_and_startup_check(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    startup = client.get("/gateway/startup-check")
    assert startup.status_code == 200
    payload = startup.json()
    assert payload["webhook_path"] == "/webhook/feishu"
    assert payload["credential_validation"]["ok"] is True


def test_self_check_and_fullchain_visualize(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    self_check = client.get("/gateway/self-check")
    assert self_check.status_code == 200
    self_payload = self_check.json()
    assert self_payload["summary"]["total"] >= 1
    assert isinstance(self_payload["checks"], list)
    self_check_ts = datetime.fromisoformat(self_payload["timestamp"])
    assert self_check_ts.tzinfo is not None
    assert self_check_ts.utcoffset() == timedelta(0)

    fullchain = client.post("/gateway/fullchain-visualize", json={"query": "hello"})
    assert fullchain.status_code == 200
    chain_payload = fullchain.json()
    assert chain_payload["ok"] is True
    assert chain_payload["query"] == "hello"
    assert isinstance(chain_payload["stages"], list)
    fullchain_ts = datetime.fromisoformat(chain_payload["timestamp"])
    assert fullchain_ts.tzinfo is not None
    assert fullchain_ts.utcoffset() == timedelta(0)


def test_url_verification_token_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECBOT_FEISHU_VERIFICATION_TOKEN", "verify-token")
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    bad = client.post(
        "/webhook/feishu",
        json={"type": "url_verification", "challenge": "abc", "token": "bad-token"},
    )
    assert bad.status_code == 200
    assert bad.json()["success"] is False
    assert bad.json()["fallback_type"] == "verification_token_invalid"

    ok = client.post(
        "/webhook/feishu",
        json={"type": "url_verification", "challenge": "abc", "token": "verify-token"},
    )
    assert ok.status_code == 200
    assert ok.json() == {"challenge": "abc"}


def test_event_reply_pipeline(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    resp = client.post(
        "/webhook/feishu",
        json={
            "type": "event_callback",
            "event": {
                "message": {"message_id": "om_xxx", "text": "hello"},
                "sender": {"sender_type": "user"},
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["reply_ok"] is True
    assert payload["progress_reply_ok"] is False
    assert str(payload["run_id"]).startswith("run_")


def test_event_reply_fallbacks_to_chat_send_when_reply_fails(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()

    class _ReplyFailAPIClient(_DummyAPIClient):
        def __init__(self) -> None:
            super().__init__()
            self.last_receive_id = ""
            self.last_receive_id_type = ""

        def send_reply_text(self, *, message_id: str, text: str):  # noqa: ARG002
            self.reply_calls += 1
            self.reply_texts.append(text)

            class _Resp:
                ok = False
                error = "feishu_code_99991663"
                data = {"code": 99991663}

            return _Resp()

        def send_message_text(self, *, receive_id: str, text: str, receive_id_type: str = "chat_id"):  # noqa: ARG002
            self.message_calls += 1
            self.last_receive_id = receive_id
            self.last_receive_id_type = receive_id_type

            class _Resp:
                ok = True
                error = ""
                data = {"message_id": "om_fallback_ok"}

            return _Resp()

    api = _ReplyFailAPIClient()
    app.state.event_service.api_client = api

    client = TestClient(app)
    resp = client.post(
        "/webhook/feishu",
        json={
            "type": "event_callback",
            "event": {
                "message": {"message_id": "om_reply_fail", "chat_id": "oc_test_chat", "text": "hello"},
                "sender": {"sender_type": "user"},
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["reply_ok"] is True
    assert payload["reply_error"] == ""
    assert api.reply_calls >= 1
    assert api.message_calls >= 1
    assert api.last_receive_id == "oc_test_chat"
    assert api.last_receive_id_type == "chat_id"


def test_event_reply_pipeline_sends_progress_before_final_when_search_needed(tmp_path: Path) -> None:
    cfg = _build_config(
        tmp_path,
        extra={"search": {"web_search_enabled": True, "search_progress_enabled": True}},
    )
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    api = _DummyAPIClient()
    app.state.event_service.api_client = api

    client = TestClient(app)
    resp = client.post(
        "/webhook/feishu",
        json={
            "type": "event_callback",
            "event": {
                "message": {"message_id": "om_progress", "text": "最近跨境电商政策变化有哪些"},
                "sender": {"sender_type": "user"},
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["progress_reply_ok"] is True
    assert payload["reply_ok"] is True
    assert len(api.reply_texts) == 2
    assert api.reply_texts[0].startswith("正在搜索：")
    assert api.reply_texts[1] == "ok"


def test_progress_text_prefers_entities_and_intent_terms(tmp_path: Path) -> None:
    cfg = _build_config(
        tmp_path,
        extra={"search": {"web_search_enabled": True, "search_progress_enabled": True}},
    )
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    api = _DummyAPIClient()
    app.state.event_service.api_client = api

    client = TestClient(app)
    resp = client.post(
        "/webhook/feishu",
        json={
            "type": "event_callback",
            "event": {
                "message": {
                    "message_id": "om_progress_entity",
                    "text": "在最新的平台政策下，如何打造类似 哭哭马 拉布布 的爆款",
                },
                "sender": {"sender_type": "user"},
            },
        },
    )
    assert resp.status_code == 200
    assert len(api.reply_texts) == 2
    progress = api.reply_texts[0]
    assert progress.startswith("正在搜索：")
    assert "哭哭马" in progress
    assert "拉布布" in progress
    assert "在最" not in progress


def test_duplicate_event_callback_is_ignored(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    event_payload = {
        "type": "event_callback",
        "event_id": "evt_dedup_001",
        "event": {
            "message": {"message_id": "om_dup_001", "text": "hello"},
            "sender": {"sender_type": "user"},
        },
    }
    first = client.post("/webhook/feishu", json=event_payload)
    second = client.post("/webhook/feishu", json=event_payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["success"] is True
    assert first.json()["reply_ok"] is True
    assert second.json()["success"] is True
    assert second.json()["message"] == "ignored_duplicate_event"
    assert second.json()["duplicate"] is True
    assert app.state.event_service.api_client.reply_calls == 1


def test_event_reply_pipeline_persists_memory_records(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    resp = client.post(
        "/webhook/feishu",
        json={
            "type": "event_callback",
            "event_id": "evt_mem_001",
            "event": {
                "message": {"message_id": "om_mem_001", "text": "hello memory"},
                "sender": {"sender_type": "user"},
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    run_id = payload["run_id"]
    assert str(run_id).startswith("run_")

    with sqlite3.connect(cfg.database.db_path) as conn:
        run_row = conn.execute(
            "SELECT success, event_id, message_id, query_text FROM qa_run WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert run_row is not None
        assert int(run_row[0]) == 1
        assert run_row[1] == "evt_mem_001"
        assert run_row[2] == "om_mem_001"
        assert run_row[3] == "hello memory"

        io_types = {
            row[0]
            for row in conn.execute(
                "SELECT io_type FROM qa_io_snapshot WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        }
        assert {"input", "output", "final_citations", "full_trace"}.issubset(io_types)


def test_self_check_reuses_single_token_dialog_payload_call(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    api = _CountingTokenDialogAPIClient()
    app.state.event_service.api_client = api

    checks = app.state.event_service.run_self_check()
    assert checks["ok"] is True
    assert api.token_dialog_calls == 1


def test_log_debug_trace_keeps_index_fields_only(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()
    service = app.state.event_service

    caplog.set_level(logging.INFO, logger=event_service_module.__name__)
    service._log_debug_trace(
        event_data={
            "event_id": "evt_001",
            "event": {"message": {"message_id": "om_001"}},
        },
        result={
            "run_id": "run_001",
            "fallback_type": "no_rag_hit",
            "success": False,
            "reply_ok": False,
            "reply_error": "some_error",
            "debug_trace": {"k": "v"},
        },
    )

    log_line = next(
        (record.getMessage() for record in caplog.records if "gateway_debug_trace " in record.getMessage()),
        "",
    )
    assert log_line
    payload = json.loads(log_line.split("gateway_debug_trace ", 1)[1])
    assert payload == {
        "run_id": "run_001",
        "event_id": "evt_001",
        "message_id": "om_001",
        "fallback_type": "no_rag_hit",
        "success": False,
        "reply_ok": False,
    }


def test_search_progress_markers_and_text_are_utf8_readable(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()
    service = app.state.event_service

    assert "最近" in service._SEARCH_PROGRESS_MARKERS
    assert "最新" in service._SEARCH_PROGRESS_MARKERS
    progress_text = service._build_search_progress_text("最近跨境电商政策变化")
    assert progress_text.startswith("正在搜索：")
    assert "最近" in progress_text

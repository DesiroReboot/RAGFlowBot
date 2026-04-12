from __future__ import annotations

import json
import logging
from pathlib import Path
import types

from src.config import Config
from src.fastapi_gateway import long_connection


def _build_config(tmp_path: Path) -> Config:
    payload = {
        "database": {"db_path": str(tmp_path / "gateway_test.db")},
        "gateway": {
            "feishu": {
                "enabled": True,
                "receive_mode": "long_connection",
                "app_id": "app-id",
                "app_secret": "app-secret",
                "encrypt_key": "enc-key",
                "verification_token": "verify-token",
            }
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return Config(str(path))


def test_long_connection_dispatcher_builder_uses_encrypt_key_then_verification_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_config(tmp_path)
    captured: dict[str, tuple[str, str]] = {}

    class _DummyBootstrap:
        def __init__(self, config: Config) -> None:  # noqa: ARG002
            pass

        def start(self) -> dict[str, object]:
            return {}

    class _DummyService:
        def __init__(self, config: Config) -> None:  # noqa: ARG002
            pass

        def handle_event(self, event_data, **kwargs):  # noqa: ANN001, ARG002
            return {"success": True, "message": "ok"}

    class _DispatcherBuilder:
        def register_p2_im_message_receive_v1(self, handler):  # noqa: ANN001
            self._handler = handler
            return self

        def register_p2_im_message_read_v1(self, handler):  # noqa: ANN001
            self._read_handler = handler
            return self

        def build(self) -> object:
            return object()

    class _EventDispatcherHandler:
        @staticmethod
        def builder(encrypt_key: str, verification_token: str):
            captured["args"] = (encrypt_key, verification_token)
            return _DispatcherBuilder()

    class _ClientBuilder:
        def __init__(self, app_id: str, app_secret: str):  # noqa: ARG002
            pass

        def event_handler(self, dispatcher):  # noqa: ANN001
            self._dispatcher = dispatcher
            return self

        def log_level(self, level):  # noqa: ANN001
            self._level = level
            return self

        def build(self):
            class _Client:
                @staticmethod
                def start() -> None:
                    return

            return _Client()

    fake_lark = types.SimpleNamespace(
        EventDispatcherHandler=_EventDispatcherHandler,
        LogLevel=types.SimpleNamespace(INFO="INFO"),
        ws=types.SimpleNamespace(Client=types.SimpleNamespace(Builder=_ClientBuilder)),
    )

    monkeypatch.setattr(long_connection, "KBaseStartupBootstrap", _DummyBootstrap)
    monkeypatch.setattr(long_connection, "FeishuEventService", _DummyService)
    monkeypatch.setitem(__import__("sys").modules, "lark_oapi", fake_lark)
    monkeypatch.setenv("ECBOT_LONG_CONN_HEALTH_PATH", str(tmp_path / "longconn_health.json"))

    long_connection.run_long_connection_client(cfg)

    assert captured["args"] == ("enc-key", "verify-token")


def test_long_connection_dispatcher_builder_supports_sdk_message_message_read_method(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_config(tmp_path)
    captured: dict[str, object] = {"read_registered": False}

    class _DummyBootstrap:
        def __init__(self, config: Config) -> None:  # noqa: ARG002
            pass

        def start(self) -> dict[str, object]:
            return {}

    class _DummyService:
        def __init__(self, config: Config) -> None:  # noqa: ARG002
            pass

        def handle_event(self, event_data, **kwargs):  # noqa: ANN001, ARG002
            return {"success": True, "message": "ok"}

    class _DispatcherBuilder:
        def register_p2_im_message_receive_v1(self, handler):  # noqa: ANN001
            self._handler = handler
            return self

        def register_p2_im_message_message_read_v1(self, handler):  # noqa: ANN001
            self._read_handler = handler
            captured["read_registered"] = True
            return self

        def build(self) -> object:
            return object()

    class _EventDispatcherHandler:
        @staticmethod
        def builder(encrypt_key: str, verification_token: str):  # noqa: ARG004
            return _DispatcherBuilder()

    class _ClientBuilder:
        def __init__(self, app_id: str, app_secret: str):  # noqa: ARG002
            pass

        def event_handler(self, dispatcher):  # noqa: ANN001
            self._dispatcher = dispatcher
            return self

        def log_level(self, level):  # noqa: ANN001
            self._level = level
            return self

        def build(self):
            class _Client:
                @staticmethod
                def start() -> None:
                    return

            return _Client()

    fake_lark = types.SimpleNamespace(
        EventDispatcherHandler=_EventDispatcherHandler,
        LogLevel=types.SimpleNamespace(INFO="INFO"),
        ws=types.SimpleNamespace(Client=types.SimpleNamespace(Builder=_ClientBuilder)),
    )

    monkeypatch.setattr(long_connection, "KBaseStartupBootstrap", _DummyBootstrap)
    monkeypatch.setattr(long_connection, "FeishuEventService", _DummyService)
    monkeypatch.setitem(__import__("sys").modules, "lark_oapi", fake_lark)
    monkeypatch.setenv("ECBOT_LONG_CONN_HEALTH_PATH", str(tmp_path / "longconn_health.json"))

    long_connection.run_long_connection_client(cfg)

    assert captured["read_registered"] is True


def test_extract_event_type_from_object_header() -> None:
    payload = types.SimpleNamespace(header=types.SimpleNamespace(event_type="im.message.receive_v1"))
    assert long_connection._extract_event_type(payload, default="fallback") == "im.message.receive_v1"


def test_extract_event_type_from_dict_header() -> None:
    payload = {"header": {"event_type": "im.message.read_v1"}}
    assert long_connection._extract_event_type(payload, default="fallback") == "im.message.read_v1"


def test_extract_event_type_from_raw_payload_prefers_header_event_type() -> None:
    raw = json.dumps({"header": {"event_type": "im.message.receive_v1"}}).encode("utf-8")
    assert long_connection._extract_event_type_from_raw_payload(raw) == "im.message.receive_v1"


def test_record_event_observation_writes_health_and_structured_log(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    health_path = tmp_path / "longconn_health.json"
    monkeypatch.setenv("ECBOT_LONG_CONN_HEALTH_PATH", str(health_path))
    caplog.set_level(logging.INFO, logger=long_connection.__name__)

    long_connection._record_event_observation(
        stage="event_processed",
        event_type="im.message.receive_v1",
        event_payload={
            "event": {
                "message": {"message_id": "om_001", "chat_id": "oc_001"},
                "sender": {"sender_type": "user"},
            }
        },
        result={"run_id": "run_001", "reply_ok": True, "reply_error": ""},
    )

    payload = json.loads(health_path.read_text(encoding="utf-8"))
    assert payload["detail"] == "event_processed"
    assert payload["last_event_type"] == "im.message.receive_v1"
    assert payload["last_message_id"] == "om_001"
    assert payload["last_chat_id"] == "oc_001"
    assert payload["last_sender_type"] == "user"
    assert payload["last_run_id"] == "run_001"
    assert payload["last_reply_ok"] is True

    assert any("long_connection_event_observation" in record.getMessage() for record in caplog.records)

from __future__ import annotations

import json
from pathlib import Path

from src.config import Config
from src.fastapi_gateway import runtime


def _build_config(tmp_path: Path, *, receive_mode: str | None = None) -> Config:
    feishu_payload = {
        "enabled": True,
        "app_id": "app-id",
        "app_secret": "app-secret",
    }
    if receive_mode is not None:
        feishu_payload["receive_mode"] = receive_mode

    payload = {"gateway": {"feishu": feishu_payload}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return Config(str(path))


def test_resolve_receive_mode_default_long_connection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ECBOT_DOTENV_PATH", ".env.not-found")
    cfg = _build_config(tmp_path)
    assert runtime.resolve_receive_mode(cfg) == "long_connection"


def test_resolve_receive_mode_invalid_falls_back(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path, receive_mode="invalid-mode")
    assert runtime.resolve_receive_mode(cfg) == "long_connection"


def test_run_gateway_uses_webhook_server(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_config(tmp_path, receive_mode="webhook")
    called = {"webhook": 0}

    def _webhook(_: Config) -> None:
        called["webhook"] += 1

    monkeypatch.setattr(runtime, "run_webhook_server", _webhook)

    runtime.run_gateway(cfg)
    assert called == {"webhook": 1}


def test_run_gateway_uses_long_connection_client(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_config(tmp_path, receive_mode="long_connection")
    called = {"long_connection": 0}

    def _long_connection(_: Config) -> None:
        called["long_connection"] += 1

    monkeypatch.setattr(runtime, "run_long_connection_client", _long_connection)

    runtime.run_gateway(cfg)
    assert called == {"long_connection": 1}

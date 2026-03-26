from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from src.config import Config
from src.fastapi_gateway.services.event_service import FeishuEventService


def _health_file_path() -> Path:
    raw = str(os.getenv("ECBOT_LONG_CONN_HEALTH_PATH", "/tmp/ecbot_longconn_health.json")).strip()
    return Path(raw or "/tmp/ecbot_longconn_health.json")


def _write_connection_health(status: str, detail: str = "") -> None:
    payload = {
        "status": status,
        "detail": detail,
        "updated_at": time.time(),
    }
    path = _health_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


class _LarkConnectionHealthHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = str(record.getMessage())
        except Exception:
            return

        lowered = message.lower()
        if "connected to wss://" in lowered:
            _write_connection_health("connected", message)
            return
        if "connect failed" in lowered or "trying to reconnect" in lowered:
            _write_connection_health("reconnecting", message)


def _install_connection_health_handler() -> None:
    logger = logging.getLogger("Lark")
    for handler in logger.handlers:
        if isinstance(handler, _LarkConnectionHealthHandler):
            return
    logger.addHandler(_LarkConnectionHealthHandler())


def _resolve_lark_log_level(lark_module: Any, configured_level: str) -> Any:
    level_name = str(configured_level or "INFO").strip().upper()
    log_level_enum = getattr(lark_module, "LogLevel", None)
    if log_level_enum is None:
        return configured_level
    return getattr(log_level_enum, level_name, getattr(log_level_enum, "INFO", configured_level))


def _extract_text(message_obj: Any) -> str:
    direct_text = getattr(message_obj, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    content = getattr(message_obj, "content", "")
    if not isinstance(content, str):
        return ""
    try:
        payload = json.loads(content)
    except Exception:
        return content
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text
    return content


def _to_event_payload(data: Any) -> dict[str, Any]:
    event = getattr(data, "event", None)
    if event is None:
        return {"type": "event_callback", "event": {}}

    message = getattr(event, "message", None)
    sender = getattr(event, "sender", None)
    return {
        "type": "event_callback",
        "event": {
            "message": {
                "message_id": str(getattr(message, "message_id", "") or ""),
                "text": _extract_text(message),
                "content": str(getattr(message, "content", "") or ""),
            },
            "sender": {
                "sender_type": str(getattr(sender, "sender_type", "user") or "user"),
            },
        },
    }


def run_long_connection_client(config: Config | None = None) -> None:
    cfg = config or Config()
    service = FeishuEventService(cfg)
    _install_connection_health_handler()
    _write_connection_health("starting")

    try:
        import lark_oapi as lark  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError("lark_oapi is required for long_connection mode") from exc

    def _handle_message(data: Any) -> None:
        event_payload = _to_event_payload(data)
        service.handle_event(event_payload, skip_signature_verification=True)

    def _handle_message_read(_: Any) -> None:
        # Explicitly consume read-receipt events to avoid "processor not found" noise.
        return

    dispatcher_builder = lark.EventDispatcherHandler.builder(
        cfg.gateway.feishu.verification_token or "",
        cfg.gateway.feishu.encrypt_key or "",
    )
    dispatcher_builder = dispatcher_builder.register_p2_im_message_receive_v1(_handle_message)
    if hasattr(dispatcher_builder, "register_p2_im_message_read_v1"):
        dispatcher_builder = dispatcher_builder.register_p2_im_message_read_v1(_handle_message_read)
    dispatcher = dispatcher_builder.build()
    log_level = _resolve_lark_log_level(lark, cfg.gateway.feishu.long_conn_log_level)

    client_builder_cls = getattr(lark.ws.Client, "Builder", None)
    if client_builder_cls is not None:
        client_builder = client_builder_cls(
            cfg.gateway.feishu.app_id,
            cfg.gateway.feishu.app_secret,
        ).event_handler(dispatcher)
        if hasattr(client_builder, "log_level"):
            client_builder = client_builder.log_level(log_level)
        ws_client = client_builder.build()
    else:
        ws_client = lark.ws.Client(
            cfg.gateway.feishu.app_id,
            cfg.gateway.feishu.app_secret,
            log_level=log_level,
            event_handler=dispatcher,
        )
    ws_client.start()

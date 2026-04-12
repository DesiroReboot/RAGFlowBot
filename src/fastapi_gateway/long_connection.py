from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from src.config import Config
from src.fastapi_gateway.services.event_service import FeishuEventService
from src.RAG.startup_bootstrap import KBaseStartupBootstrap

logger = logging.getLogger(__name__)


def _health_file_path() -> Path:
    raw = str(os.getenv("ECBOT_LONG_CONN_HEALTH_PATH", "/tmp/ecbot_longconn_health.json")).strip()
    return Path(raw or "/tmp/ecbot_longconn_health.json")


def _write_connection_health(
    status: str,
    detail: str = "",
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "detail": detail,
        "updated_at": time.time(),
    }
    if isinstance(extra, dict):
        payload.update(extra)
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
                "chat_id": str(getattr(message, "chat_id", "") or ""),
                "text": _extract_text(message),
                "content": str(getattr(message, "content", "") or ""),
            },
            "sender": {
                "sender_type": str(getattr(sender, "sender_type", "user") or "user"),
            },
        },
    }


def _extract_event_type(data: Any, *, default: str) -> str:
    header = getattr(data, "header", None)
    if header is not None:
        header_event_type = str(getattr(header, "event_type", "")).strip()
        if header_event_type:
            return header_event_type

    if isinstance(data, dict):
        direct_type = str(data.get("type", "")).strip()
        if direct_type:
            return direct_type
        header_payload = data.get("header", {})
        if isinstance(header_payload, dict):
            header_event_type = str(header_payload.get("event_type", "")).strip()
            if header_event_type:
                return header_event_type

    return default


def _to_event_payload_from_customized(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        if isinstance(data.get("event"), dict):
            payload: dict[str, Any] = {
                "type": str(data.get("type", "event_callback") or "event_callback"),
                "event": data.get("event", {}),
            }
            event_id = str(data.get("event_id", "")).strip()
            if event_id:
                payload["event_id"] = event_id
            return payload
        if isinstance(data.get("message"), dict) or isinstance(data.get("sender"), dict):
            return {
                "type": "event_callback",
                "event": {
                    "message": data.get("message", {}) if isinstance(data.get("message"), dict) else {},
                    "sender": data.get("sender", {}) if isinstance(data.get("sender"), dict) else {},
                },
            }
    return {"type": "event_callback", "event": {}}


def _extract_event_type_from_raw_payload(payload: bytes) -> str:
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    header = data.get("header", {})
    if isinstance(header, dict):
        header_event_type = str(header.get("event_type", "")).strip()
        if header_event_type:
            return header_event_type
    event = data.get("event", {})
    if isinstance(event, dict):
        event_type = str(event.get("type", "")).strip()
        if event_type:
            return event_type
    return str(data.get("type", "")).strip()


class _TracingEventHandler:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def do_without_validation(self, payload: bytes) -> Any:
        raw_event_type = _extract_event_type_from_raw_payload(payload)
        if raw_event_type:
            logger.info("long_connection_raw_event_type %s", raw_event_type)
            _write_connection_health(
                "connected",
                "raw_event_received",
                extra={
                    "last_event_at": time.time(),
                    "last_event_type": raw_event_type,
                },
            )
        return self._inner.do_without_validation(payload)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def run_long_connection_client(config: Config | None = None) -> None:
    cfg = config or Config()
    # Keep long_connection behavior consistent with webhook startup hooks.
    KBaseStartupBootstrap(cfg).start()
    service = FeishuEventService(cfg)
    _install_connection_health_handler()
    _write_connection_health("starting")

    try:
        import lark_oapi as lark  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError("lark_oapi is required for long_connection mode") from exc

    def _handle_message(data: Any) -> None:
        event_payload = _to_event_payload(data)
        event_type = _extract_event_type(data, default="im.message.receive_v1")
        _record_event_observation(
            stage="event_received",
            event_payload=event_payload,
            event_type=event_type,
        )
        logger.info("long_connection_event_type %s", event_type)
        result = service.handle_event(event_payload, skip_signature_verification=True)
        _record_event_observation(
            stage="event_processed",
            event_payload=event_payload,
            event_type=event_type,
            result=result,
        )
        if not bool(result.get("success", False)) or not bool(result.get("reply_ok", True)):
            logger.warning(
                "long_connection_event_result %s",
                json.dumps(
                    {
                        "event_id": str(event_payload.get("event_id", "")).strip(),
                        "message_id": str(
                            (event_payload.get("event", {}) or {}).get("message", {}).get("message_id", "")
                        ).strip(),
                        "success": bool(result.get("success", False)),
                        "fallback_type": str(result.get("fallback_type", "")).strip(),
                        "reply_ok": bool(result.get("reply_ok", False)),
                        "reply_error": str(result.get("reply_error", "")).strip(),
                        "run_id": str(result.get("run_id", "")).strip(),
                    },
                    ensure_ascii=False,
                ),
            )

    def _handle_customized(data: Any) -> None:
        event_payload = _to_event_payload_from_customized(data)
        event_type = str(event_payload.get("type", "")).strip()
        _record_event_observation(
            stage="customized_event_received",
            event_payload=event_payload,
            event_type=event_type,
        )
        result = service.handle_event(event_payload, skip_signature_verification=True)
        _record_event_observation(
            stage="customized_event_processed",
            event_payload=event_payload,
            event_type=event_type,
            result=result,
        )

    def _handle_message_read(_: Any) -> None:
        # Explicitly consume read-receipt events to avoid "processor not found" noise.
        return

    dispatcher_builder = lark.EventDispatcherHandler.builder(
        cfg.gateway.feishu.encrypt_key or "",
        cfg.gateway.feishu.verification_token or "",
    )
    dispatcher_builder = dispatcher_builder.register_p2_im_message_receive_v1(_handle_message)
    if hasattr(dispatcher_builder, "register_p2_customized_event"):
        for event_type in ("im.message.receive_v1",):
            try:
                dispatcher_builder = dispatcher_builder.register_p2_customized_event(
                    event_type,
                    _handle_customized,
                )
            except Exception:
                # Ignore duplicate or unsupported registration; typed handlers may already exist.
                pass
    if hasattr(dispatcher_builder, "register_p1_customized_event"):
        for event_type in ("im.message.receive_v1", "message"):
            try:
                dispatcher_builder = dispatcher_builder.register_p1_customized_event(
                    event_type,
                    _handle_customized,
                )
            except Exception:
                # Ignore duplicate or unsupported registration.
                pass
    for register_name in (
        "register_p2_im_message_read_v1",
        "register_p2_im_message_message_read_v1",
    ):
        register = getattr(dispatcher_builder, register_name, None)
        if callable(register):
            dispatcher_builder = register(_handle_message_read)
            break
    dispatcher = _TracingEventHandler(dispatcher_builder.build())
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


def _record_event_observation(
    *,
    stage: str,
    event_payload: dict[str, Any],
    event_type: str,
    result: dict[str, Any] | None = None,
) -> None:
    event = event_payload.get("event", {}) if isinstance(event_payload, dict) else {}
    event = event if isinstance(event, dict) else {}
    message = event.get("message", {}) if isinstance(event, dict) else {}
    sender = event.get("sender", {}) if isinstance(event, dict) else {}
    message = message if isinstance(message, dict) else {}
    sender = sender if isinstance(sender, dict) else {}

    payload = {
        "event_type": str(event_type or "").strip(),
        "message_id": str(message.get("message_id", "")).strip(),
        "chat_id": str(message.get("chat_id", "")).strip(),
        "sender_type": str(sender.get("sender_type", "")).strip(),
        "run_id": str((result or {}).get("run_id", "")).strip(),
        "reply_ok": bool((result or {}).get("reply_ok", False)),
        "reply_error": str((result or {}).get("reply_error", "")).strip(),
    }
    _write_connection_health(
        "connected",
        stage,
        extra={
            "last_event_at": time.time(),
            "last_event_type": payload["event_type"],
            "last_message_id": payload["message_id"],
            "last_chat_id": payload["chat_id"],
            "last_sender_type": payload["sender_type"],
            "last_run_id": payload["run_id"],
            "last_reply_ok": payload["reply_ok"],
            "last_reply_error": payload["reply_error"],
        },
    )
    logger.info("long_connection_event_observation %s", json.dumps({"stage": stage, **payload}, ensure_ascii=False))

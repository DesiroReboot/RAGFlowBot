from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
from threading import Lock
import time
from typing import Any

from src.config import Config
from src.core.bot_agent import ReActAgent
from src.core.search.query_preprocessor import QueryPreprocessor
from src.core.trace_builder import build_debug_trace, extract_first_strategy_reason
from src.fastapi_gateway.security.verifier import FeishuAuthVerifier
from src.fastapi_gateway.services.feishu_client import FeishuAPIClient

logger = logging.getLogger(__name__)


@dataclass
class GatewayResult:
    success: bool
    message: str
    fallback_type: str | None = None
    embedding_dialog: dict[str, Any] | None = None
    debug_trace: dict[str, Any] | None = None


EventHandler = Callable[[dict[str, Any]], dict[str, Any]]


class FeishuEventService:
    _SEARCH_PROGRESS_MARKERS: tuple[str, ...] = (
        "latest",
        "recent",
        "today",
        "this year",
        "this month",
        "最近",
        "最新",
        "近期",
        "本周",
        "本月",
        "今年",
    )

    def __init__(self, config: Config):
        self.config = config
        self.agent = ReActAgent(config)
        self.api_client = FeishuAPIClient(config.gateway.feishu)
        self.query_preprocessor = QueryPreprocessor()
        self._dedup_ttl_seconds = 120
        self._dedup_max_entries = 2048
        self._processed_event_keys: dict[str, float] = {}
        self._dedup_lock = Lock()
        self._event_handlers: dict[str, EventHandler] = {
            "url_verification": self._handle_url_verification_event,
            "event_callback": self._handle_event_callback,
        }

    def validate_startup(self) -> dict[str, Any]:
        feishu_cfg = self.config.gateway.feishu
        token_dialog = self.api_client.token_dialog_payload()
        return {
            "enabled": feishu_cfg.enabled,
            "receive_mode": feishu_cfg.receive_mode,
            "openapi_base_url_ok": self.api_client.openapi_base_url_ok(),
            "token_dialog": token_dialog,
            "credential_validation": self.api_client.validate_credentials(),
            "webhook_path": feishu_cfg.webhook_path,
            "webhook_port": feishu_cfg.webhook_port,
        }

    def run_self_check(self) -> dict[str, Any]:
        feishu_cfg = self.config.gateway.feishu
        credential_validation = self.api_client.validate_credentials()
        retrieval_provider = str(getattr(self.config.search, "rag_provider", "legacy")).strip().lower() or "legacy"
        ragflow_cfg = getattr(self.config, "ragflow", None)
        ragflow_ready = bool(
            str(getattr(ragflow_cfg, "base_url", "")).strip()
            and str(getattr(ragflow_cfg, "api_key", "")).strip()
            and bool(getattr(ragflow_cfg, "dataset_map", {}))
        )
        checks = [
            {
                "stage": "gateway.enabled",
                "ok": bool(feishu_cfg.enabled),
                "detail": {"enabled": feishu_cfg.enabled},
            },
            {
                "stage": "gateway.receive_mode",
                "ok": feishu_cfg.receive_mode in {"webhook", "long_connection"},
                "detail": {"receive_mode": feishu_cfg.receive_mode},
            },
            {
                "stage": "gateway.openapi_base_url",
                "ok": self.api_client.openapi_base_url_ok(),
                "detail": {"openapi_base_url": feishu_cfg.openapi_base_url},
            },
            {
                "stage": "gateway.app_credentials",
                "ok": self.api_client.token_dialog_payload()["ok"],
                "detail": self.api_client.token_dialog_payload(),
            },
            {
                "stage": "gateway.encrypt_key",
                "ok": not self._is_placeholder(feishu_cfg.encrypt_key),
                "detail": {"set": bool(str(feishu_cfg.encrypt_key or "").strip())},
            },
            {
                "stage": "gateway.verification_token",
                "ok": not self._is_placeholder(feishu_cfg.verification_token),
                "detail": {"set": bool(str(feishu_cfg.verification_token or "").strip())},
            },
            {
                "stage": "gateway.tenant_access_token",
                "ok": bool(credential_validation["ok"]),
                "detail": credential_validation,
            },
            {
                "stage": "retrieval.provider",
                "ok": retrieval_provider in {"legacy", "ragflow"},
                "detail": {"provider": retrieval_provider},
            },
            {
                "stage": "retrieval.ragflow_config",
                "ok": True if retrieval_provider != "ragflow" else ragflow_ready,
                "detail": {
                    "enabled": retrieval_provider == "ragflow",
                    "base_url_set": bool(str(getattr(ragflow_cfg, "base_url", "")).strip()),
                    "api_key_set": bool(str(getattr(ragflow_cfg, "api_key", "")).strip()),
                    "dataset_count": len(dict(getattr(ragflow_cfg, "dataset_map", {}) or {})),
                    "timeout_ms": int(getattr(ragflow_cfg, "timeout_ms", 0) or 0),
                    "fallback_to_legacy": bool(getattr(ragflow_cfg, "fallback_to_legacy", True)),
                },
            },
        ]
        ok_count = sum(1 for item in checks if item["ok"])
        return {
            "ok": ok_count == len(checks),
            "summary": {"passed": ok_count, "total": len(checks)},
            "checks": checks,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def visualize_fullchain(self, query: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        stages: list[dict[str, Any]] = []
        normalized_query = str(query or "").strip()
        checks = self.run_self_check()
        stages.append(
            {
                "stage": "self_check",
                "status": "ok" if checks["ok"] else "degraded",
                "detail": checks["summary"],
            }
        )

        if not normalized_query:
            stages.append(
                {
                    "stage": "extract_query",
                    "status": "error",
                    "detail": {"reason": "query_empty"},
                }
            )
            return {
                "ok": False,
                "stages": stages,
                "query": "",
                "trace": {},
                "duration_ms": int((time.perf_counter() - started_at) * 1000),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        stages.append(
            {
                "stage": "extract_query",
                "status": "ok",
                "detail": {"query_length": len(normalized_query)},
            }
        )

        try:
            response = self.agent.run_sync(normalized_query, include_trace=True)
            stages.append(
                {
                    "stage": "agent_run",
                    "status": "ok",
                    "detail": {
                        "retrieval_confidence": response.retrieval_confidence,
                        "answer_preview": response.answer[:120],
                    },
                }
            )
            reply_route = "event_reply"
            if not checks["checks"][6]["ok"]:
                reply_route = "blocked_by_token"
            elif not self.config.gateway.feishu.target_chat_id:
                reply_route = "event_reply_or_message_id_required"
            stages.append(
                {
                    "stage": "reply_route",
                    "status": "ok" if reply_route != "blocked_by_token" else "degraded",
                    "detail": {"route": reply_route},
                }
            )
            ok = True
            trace = response.trace
        except Exception as exc:
            stages.append(
                {
                    "stage": "agent_run",
                    "status": "error",
                    "detail": {"error": str(exc)},
                }
            )
            ok = False
            trace = {}

        return {
            "ok": ok,
            "query": normalized_query,
            "stages": stages,
            "trace": trace,
            "duration_ms": int((time.perf_counter() - started_at) * 1000),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def handle_event(
        self,
        event_data: dict[str, Any],
        *,
        headers: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
        skip_signature_verification: bool = False,
    ) -> dict[str, Any]:
        headers = headers or {}
        raw_body = raw_body or json.dumps(event_data, ensure_ascii=False).encode("utf-8")
        feishu_cfg = self.config.gateway.feishu

        if not skip_signature_verification and not FeishuAuthVerifier.verify_signature(
            headers=headers,
            raw_body=raw_body,
            encrypt_key=feishu_cfg.encrypt_key,
        ):
            return self._build_error_payload("signature_invalid")

        handler = self._resolve_event_handler(event_data)
        return handler(event_data)

    def _resolve_event_handler(self, event_data: dict[str, Any]) -> EventHandler:
        event_type = str(event_data.get("type", "")).strip()
        return self._event_handlers.get(event_type, self._handle_event_callback)

    def _handle_url_verification_event(self, event_data: dict[str, Any]) -> dict[str, Any]:
        if not FeishuAuthVerifier.verify_verification_token(
            event_data,
            self.config.gateway.feishu.verification_token,
        ):
            return self._build_error_payload("verification_token_invalid")
        return {"challenge": event_data.get("challenge", "")}

    @staticmethod
    def _build_error_payload(fallback_type: str) -> dict[str, Any]:
        return {
            "success": False,
            "message": FeishuEventService._fallback_message(fallback_type),
            "fallback_type": fallback_type,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _handle_event_callback(self, event_data: dict[str, Any]) -> dict[str, Any]:
        dedup_key = self._event_dedup_key(event_data)
        if dedup_key and self._is_duplicate_event(dedup_key):
            return {
                "success": True,
                "message": "ignored_duplicate_event",
                "fallback_type": None,
                "reply_ok": True,
                "reply_error": "",
                "token_dialog": None,  # nosec B105
                "duplicate": True,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        if self._is_bot_message(event_data):
            return {
                "success": True,
                "message": "ignored_self_message",
                "fallback_type": None,
                "embedding_dialog": None,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        query = self._extract_query(event_data)
        result, progress_reply_result = self._run_query_flow(event_data=event_data, query=query)

        reply_result = self._reply_to_event(event_data=event_data, text=result.message)
        payload = self._build_callback_payload(
            result=result,
            reply_result=reply_result,
            progress_reply_result=progress_reply_result,
        )
        self._log_debug_trace(event_data=event_data, result=payload)
        return payload

    def _run_query_flow(
        self,
        *,
        event_data: dict[str, Any],
        query: str,
    ) -> tuple[GatewayResult, dict[str, Any]]:
        if not query.strip():
            return (
                GatewayResult(
                    success=False,
                    message=self._fallback_message("invalid_input"),
                    fallback_type="error",
                ),
                {"ok": False, "error": "query_empty", "data": {}},
            )
        progress_reply_result = self._try_send_progress_reply(event_data=event_data, query=query)
        return self._process_query(query), progress_reply_result

    def _build_callback_payload(
        self,
        *,
        result: GatewayResult,
        reply_result: dict[str, Any],
        progress_reply_result: dict[str, Any],
    ) -> dict[str, Any]:
        token_dialog = self.api_client.token_dialog_payload()
        return {
            "success": result.success,
            "message": result.message,
            "fallback_type": result.fallback_type,
            "reply_ok": reply_result.get("ok", False),
            "reply_error": reply_result.get("error", ""),
            "progress_reply_ok": progress_reply_result.get("ok", False),
            "progress_reply_error": progress_reply_result.get("error", ""),
            "token_dialog": token_dialog if not token_dialog["ok"] else None,
            "embedding_dialog": result.embedding_dialog,
            "debug_trace": result.debug_trace,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _extract_query(self, event_data: dict[str, Any]) -> str:
        event = self._extract_event(event_data)
        message = event.get("message", {})
        text = message.get("text")
        if isinstance(text, str):
            return text
        content = message.get("content", "")
        if isinstance(content, str):
            maybe_text = self._parse_message_content(content)
            if maybe_text:
                return maybe_text
            return content
        return ""

    def _process_query(self, query: str) -> GatewayResult:
        try:
            response = self.agent.run_sync(query, include_trace=True)
        except TimeoutError:
            return GatewayResult(
                success=False,
                message=self._fallback_message("timeout"),
                fallback_type="timeout",
                debug_trace={"error": "timeout"},
            )
        except Exception:
            return GatewayResult(
                success=False,
                message=self._fallback_message("error"),
                fallback_type="error",
                debug_trace={"error": "internal_error"},
            )
        embedding_dialog = self._embedding_dialog_from_trace(response.trace)
        debug_trace = self._extract_debug_trace(trace=response.trace, query=query)

        output_guardrail = self.config.guardrails.output
        if (
            output_guardrail.enabled
            and response.retrieval_confidence < output_guardrail.min_retrieval_confidence
        ):
            return GatewayResult(
                success=False,
                message=self._fallback_message("no_rag_hit"),
                fallback_type="no_rag_hit",
                embedding_dialog=embedding_dialog,
                debug_trace=debug_trace,
            )

        return GatewayResult(
            success=True,
            message=response.answer,
            embedding_dialog=embedding_dialog,
            debug_trace=debug_trace,
        )

    def _try_send_progress_reply(self, *, event_data: dict[str, Any], query: str) -> dict[str, Any]:
        if not self._should_send_search_progress(query):
            return {"ok": False, "error": "progress_not_required", "data": {}}
        text = self._build_search_progress_text(query)
        try:
            return self._reply_to_event(event_data=event_data, text=text)
        except Exception as exc:
            return {"ok": False, "error": f"progress_send_error:{exc}", "data": {}}

    def _should_send_search_progress(self, query: str) -> bool:
        if not bool(self.config.search.search_progress_enabled):
            return False
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        if any(marker in lowered for marker in self._SEARCH_PROGRESS_MARKERS):
            return True
        processed = self.query_preprocessor.process(query)
        intent = processed.get("query_intent", {})
        if isinstance(intent, dict) and bool(intent.get("need_web_search", False)):
            return True
        temporal_terms = intent.get("temporal_terms", []) if isinstance(intent, dict) else []
        if isinstance(temporal_terms, list) and any(str(item).strip() for item in temporal_terms):
            return True
        return False

    def _build_search_progress_text(self, query: str) -> str:
        top_k = max(1, int(self.config.search.search_progress_keyword_top_k))
        keywords = self.query_preprocessor.extract_progress_keywords(query, top_k=top_k)
        keyword_text = ", ".join(keywords[:top_k]) if keywords else str(query).strip()
        return f"正在搜索：{keyword_text}"

    def _reply_to_event(self, *, event_data: dict[str, Any], text: str) -> dict[str, Any]:
        event = self._extract_event(event_data)
        message = event.get("message", {})
        message_id = str(message.get("message_id", "")).strip()
        if message_id:
            reply = self.api_client.send_reply_text(message_id=message_id, text=text)
            return {"ok": reply.ok, "error": reply.error, "data": reply.data}

        target_chat_id = self.config.gateway.feishu.target_chat_id
        if target_chat_id:
            reply = self.api_client.send_message_text(receive_id=target_chat_id, text=text)
            return {"ok": reply.ok, "error": reply.error, "data": reply.data}

        return {"ok": False, "error": "missing_message_id_and_target_chat_id", "data": {}}

    @staticmethod
    def _extract_event(event_data: dict[str, Any]) -> dict[str, Any]:
        event = event_data.get("event", {})
        if isinstance(event, dict):
            return event
        return {}

    def _event_dedup_key(self, event_data: dict[str, Any]) -> str:
        event_id = str(event_data.get("event_id", "")).strip()
        if event_id:
            return f"event_id:{event_id}"

        event = self._extract_event(event_data)
        message = event.get("message", {})
        if isinstance(message, dict):
            message_id = str(message.get("message_id", "")).strip()
            if message_id:
                return f"message_id:{message_id}"
        return ""

    def _is_duplicate_event(self, dedup_key: str) -> bool:
        now = time.time()
        with self._dedup_lock:
            expire_before = now - self._dedup_ttl_seconds
            expired_keys = [
                key for key, ts in self._processed_event_keys.items() if ts < expire_before
            ]
            for key in expired_keys:
                self._processed_event_keys.pop(key, None)

            if dedup_key in self._processed_event_keys:
                return True

            if len(self._processed_event_keys) >= self._dedup_max_entries:
                oldest_key = min(
                    self._processed_event_keys,
                    key=lambda key: self._processed_event_keys[key],
                )
                self._processed_event_keys.pop(oldest_key, None)

            self._processed_event_keys[dedup_key] = now
            return False

    def _is_bot_message(self, event_data: dict[str, Any]) -> bool:
        event = self._extract_event(event_data)
        sender = event.get("sender", {})
        sender_type = str(sender.get("sender_type", "")).lower()
        return sender_type == "bot"

    @staticmethod
    def _parse_message_content(content: str) -> str:
        try:
            data = json.loads(content)
        except Exception:
            return ""
        if isinstance(data, dict):
            text = data.get("text")
            if isinstance(text, str):
                return text
        return ""

    @staticmethod
    def _fallback_message(reason: str) -> str:
        fallback_map = {
            "timeout": "request timeout",
            "rate_limit": "rate limited",
            "no_rag_hit": "no relevant knowledge found",
            "invalid_input": "invalid message input",
            "signature_invalid": "signature verification failed",
            "verification_token_invalid": "verification credential check failed",  # nosec B105
            "error": "temporary internal error",
        }
        return fallback_map.get(reason, fallback_map["error"])

    @staticmethod
    def _embedding_dialog_from_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
        search_trace = trace.get("search", {}) if isinstance(trace, dict) else {}
        generation_trace = (
            search_trace.get("generation", {}) if isinstance(search_trace, dict) else {}
        )
        branch_errors = (
            generation_trace.get("branch_errors", {})
            if isinstance(generation_trace, dict)
            else {}
        )
        vec_error = str(branch_errors.get("vec", "")).strip() if isinstance(branch_errors, dict) else ""
        if not vec_error:
            return None

        lowered = vec_error.lower()
        is_embedding_failure = (
            "remote embedding failed" in lowered
            or "embedding model mismatch" in lowered
            or "embedding dimension mismatch" in lowered
        )
        if not is_embedding_failure:
            return None

        return {
            "ok": False,
            "reason": "embedding_unavailable",
            "message": "embedding 调用失败，请检查 ECBOT_EMBEDDING_API_KEY / ECBOT_EMBEDDING_MODEL。",
            "detail": vec_error,
        }

    @staticmethod
    def _extract_debug_trace(*, trace: dict[str, Any], query: str) -> dict[str, Any]:
        search_trace = trace.get("search", {}) if isinstance(trace, dict) else {}
        planner = search_trace.get("planner", {}) if isinstance(search_trace, dict) else {}
        rag = search_trace.get("rag", {}) if isinstance(search_trace, dict) else {}
        fallback_reason = extract_first_strategy_reason(trace)

        query_hash = hashlib.md5(str(query or "").encode("utf-8")).hexdigest()[:10]
        query_preview = str(query or "").strip().replace("\n", " ")[:40]
        final_results = search_trace.get("final_results", []) if isinstance(search_trace, dict) else []
        result_count = len(final_results) if isinstance(final_results, list) else 0
        allow_rag = True
        filter_reason = ""
        if isinstance(planner, dict):
            allow_rag = bool(planner.get("allow_rag", True))
            filter_reason = str(planner.get("filter_reason", "")).strip()
        return build_debug_trace(
            query_hash=query_hash,
            query_preview=query_preview,
            allow_rag=allow_rag,
            filter_reason=filter_reason,
            rag_executed=bool(rag.get("executed", False)) if isinstance(rag, dict) else False,
            rag_skip_reason=str(rag.get("skip_reason", "")).strip() if isinstance(rag, dict) else "",
            result_count=result_count,
            fallback_reason=fallback_reason,
        )

    def _log_debug_trace(self, *, event_data: dict[str, Any], result: dict[str, Any]) -> None:
        event = self._extract_event(event_data)
        message = event.get("message", {}) if isinstance(event, dict) else {}
        log_payload = {
            "event_id": str(event_data.get("event_id", "")).strip(),
            "message_id": str(message.get("message_id", "")).strip() if isinstance(message, dict) else "",
            "fallback_type": result.get("fallback_type"),
            "success": bool(result.get("success", False)),
            "reply_ok": bool(result.get("reply_ok", False)),
            "reply_error": str(result.get("reply_error", "")).strip(),
            "debug_trace": result.get("debug_trace"),
        }
        logger.info("gateway_debug_trace %s", json.dumps(log_payload, ensure_ascii=False))

    @staticmethod
    def _is_placeholder(value: str) -> bool:
        normalized = str(value or "").strip().upper()
        if not normalized:
            return True
        return normalized.startswith("YOUR_FEISHU_") or normalized.startswith("YOUR_")

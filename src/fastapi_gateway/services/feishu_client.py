from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any
from urllib import error, parse, request

from src.config import GatewayFeishuConfig


@dataclass
class FeishuAPIResponse:
    ok: bool
    data: dict[str, Any]
    error: str = ""


class FeishuAPIClient:
    def __init__(self, config: GatewayFeishuConfig):
        self.config = config
        self._tenant_access_token: str | None = None
        self._tenant_access_token_expire_at = 0.0

    def required_tokens(self) -> list[str]:
        missing: list[str] = []
        if self._is_unset(self.config.app_id):
            missing.append("gateway.feishu.app_id")
        if self._is_unset(self.config.app_secret):
            missing.append("gateway.feishu.app_secret")
        return missing

    def token_dialog_payload(self) -> dict[str, Any]:
        missing = self.required_tokens()
        message = "ok" if not missing else "missing Feishu app credentials"
        return {
            "ok": not missing,
            "missing_tokens": missing,
            "message": message,
        }

    def openapi_base_url_ok(self) -> bool:
        return self.config.openapi_base_url.rstrip("/") == "https://open.feishu.cn/open-apis"

    def validate_credentials(self) -> dict[str, Any]:
        token_result = self._get_tenant_access_token(force_refresh=True)
        data = token_result.data or {}
        expire = int(data.get("expire", 0) or 0)
        return {
            "ok": token_result.ok,
            "error": token_result.error,
            "feishu_code": int(data.get("code", 0) or 0),
            "feishu_msg": str(data.get("msg", "") or ""),
            "expire": expire,
            "tenant_access_token_ready": bool(self._tenant_access_token),
        }

    def send_reply_text(self, *, message_id: str, text: str) -> FeishuAPIResponse:
        token_result = self._get_tenant_access_token(force_refresh=False)
        if not token_result.ok:
            return token_result

        payload = {
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        return self._request(
            method="POST",
            path=f"/im/v1/messages/{message_id}/reply",
            json_payload=payload,
            access_token=token_result.data["tenant_access_token"],
        )

    def send_message_text(
        self,
        *,
        receive_id: str,
        text: str,
        receive_id_type: str = "chat_id",
    ) -> FeishuAPIResponse:
        token_result = self._get_tenant_access_token(force_refresh=False)
        if not token_result.ok:
            return token_result

        payload = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        return self._request(
            method="POST",
            path=f"/im/v1/messages?{parse.urlencode({'receive_id_type': receive_id_type})}",
            json_payload=payload,
            access_token=token_result.data["tenant_access_token"],
        )

    def _get_tenant_access_token(self, *, force_refresh: bool) -> FeishuAPIResponse:
        now = time.time()
        if (
            not force_refresh
            and self._tenant_access_token
            and now < self._tenant_access_token_expire_at
        ):
            return FeishuAPIResponse(
                ok=True,
                data={"tenant_access_token": self._tenant_access_token},
            )

        missing = self.required_tokens()
        if missing:
            return FeishuAPIResponse(
                ok=False,
                data={"missing_tokens": missing},
                error="missing_credentials",
            )

        resp = self._request(
            method="POST",
            path="/auth/v3/tenant_access_token/internal",
            json_payload={
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret,
            },
        )
        if not resp.ok:
            return resp

        tenant_access_token = str(resp.data.get("tenant_access_token", ""))
        expire = int(resp.data.get("expire", 0))
        if not tenant_access_token:
            return FeishuAPIResponse(
                ok=False,
                data=resp.data,
                error="tenant_access_token_missing",
            )

        self._tenant_access_token = tenant_access_token
        self._tenant_access_token_expire_at = time.time() + max(expire - 300, 60)
        return FeishuAPIResponse(
            ok=True,
            data={"tenant_access_token": self._tenant_access_token, "expire": expire},
        )

    def _request(
        self,
        *,
        method: str,
        path: str,
        json_payload: dict[str, Any],
        access_token: str | None = None,
    ) -> FeishuAPIResponse:
        url = self._safe_http_url(f"{self.config.openapi_base_url.rstrip('/')}{path}")
        body = json.dumps(json_payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        req = request.Request(url=url, data=body, headers=headers, method=method)

        try:
            with request.urlopen(req, timeout=max(1, self.config.request_timeout)) as resp:  # nosec B310
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            return FeishuAPIResponse(
                ok=False,
                data=self._safe_json(raw),
                error=f"http_{exc.code}",
            )
        except Exception as exc:
            return FeishuAPIResponse(ok=False, data={}, error=str(exc))

        code = int(data.get("code", -1))
        if code != 0:
            return FeishuAPIResponse(ok=False, data=data, error=f"feishu_code_{code}")
        return FeishuAPIResponse(ok=True, data=data)

    @staticmethod
    def _safe_json(raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            return {"raw": raw}
        return {"raw": raw}

    @staticmethod
    def _safe_http_url(raw_url: str) -> str:
        parsed = parse.urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("missing url host")
        return raw_url

    @staticmethod
    def _is_unset(value: str) -> bool:
        normalized = str(value or "").strip()
        if not normalized:
            return True
        return normalized.upper().startswith("YOUR_FEISHU_")

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from urllib import error, request


def _mode() -> str:
    return str(os.getenv("ECBOT_FEISHU_RECEIVE_MODE", "long_connection")).strip().lower()


def _check_webhook() -> int:
    port = str(os.getenv("ECBOT_FEISHU_WEBHOOK_PORT", "8000")).strip() or "8000"
    url = f"http://127.0.0.1:{port}/health"
    try:
        with request.urlopen(url, timeout=3) as resp:
            if 200 <= int(getattr(resp, "status", 0)) < 300:
                return 0
    except Exception:
        return 1
    return 1


def _check_long_connection() -> int:
    path = Path(str(os.getenv("ECBOT_LONG_CONN_HEALTH_PATH", "/tmp/ecbot_longconn_health.json")).strip())
    ttl_sec = int(str(os.getenv("ECBOT_LONG_CONN_HEALTH_TTL_SEC", "600")).strip() or "600")
    if not path.exists():
        return 1

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 1

    status = str(payload.get("status", "")).strip().lower()
    updated_at = float(payload.get("updated_at", 0.0))
    if status == "connected":
        return 0
    if status == "reconnecting" and updated_at > 0 and (time.time() - updated_at) <= ttl_sec:
        return 0
    return 1


def main() -> int:
    mode = _mode()
    if mode == "webhook":
        return _check_webhook()
    if mode == "long_connection":
        return _check_long_connection()
    return 1


if __name__ == "__main__":
    sys.exit(main())

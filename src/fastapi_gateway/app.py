from __future__ import annotations

from fastapi import FastAPI

from src.config import Config
from src.fastapi_gateway.routes.diagnostics import create_diagnostics_router
from src.fastapi_gateway.routes.health import create_health_router
from src.fastapi_gateway.routes.startup_check import create_startup_check_router
from src.fastapi_gateway.routes.webhook import create_webhook_router
from src.fastapi_gateway.services.event_service import FeishuEventService
from src.RAG.startup_bootstrap import KBaseStartupBootstrap


def create_app(
    config: Config | None = None,
    *,
    event_service: FeishuEventService | None = None,
) -> FastAPI:
    cfg = config or Config()
    if str(cfg.gateway.feishu.receive_mode or "").strip().lower() == "long_connection":
        raise RuntimeError(
            "receive_mode=long_connection must be started via `py -m src.main`; "
            "direct uvicorn app factory startup only supports webhook mode."
        )
    service = event_service or FeishuEventService(cfg)

    app = FastAPI(title="ECBot Gateway", version="1.0.0")
    app.state.gateway_config = cfg
    app.state.event_service = service

    webhook_path = cfg.gateway.feishu.webhook_path.rstrip("/") or "/webhook/feishu"
    app.include_router(create_health_router())
    app.include_router(create_startup_check_router(service))
    app.include_router(create_diagnostics_router(service))
    app.include_router(create_webhook_router(webhook_path, service))

    @app.on_event("startup")
    def _kb_startup_init() -> None:
        # Opt-in KB init; honors blocking/fail-open settings in config.
        KBaseStartupBootstrap(cfg).start()
    return app

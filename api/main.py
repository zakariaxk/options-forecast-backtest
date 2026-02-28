"""FastAPI application — single entrypoint for backend + UI."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.core.errors import ApiError
from api.core.logging import configure_logging
from api.core.settings import get_settings
from api.routers import backtest
from api.schemas.io import HealthResponse


def create_app() -> FastAPI:
    configure_logging()
    get_settings()

    app = FastAPI(title="Options Forecast & Backtest", version="2.0.0")

    # --- Exception handlers ------------------------------------------------

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={"error_code": "VALIDATION_ERROR", "message": str(exc), "details": {}},
        )

    # --- Middleware ---------------------------------------------------------

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routes ------------------------------------------------------------

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", time=datetime.now(timezone.utc))

    app.include_router(backtest.router, prefix="/api/v1")

    # --- Static UI ---------------------------------------------------------

    web_dir = Path(__file__).resolve().parents[1] / "web"
    static_dir = web_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir.as_posix()), name="static")
    index_path = web_dir / "index.html"

    @app.get("/")
    def ui_index():
        if index_path.exists():
            return FileResponse(index_path.as_posix())
        return {"message": "UI not built. See README.md."}

    return app


app = create_app()

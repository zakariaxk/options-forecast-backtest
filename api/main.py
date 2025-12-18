from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.core.logging import configure_logging
from api.core.settings import get_settings
from api.routers import backtest, models, pipelines, predict
from api.schemas.io import HealthResponse


def create_app() -> FastAPI:
    configure_logging()
    get_settings()
    app = FastAPI(title="Options Forecast & Backtest API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", time=datetime.now(timezone.utc))

    app.include_router(predict.router, prefix="/api/v1")
    app.include_router(backtest.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(pipelines.router, prefix="/api/v1")

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

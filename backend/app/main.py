"""TerraVision AI backend - FastAPI application factory.

Run locally:  uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app import __version__
from backend.app.config import Settings, get_settings
from backend.app.db import Database
from backend.app.errors import install_error_handlers
from backend.app.logging_conf import configure_logging
from backend.app.routers import analyses, evaluation, satellite, system
from backend.app.satellite.registry import ProviderRegistry
from backend.app.services.analysis import AnalysisService
from backend.app.services.model_manager import ModelManager
from backend.app.services.samples import SampleService
from backend.app.state import AppState

log = logging.getLogger("terravision.app")


def create_app(settings: Settings | None = None, *, provider_transport: httpx.BaseTransport | None = None,
               log_to_file: bool = True) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        configure_logging(settings.log_level, settings.storage_path / "logs" if log_to_file else None)
        db = Database(settings.db_url)
        db.create_all()
        models = ModelManager(settings)
        services = AppState(
            settings=settings, db=db, models=models,
            analyses=AnalysisService(settings, db, models),
            samples=SampleService(settings),
            providers=ProviderRegistry(settings, provider_transport),
        )
        app.state.services = services
        interrupted = services.analyses.recover_interrupted()
        orphans = services.analyses.cleanup_orphans()
        log.info("startup complete", extra={
            "version": __version__, "device": models.device.type, "interrupted_jobs": interrupted,
            "orphan_folders_removed": orphans,
            "models_available": [m["key"] for m in models.list_models() if m["available"]],
        })
        try:
            yield
        finally:
            services.analyses.shutdown()
            models.unload_all()
            db.dispose()
            log.info("shutdown complete")

    app = FastAPI(
        title="TerraVision AI - Satellite Change Intelligence",
        version=__version__,
        description="Bitemporal satellite change detection with a Siamese Swin Transformer and a CNN baseline.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"],
        expose_headers=["Content-Disposition", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        log.info("request", extra={"request_id": request_id, "method": request.method,
                                   "path": request.url.path, "status": response.status_code,
                                   "ms": round((time.perf_counter() - start) * 1000, 1)})
        return response

    install_error_handlers(app)
    app.include_router(system.router)
    app.include_router(analyses.router)
    app.include_router(evaluation.router)
    app.include_router(satellite.router)
    return app


app = create_app()

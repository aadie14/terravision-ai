"""Application errors and the single JSON error envelope used by every endpoint.

    {"error": {"code": "IMAGE_DIMENSION_MISMATCH", "message": "...", "details": {...}}}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("terravision.errors")


class AppError(Exception):
    """An expected, user-facing failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, status: int = 400, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


class NotFoundError(AppError):
    def __init__(self, what: str, identifier: str) -> None:
        super().__init__("NOT_FOUND", f"{what} '{identifier}' was not found", status=404)


def _envelope(code: str, message: str, status: int, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                               "details": details or {}}})


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return _envelope(exc.code, exc.message, exc.status, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        issues = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return _envelope("VALIDATION_ERROR", "The request is invalid", 422, {"issues": issues})

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 413: "PAYLOAD_TOO_LARGE"}.get(exc.status_code,
                                                                                         "HTTP_ERROR")
        return _envelope(code, str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return _envelope("INTERNAL_ERROR", "An unexpected server error occurred", 500)

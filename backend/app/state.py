"""Application service container, created once at startup and shared by all requests."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from backend.app.config import Settings
from backend.app.db import Database
from backend.app.satellite.registry import ProviderRegistry
from backend.app.services.analysis import AnalysisService
from backend.app.services.model_manager import ModelManager
from backend.app.services.samples import SampleService


@dataclass
class AppState:
    settings: Settings
    db: Database
    models: ModelManager
    analyses: AnalysisService
    samples: SampleService
    providers: ProviderRegistry


def get_state(request: Request) -> AppState:
    return request.app.state.services

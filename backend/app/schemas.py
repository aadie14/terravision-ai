"""Typed request / response contracts for the HTTP API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Status = Literal["queued", "running", "completed", "failed"]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    time: datetime
    database: bool
    models_available: int


class AnalysisCreate(BaseModel):
    """Start an analysis from a bundled sample, or re-run stored inputs with another model."""

    model_config = ConfigDict(protected_namespaces=())

    sample_id: str | None = Field(None, max_length=64)
    rerun_of: str | None = Field(None, min_length=32, max_length=32)
    model: str | None = Field(None, max_length=64)
    threshold: float | None = Field(None, gt=0, lt=1)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> AnalysisCreate:
        if (self.sample_id is None) == (self.rerun_of is None):
            raise ValueError("Provide exactly one of 'sample_id' or 'rerun_of'")
        return self


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., gt=0, lt=1)


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    name: str
    status: Status
    source: str
    model_key: str
    threshold: float
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    width: int | None
    height: int | None
    georeferenced: bool
    has_ground_truth: bool
    changed_fraction: float | None
    inference_seconds: float | None
    error_code: str | None
    error_message: str | None


class AnalysisDetail(AnalysisSummary):
    inputs: dict[str, Any]
    artifacts: list[str]


class AnalysisList(BaseModel):
    items: list[AnalysisSummary]
    total: int
    limit: int
    offset: int


class AnalysisResults(BaseModel):
    analysis: AnalysisDetail
    results: dict[str, Any]
    downloads: dict[str, str]


class StatsResponse(BaseModel):
    total: int
    completed: int
    failed: int
    in_progress: int
    mean_changed_percent: float | None
    last_changed_percent: float | None
    mean_inference_seconds: float | None
    by_model: dict[str, int]


class EvaluationRequest(BaseModel):
    """Evaluate models on bundled samples that have ground truth (a quick in-app check)."""

    models: list[str] | None = Field(None, max_length=8)
    sample_ids: list[str] | None = Field(None, max_length=64)
    threshold: float | None = Field(None, gt=0, lt=1)  # None: each model's calibrated threshold


class BBoxModel(BaseModel):
    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-85, le=85)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-85, le=85)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


class SatelliteSearchRequest(BaseModel):
    provider: str = "earth_search"
    bbox: BBoxModel
    start: date
    end: date
    max_cloud_cover: float = Field(30, ge=0, le=100)
    limit: int = Field(10, ge=1, le=30)

    @model_validator(mode="after")
    def _dates(self) -> SatelliteSearchRequest:
        if self.end < self.start:
            raise ValueError("'end' must not be before 'start'")
        if self.start < date(2015, 6, 23):
            raise ValueError("Sentinel-2 imagery is only available from 2015-06-23 onwards")
        return self


class SatelliteRetrieveRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider: str = "earth_search"
    bbox: BBoxModel
    t1_scene_id: str = Field(..., min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.\-]+$")
    t2_scene_id: str = Field(..., min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.\-]+$")
    model: str | None = Field(None, max_length=64)
    threshold: float | None = Field(None, gt=0, lt=1)
    name: str | None = Field(None, max_length=120)

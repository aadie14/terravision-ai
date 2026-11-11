"""Environment-driven configuration. Every value can be overridden with a ``TV_*`` variable or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TV_", env_file=".env", extra="ignore")

    # --- paths (relative paths resolve against the repository root)
    storage_dir: Path = Path("storage")
    checkpoint_dir: Path = Path("checkpoints")
    reports_dir: Path = Path("reports/evaluation")
    samples_dir: Path = Path("samples")
    database_url: str = ""  # default: sqlite file inside storage_dir

    # --- inference
    device: str = Field("auto", pattern="^(auto|cpu|cuda)$")
    amp: bool = True
    inference_batch_size: int = Field(8, ge=1, le=64)
    tile_overlap: int = Field(64, ge=0, le=192)
    default_model: str = "swin_siamese"
    default_threshold: float = Field(0.5, gt=0, lt=1)

    # --- upload limits
    max_upload_mb: int = Field(50, ge=1, le=1024)
    min_image_side: int = Field(32, ge=8)
    max_image_side: int = Field(4096, ge=256)

    # --- web
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    log_level: str = "INFO"

    # --- satellite providers
    satellite_max_aoi_km: float = Field(25.0, gt=0)
    satellite_min_aoi_km: float = Field(1.0, gt=0)
    satellite_timeout_s: float = Field(60.0, gt=0)
    earth_search_url: str = "https://earth-search.aws.element84.com/v1"
    cdse_stac_url: str = "https://stac.dataspace.copernicus.eu/v1"
    cdse_token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"  # noqa: S105 - a URL, not a secret
    cdse_process_url: str = "https://sh.dataspace.copernicus.eu/api/v1/process"
    cdse_client_id: str = ""
    cdse_client_secret: str = ""
    geocoder_url: str = "https://nominatim.openstreetmap.org/search"

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def storage_path(self) -> Path:
        return self.resolve(self.storage_dir)

    @property
    def analyses_path(self) -> Path:
        return self.storage_path / "analyses"

    @property
    def checkpoint_path(self) -> Path:
        return self.resolve(self.checkpoint_dir)

    @property
    def reports_path(self) -> Path:
        return self.resolve(self.reports_dir)

    @property
    def samples_path(self) -> Path:
        return self.resolve(self.samples_dir)

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.storage_path / 'terravision.sqlite3').as_posix()}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()

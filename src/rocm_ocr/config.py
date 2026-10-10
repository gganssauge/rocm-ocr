"""Configuration settings for the ROCm OCR microservice."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hsa_override_gfx_version: str = "11.0.0"
    ocr_concurrency: int = 1
    ocr_model_dir: Path = Path("/models")
    ocr_rec_model_path: Path | None = None
    ocr_rec_keys_path: Path | None = None
    log_level: str = "INFO"
    port: int = 8000

    def apply_rocm_env_overrides(self) -> None:
        """Set required environment overrides for ROCm RDNA3 consumer GPUs."""
        if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = self.hsa_override_gfx_version


settings = Settings()
settings.apply_rocm_env_overrides()

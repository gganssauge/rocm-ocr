"""Unit tests for configuration and environment handling."""

import os
from pathlib import Path

import pytest

from rocm_ocr.config import Settings


def test_settings_default_values() -> None:
    """Verify default configuration values."""
    cfg = Settings()
    assert cfg.hsa_override_gfx_version == "11.0.0"
    assert cfg.ocr_concurrency == 1
    assert cfg.ocr_model_dir == Path("/models")
    assert cfg.ocr_rec_model_path is None
    assert cfg.ocr_rec_keys_path is None
    assert cfg.log_level == "INFO"
    assert cfg.port == 8000


def test_apply_rocm_env_overrides(monkeypatch) -> None:
    """Verify HSA_OVERRIDE_GFX_VERSION is set in environment if not present."""
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    cfg = Settings(hsa_override_gfx_version="11.0.0")
    cfg.apply_rocm_env_overrides()
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"


def test_apply_rocm_env_overrides_preserves_existing(monkeypatch) -> None:
    """Verify existing HSA_OVERRIDE_GFX_VERSION in environment is preserved."""
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.2")
    cfg = Settings(hsa_override_gfx_version="11.0.0")
    cfg.apply_rocm_env_overrides()
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.2"


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("HSA_OVERRIDE_GFX_VERSION", "11.0.0"),
        ("MIOPEN_DEBUG_CONV_WINOGRAD", "0"),
        ("MIOPEN_FIND_MODE", "2"),
    ],
)
def test_all_rocm_defaults_preserve_explicit_environment_values(
    monkeypatch: pytest.MonkeyPatch, name: str, default: str
) -> None:
    """Set missing defaults but preserve user settings, even explicit emptiness."""
    cfg = Settings(hsa_override_gfx_version="11.0.0")
    monkeypatch.delenv(name, raising=False)
    cfg.apply_rocm_env_overrides()
    assert os.environ[name] == default
    for value in ("custom", ""):
        monkeypatch.setenv(name, value)
        cfg.apply_rocm_env_overrides()
        assert os.environ[name] == value

"""Unit tests for OCR engine and coordinate conversions."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from rocm_ocr.config import Settings
from rocm_ocr.engine import (
    ROCM_PROVIDER,
    GpuInferenceError,
    OcrEngine,
    _convert_quad_to_bbox,
    _resolve_model_kwargs,
)
from rocm_ocr.parser import RenderedPage


def test_convert_quad_to_bbox() -> None:
    """Test quad to axis-aligned bounding box conversion and scaling."""
    quad = [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]]
    bbox = _convert_quad_to_bbox(quad, scale=2.0)
    assert bbox == [5.0, 10.0, 25.0, 30.0]


def test_resolve_model_kwargs_empty_dir(tmp_path: Path) -> None:
    """Test resolving model kwargs when directory has no ONNX models."""
    kwargs = _resolve_model_kwargs(tmp_path)
    assert kwargs == {}


def test_resolve_model_kwargs_custom_models(tmp_path: Path) -> None:
    """Test resolving model kwargs when custom ONNX models and dict exist."""
    (tmp_path / "det.onnx").touch()
    (tmp_path / "cls.onnx").touch()
    (tmp_path / "latin_PP-OCRv3_rec_infer.onnx").touch()
    (tmp_path / "latin_dict.txt").touch()
    kwargs = _resolve_model_kwargs(tmp_path)
    assert "det_model_path" in kwargs
    assert "cls_model_path" in kwargs
    assert "rec_model_path" in kwargs
    assert "rec_keys_path" in kwargs
    assert "latin_dict.txt" in kwargs["rec_keys_path"]


def test_resolve_model_kwargs_ppocrv6(tmp_path: Path) -> None:
    """Test resolving model kwargs when PP-OCRv6 models and dict exist."""
    (tmp_path / "ch_PP-OCRv4_det_infer.onnx").touch()
    (tmp_path / "ch_ppocr_mobile_v2.0_cls_infer.onnx").touch()
    (tmp_path / "PP-OCRv6_small_rec.onnx").touch()
    (tmp_path / "ppocrv6_dict.txt").touch()
    kwargs = _resolve_model_kwargs(tmp_path)
    assert "det_model_path" in kwargs
    assert "cls_model_path" in kwargs
    assert kwargs["rec_model_path"].endswith("PP-OCRv6_small_rec.onnx")
    assert kwargs["rec_keys_path"].endswith("ppocrv6_dict.txt")


def test_ocr_engine_is_ready() -> None:
    """Verify that OcrEngine.is_ready() reports true for healthy engine."""
    engine = OcrEngine(Settings())
    assert engine.is_ready() is True
    assert engine.active_provider is not None


def test_ocr_engine_is_ready_failure_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify that OcrEngine.is_ready() returns False and logs warning on failure."""
    engine = OcrEngine(Settings())
    engine._engine = MagicMock(side_effect=RuntimeError("MIOpen compiler failure"))
    with caplog.at_level(logging.WARNING):
        assert engine.is_ready() is False
    assert "Readiness probe check failed: MIOpen compiler failure" in caplog.text


def test_ocr_engine_infer_page_empty() -> None:
    """Verify inference on blank image returns empty bounding boxes."""
    engine = OcrEngine(Settings())
    img = Image.new("RGB", (64, 64), color="white")
    page = RenderedPage(page_number=0, image=img, width=64.0, height=64.0, scale=1.0)
    boxes = engine.infer_page(page)
    assert isinstance(boxes, list)


def test_ocr_engine_gpu_error_fail_fast() -> None:
    """Verify fail-fast GpuInferenceError is raised on ROCm GPU failures."""
    engine = OcrEngine(Settings())
    engine._active_provider = ROCM_PROVIDER
    mock_rapid = MagicMock(
        side_effect=RuntimeError("HIP out of memory error on device gfx1101")
    )
    engine._engine = mock_rapid

    img = Image.new("RGB", (64, 64), color="white")
    page = RenderedPage(page_number=0, image=img, width=64.0, height=64.0, scale=1.0)

    with pytest.raises(GpuInferenceError) as exc_info:
        engine.infer_page(page)

    err = exc_info.value
    assert err.detail == "ROCm GPU inference failed"
    assert err.provider_attempted == ROCM_PROVIDER
    assert "HIP out of memory" in err.error_message

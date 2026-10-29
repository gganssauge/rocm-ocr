"""Unit tests for OCR engine and coordinate conversions."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from rapidocr_onnxruntime.utils.infer_engine import ONNXRuntimeError

from rocm_ocr.config import Settings
from rocm_ocr.engine import (
    ROCM_PROVIDER,
    GpuInferenceError,
    OcrEngine,
    _convert_quad_to_bbox,
    _resolve_model_kwargs,
    create_rapidocr,
)
from rocm_ocr.parser import RenderedPage

pytestmark = pytest.mark.usefixtures("native_rapid")


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
    caplog: pytest.LogCaptureFixture, native_rapid: MagicMock
) -> None:
    """Verify that OcrEngine.is_ready() returns False and logs warning on failure."""
    engine = OcrEngine(Settings())
    native_rapid.side_effect = RuntimeError("MIOpen compiler failure")
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


def test_ocr_engine_gpu_error_fail_fast(native_rapid: MagicMock) -> None:
    """Verify fail-fast GpuInferenceError is raised on ROCm GPU failures."""
    engine = OcrEngine(Settings())
    native_rapid.side_effect = RuntimeError("HIP out of memory error on device gfx1101")

    img = Image.new("RGB", (64, 64), color="white")
    page = RenderedPage(page_number=0, image=img, width=64.0, height=64.0, scale=1.0)

    with pytest.raises(GpuInferenceError) as exc_info:
        engine.infer_page(page)

    err = exc_info.value
    assert err.detail == "ROCm GPU inference failed"
    assert err.provider_attempted == ROCM_PROVIDER
    assert "HIP out of memory" in err.error_message


@pytest.mark.parametrize("stage", ["text_det", "text_cls"])
def test_cpu_fallback_is_never_ready(stage: str, native_rapid: MagicMock) -> None:
    """Both GPU stages must really use ROCm, even if a dummy run would succeed."""
    getattr(native_rapid, stage).infer.session.get_providers.return_value = [
        "CPUExecutionProvider"
    ]
    engine = OcrEngine(Settings())
    assert engine.is_ready() is False


def test_native_stderr_and_original_oserror_remain_visible(
    native_rapid: MagicMock, capfd: pytest.CaptureFixture[str]
) -> None:
    """Never redirect the process-wide stderr descriptor during inference."""

    def native_failure(image: object) -> None:
        os.write(2, b"native diagnostic\n")
        raise OSError("native operating-system failure")

    native_rapid.side_effect = native_failure
    engine = OcrEngine(Settings())
    with Image.new("RGB", (64, 64)) as image:
        page = RenderedPage(0, image, 64, 64, 1)
        with pytest.raises(GpuInferenceError) as error:
            engine.infer_page(page)
    assert error.value.error_message == "native operating-system failure"
    assert "native diagnostic" in capfd.readouterr().err


def test_cached_readiness_rechecks_both_gpu_providers(native_rapid: MagicMock) -> None:
    """A once-healthy engine must not remain ready after a provider change."""
    engine = OcrEngine(Settings())
    assert engine.is_ready() is True
    native_rapid.text_cls.infer.session.get_providers.return_value = [
        "CPUExecutionProvider"
    ]
    assert engine.is_ready() is False


def test_missing_rocm_is_reported_without_constructing_cpu_models(
    native_rapid: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep probes available but refuse inference when ROCm is unavailable."""
    monkeypatch.setattr(
        "rocm_ocr.engine.ort.get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    constructor = MagicMock(return_value=native_rapid)
    monkeypatch.setattr("rocm_ocr.engine.RapidOCR", constructor)
    engine = OcrEngine(Settings())
    constructor.assert_not_called()
    assert engine.is_ready() is False
    assert engine.active_provider == "UnavailableExecutionProvider"
    with (
        Image.new("RGB", (64, 64)) as image,
        pytest.raises(GpuInferenceError, match="ROCm GPU inference failed"),
    ):
        engine.infer_page(RenderedPage(0, image, 64, 64, 1))


def test_rapidocr_native_errors_keep_gpu_error_contract(
    native_rapid: MagicMock,
) -> None:
    """RapidOCR wraps native ORT failures in its own non-RuntimeError exception."""
    engine = OcrEngine(Settings())
    native_rapid.side_effect = ONNXRuntimeError("HIP runtime failure")
    with (
        Image.new("RGB", (64, 64)) as image,
        pytest.raises(GpuInferenceError) as error,
    ):
        engine.infer_page(RenderedPage(0, image, 64, 64, 1))
    assert error.value.error_message == "HIP runtime failure"


def test_provider_setup_is_per_instance_and_recognition_stays_on_cpu(
    native_rapid: MagicMock,
) -> None:
    """Configure named stages explicitly rather than infer stage names from paths."""
    OcrEngine(Settings())
    for stage in (native_rapid.text_det, native_rapid.text_cls):
        providers = stage.infer.session.set_providers.call_args.args[0]
        assert providers[0][0] == ROCM_PROVIDER
        stage.infer.session.disable_fallback.assert_called_once()
    native_rapid.text_rec.infer.session.set_providers.assert_not_called()
    assert native_rapid.text_rec.infer.session.get_providers() == [
        "CPUExecutionProvider"
    ]


def test_shared_factory_refuses_provider_fallback(native_rapid: MagicMock) -> None:
    """Benchmark callers receive the same fail-fast guarantee as the HTTP engine."""
    native_rapid.text_cls.infer.session.get_providers.return_value = [
        "CPUExecutionProvider"
    ]
    with pytest.raises(GpuInferenceError) as error:
        create_rapidocr()
    assert "Classification uses" in error.value.error_message

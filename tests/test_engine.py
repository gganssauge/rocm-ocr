"""Unit tests for OCR engine and coordinate conversions."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from rapidocr_onnxruntime.utils.infer_engine import ONNXRuntimeError

from rocm_ocr.config import Settings
from rocm_ocr.engine import (
    ROCM_PROVIDER,
    GpuInferenceError,
    OcrEngine,
    _convert_quad_to_bbox,
    _find_in_dirs,
    _resolve_dict_path,
    _resolve_model_kwargs,
    create_rapidocr,
)
from rocm_ocr.models import BoundingBox
from rocm_ocr.parser import RenderedPage

pytestmark = pytest.mark.usefixtures("native_rapid")


@pytest.fixture
def native_pipeline(native_rapid: MagicMock) -> MagicMock:
    """Keep the pipeline callable while simulating its external native stages."""

    def run(session: MagicMock, image: object) -> object:
        return session.run(["output"], {"input": image})

    for stage in (native_rapid.text_det, native_rapid.text_cls):
        stage.infer.side_effect = partial(run, stage.infer.session)

    def pipeline(image: object) -> tuple[None, None]:
        for stage in (native_rapid.text_det, native_rapid.text_cls):
            stage.infer(image)
        native_rapid.text_rec(image)
        return None, None

    native_rapid.side_effect = pipeline
    return native_rapid


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


def test_model_lookup_preserves_directory_and_candidate_priority(
    tmp_path: Path,
) -> None:
    """Prefer the first directory, then the first existing filename within it."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "fallback.onnx").touch()
    (second / "preferred.onnx").touch()
    dirs = [tmp_path / "missing", first, second]
    candidates = ["preferred.onnx", "fallback.onnx"]
    assert _find_in_dirs(dirs, candidates) == str(first / "fallback.onnx")
    (first / "preferred.onnx").touch()
    assert _find_in_dirs(dirs, candidates) == str(first / "preferred.onnx")


@pytest.mark.parametrize("valid_override", [False, True])
def test_explicit_recognition_paths_override_discovered_files(
    tmp_path: Path, valid_override: bool
) -> None:
    """Missing overrides must still fall back to discovery; valid ones win."""
    discovered_model = tmp_path / "rec.onnx"
    discovered_dict = tmp_path / "dict.txt"
    explicit_model = tmp_path / "custom.onnx"
    explicit_dict = tmp_path / "custom.txt"
    discovered_model.touch()
    discovered_dict.touch()
    if valid_override:
        explicit_model.touch()
        explicit_dict.touch()
    kwargs = _resolve_model_kwargs(tmp_path, explicit_model, explicit_dict)
    assert kwargs == {
        "rec_model_path": str(explicit_model if valid_override else discovered_model),
        "rec_keys_path": str(explicit_dict if valid_override else discovered_dict),
    }


@pytest.mark.parametrize(
    ("model_name", "dictionary"),
    [
        ("LATIN_PP-OCRv3_rec.onnx", "latin_dict.txt"),
        ("PP-OCRv3_rec.onnx", "latin_dict.txt"),
        ("PP-OCRv6_small_rec.onnx", "ppocrv6_dict.txt"),
    ],
)
def test_dictionary_lookup_preserves_bundled_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    dictionary: str,
) -> None:
    """Prefer local generic dictionaries to bundled ones, without mixing models."""
    monkeypatch.setattr("rocm_ocr.engine.__file__", str(tmp_path / "engine.py"))
    assert _resolve_dict_path([], model_name) is None
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / dictionary).touch()
    assert _resolve_dict_path([], model_name) == str(resources / dictionary)
    (tmp_path / "dict.txt").touch()
    assert _resolve_dict_path([tmp_path], model_name) == str(tmp_path / "dict.txt")
    (tmp_path / dictionary).touch()
    assert _resolve_dict_path([tmp_path], model_name) == str(tmp_path / dictionary)


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
    with Image.new("RGB", (64, 64)) as image:
        page = RenderedPage(0, image, 64, 64, 1)
        with pytest.raises(GpuInferenceError, match="ROCm GPU inference failed"):
            engine.infer_page(page)


def test_rapidocr_native_errors_keep_gpu_error_contract(
    native_rapid: MagicMock,
) -> None:
    """RapidOCR wraps native ORT failures in its own non-RuntimeError exception."""
    engine = OcrEngine(Settings())
    native_rapid.side_effect = ONNXRuntimeError("HIP runtime failure")
    with Image.new("RGB", (64, 64)) as image:
        page = RenderedPage(0, image, 64, 64, 1)
        with pytest.raises(GpuInferenceError) as error:
            engine.infer_page(page)
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


@pytest.mark.parametrize("stage_name", ["text_det", "text_cls"])
def test_concurrent_pages_can_share_a_non_reentrant_gpu_session(
    native_pipeline: MagicMock, stage_name: str
) -> None:
    """Concurrent page workers must not enter the same native GPU session."""
    busy = Lock()
    collision = Event()
    first_call = Event()
    workers = Barrier(2)

    def native_run(outputs: list[str], inputs: dict[str, object]) -> list[object]:
        if not busy.acquire(blocking=False):
            collision.set()
            raise RuntimeError("Concurrent GPU session access")
        try:
            if not first_call.is_set():
                first_call.set()
                collision.wait(timeout=0.5)
            return []
        finally:
            busy.release()

    getattr(native_pipeline, stage_name).infer.session.run.side_effect = native_run
    engine = OcrEngine(Settings())

    def infer() -> list[BoundingBox]:
        with Image.new("RGB", (64, 64)) as image:
            workers.wait(timeout=3)
            return engine.infer_page(RenderedPage(0, image, 64, 64, 1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(infer) for _ in range(2)]
        assert [result.result(timeout=3) for result in results] == [[], []]


def test_cpu_recognition_remains_concurrent(native_pipeline: MagicMock) -> None:
    """GPU serialization must not serialize the entire OCR pipeline."""
    cpu_workers = Barrier(2)
    native_pipeline.text_rec.side_effect = lambda image: cpu_workers.wait(timeout=3)
    engine = OcrEngine(Settings())

    def infer() -> list[BoundingBox]:
        with Image.new("RGB", (64, 64)) as image:
            return engine.infer_page(RenderedPage(0, image, 64, 64, 1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(infer) for _ in range(2)]
        assert [result.result(timeout=5) for result in results] == [[], []]


def test_gpu_stages_do_not_share_a_global_lock(native_pipeline: MagicMock) -> None:
    """Independent detection and classification sessions can execute together."""
    stages = Barrier(2)

    def native_run(outputs: list[str], inputs: dict[str, object]) -> list[object]:
        stages.wait(timeout=3)
        return []

    for stage in (native_pipeline.text_det, native_pipeline.text_cls):
        stage.infer.session.run.side_effect = native_run
    engine = create_rapidocr()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            pool.submit(stage.infer, image)
            for stage in (engine.text_det, engine.text_cls)
        ]
        assert [result.result(timeout=5) for result in results] == [[], []]


@pytest.mark.parametrize("stage_name", ["text_det", "text_cls"])
def test_gpu_failure_releases_the_session_for_the_next_page(
    native_pipeline: MagicMock, stage_name: str
) -> None:
    """Preserve the GPU error and allow subsequent workers to acquire the session."""
    getattr(native_pipeline, stage_name).infer.session.run.side_effect = [
        RuntimeError("native GPU failure"),
        [],
    ]
    engine = OcrEngine(Settings())
    with Image.new("RGB", (64, 64)) as image:
        page = RenderedPage(0, image, 64, 64, 1)
        with pytest.raises(GpuInferenceError) as error:
            engine.infer_page(page)
        assert error.value.error_message == "native GPU failure"
        results: list[list[BoundingBox]] = []
        worker = Thread(
            target=lambda: results.append(engine.infer_page(page)), daemon=True
        )
        worker.start()
        worker.join(timeout=3)
        assert not worker.is_alive(), "GPU session remained locked after failure"
        assert results == [[]]

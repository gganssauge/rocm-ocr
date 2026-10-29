import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.capi import (
    onnxruntime_pybind11_state as ort_errors,  # ty: ignore[unresolved-import]
)
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.utils.infer_engine import ONNXRuntimeError

from rocm_ocr.config import Settings, settings
from rocm_ocr.models import BoundingBox
from rocm_ocr.parser import RenderedPage

ROCM_PROVIDER = "ROCMExecutionProvider"
logger = logging.getLogger(__name__)
# ORT exposes these exception classes from its untyped native extension.
ENGINE_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    RuntimeError,
    ONNXRuntimeError,
    ort_errors.EPFail,
    ort_errors.Fail,
    ort_errors.InvalidArgument,
    ort_errors.InvalidGraph,
    ort_errors.NoSuchFile,
    ort_errors.NotImplemented,
    ort_errors.RuntimeException,
)


class GpuInferenceError(Exception):
    """Raised when GPU hardware, HIP runtime, or execution provider fails."""

    def __init__(
        self, detail: str, provider_attempted: str, error_message: str
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.provider_attempted = provider_attempted
        self.error_message = error_message


def _require_rocm_session(session: ort.InferenceSession, stage_name: str) -> None:
    providers = session.get_providers()
    if not providers or providers[0] != ROCM_PROVIDER:
        raise GpuInferenceError(
            "ROCm GPU inference failed",
            ROCM_PROVIDER,
            f"{stage_name} uses {providers}; required {ROCM_PROVIDER}",
        )


def create_rapidocr(**model_kwargs: str) -> RapidOCR:
    """Build the hybrid pipeline using per-instance public ORT provider APIs."""
    if ROCM_PROVIDER not in ort.get_available_providers():
        raise GpuInferenceError(
            "ROCm GPU initialization failed",
            ROCM_PROVIDER,
            "ROCMExecutionProvider is not available",
        )
    engine = RapidOCR(
        **model_kwargs,
        intra_op_num_threads=4,
        det_use_cuda=False,
        det_use_dml=False,
        cls_use_cuda=False,
        cls_use_dml=False,
        rec_use_cuda=False,
        rec_use_dml=False,
    )
    rocm_options = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "miopen_conv_exhaustive_search": False,
        "do_copy_in_default_stream": True,
    }
    for name, stage in (
        ("Detection", engine.text_det),
        ("Classification", engine.text_cls),
    ):
        stage.infer.session.set_providers([(ROCM_PROVIDER, rocm_options)])
        stage.infer.session.disable_fallback()
        _require_rocm_session(stage.infer.session, name)
    return engine


def _find_candidate_file(directory: Path, candidates: list[str]) -> str | None:
    """Find the first matching candidate file in directory."""
    for cand in candidates:
        cand_path = directory / cand
        if cand_path.is_file():
            return str(cand_path)
    return None


def _find_in_dirs(dirs: list[Path], candidates: list[str]) -> str | None:
    """Find candidate file across a list of directories in order."""
    for d in dirs:
        found = (
            _find_candidate_file(d, candidates) if d.is_file() or d.exists() else None
        )
        if found:
            return found
    return None


def _collect_model_dirs(model_dir: Path) -> list[Path]:
    """Return directories to inspect for models."""
    dirs = [model_dir]
    if model_dir == Path("/models") and not model_dir.exists():
        local_models = Path("models")
        if local_models.exists():
            dirs.append(local_models)
    return [d for d in dirs if d.exists()]


LATIN_DICT_FILENAME = "latin_dict.txt"
PPOCR_V6_DICT_FILENAME = "ppocrv6_dict.txt"


def _resolve_dict_path(dirs: list[Path], rec_model_path: str) -> str | None:
    """Find matching dictionary file for the selected recognition model."""
    rec_model = rec_model_path.lower()
    if "latin" in rec_model or "pp-ocrv3" in rec_model:
        dict_candidates = [LATIN_DICT_FILENAME, "dict.txt"]
        bundled_fallback = Path(__file__).parent / "resources" / LATIN_DICT_FILENAME
    else:
        dict_candidates = [PPOCR_V6_DICT_FILENAME, LATIN_DICT_FILENAME, "dict.txt"]
        bundled_fallback = Path(__file__).parent / "resources" / PPOCR_V6_DICT_FILENAME

    found = _find_in_dirs(dirs, dict_candidates)
    if found:
        return found
    if bundled_fallback.is_file():
        return str(bundled_fallback)
    return None


def _resolve_model_kwargs(
    model_dir: Path,
    rec_model_path: Path | None = None,
    rec_keys_path: Path | None = None,
) -> dict[str, str]:
    """Inspect model directory and map custom ONNX model paths if present."""
    kwargs: dict[str, str] = {}
    if rec_model_path and rec_model_path.is_file():
        kwargs["rec_model_path"] = str(rec_model_path)
    if rec_keys_path and rec_keys_path.is_file():
        kwargs["rec_keys_path"] = str(rec_keys_path)

    dirs = _collect_model_dirs(model_dir)

    lookup_map = [
        ("det_model_path", ["ch_PP-OCRv4_det_infer.onnx", "det.onnx"]),
        ("cls_model_path", ["ch_ppocr_mobile_v2.0_cls_infer.onnx", "cls.onnx"]),
        (
            "rec_model_path",
            [
                "PP-OCRv6_small_rec.onnx",
                "PP-OCRv6_small_rec_infer.onnx",
                "PP-OCRv6_medium_rec.onnx",
                "PP-OCRv6_rec.onnx",
                "latin_PP-OCRv3_rec_infer.onnx",
                "rec_latin.onnx",
                "rec.onnx",
            ],
        ),
    ]
    for key, candidates in lookup_map:
        if key not in kwargs:
            found = _find_in_dirs(dirs, candidates)
            if found:
                kwargs[key] = found

    if "rec_keys_path" not in kwargs and "rec_model_path" in kwargs:
        dict_path = _resolve_dict_path(dirs, kwargs["rec_model_path"])
        if dict_path:
            kwargs["rec_keys_path"] = dict_path
    return kwargs


def _convert_quad_to_bbox(quad: Any, scale: float) -> list[float]:
    """Convert four-point polygon to [x0, y0, x1, y1] normalized to page scale."""
    x_coords = [pt[0] for pt in quad]
    y_coords = [pt[1] for pt in quad]
    return [
        round(min(x_coords) / scale, 2),
        round(min(y_coords) / scale, 2),
        round(max(x_coords) / scale, 2),
        round(max(y_coords) / scale, 2),
    ]


class OcrEngine:
    """Manages the RapidOCR lifecycle and ROCm execution provider interactions."""

    def __init__(self, cfg: Settings = settings) -> None:
        """Initialize hybrid sessions, retaining startup errors for HTTP 503."""
        self._config = cfg
        self._config.apply_rocm_env_overrides()
        model_kwargs = _resolve_model_kwargs(
            self._config.ocr_model_dir,
            self._config.ocr_rec_model_path,
            self._config.ocr_rec_keys_path,
        )
        self._engine: RapidOCR | None = None
        self._is_ready = False
        self._initialization_error = ""
        try:
            self._engine = create_rapidocr(**model_kwargs)
        except (GpuInferenceError, *ENGINE_ERRORS) as exc:
            self._initialization_error = str(exc)
            if isinstance(exc, GpuInferenceError):
                self._initialization_error = exc.error_message
            logger.error(
                "OCR engine initialization failed: %s", self._initialization_error
            )

    def _require_gpu(self) -> RapidOCR:
        """Reject unavailable engines and fallback in either GPU stage."""
        if self._engine is None:
            raise GpuInferenceError(
                "ROCm GPU inference failed", ROCM_PROVIDER, self._initialization_error
            )
        for name, stage in (
            ("Detection", self._engine.text_det),
            ("Classification", self._engine.text_cls),
        ):
            _require_rocm_session(stage.infer.session, name)
        return self._engine

    @property
    def active_provider(self) -> str:
        """Name of the active ONNX Runtime execution provider."""
        if self._engine is None:
            return "UnavailableExecutionProvider"
        providers = self._engine.text_det.infer.session.get_providers()
        return str(providers[0]) if providers else "UnavailableExecutionProvider"

    def is_ready(self) -> bool:
        """Verify that the engine and execution provider can perform inference."""
        try:
            engine = self._require_gpu()
            if not self._is_ready:
                dummy_arr = np.ones((64, 64, 3), dtype=np.uint8) * 255
                engine(dummy_arr)
                self._require_gpu()
                self._is_ready = True
            return True
        except (GpuInferenceError, *ENGINE_ERRORS) as exc:
            self._is_ready = False
            detail = (
                exc.error_message if isinstance(exc, GpuInferenceError) else str(exc)
            )
            logger.warning("Readiness probe check failed: %s", detail, exc_info=True)
            return False

    def infer_page(self, page: RenderedPage) -> list[BoundingBox]:
        """Execute OCR detection, orientation classification, and text recognition."""
        try:
            engine = self._require_gpu()
            raw_results, _ = engine(np.array(page.image))
            self._require_gpu()
        except GpuInferenceError:
            self._is_ready = False
            raise
        except ENGINE_ERRORS as exc:
            self._is_ready = False
            raise GpuInferenceError(
                detail="ROCm GPU inference failed",
                provider_attempted=ROCM_PROVIDER,
                error_message=str(exc),
            ) from exc

        if not raw_results:
            return []

        boxes: list[BoundingBox] = []
        for item in raw_results:
            quad, text, score = item[0], str(item[1]), float(item[2])
            bbox = _convert_quad_to_bbox(quad, page.scale)
            conf = max(0.0, min(1.0, round(score, 4)))
            boxes.append(BoundingBox(text=text, bbox=bbox, confidence=conf))
        return boxes

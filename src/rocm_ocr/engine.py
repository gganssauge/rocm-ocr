"""OCR inference engine wrapping RapidOCR and ONNX Runtime with ROCm support."""

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.utils.infer_engine import OrtInferSession

from rocm_ocr.config import Settings, settings
from rocm_ocr.models import BoundingBox
from rocm_ocr.parser import RenderedPage

ROCM_PROVIDER = "ROCMExecutionProvider"


class GpuInferenceError(Exception):
    """Raised when GPU hardware, HIP runtime, or execution provider fails."""

    def __init__(
        self, detail: str, provider_attempted: str, error_message: str
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.provider_attempted = provider_attempted
        self.error_message = error_message


def _patch_rapidocr_rocm() -> None:
    """Ensure OrtInferSession prioritizes ROCMExecutionProvider when available."""
    original_get_ep_list = OrtInferSession._get_ep_list

    def _rocm_get_ep_list(self: OrtInferSession) -> list[tuple[str, dict[str, Any]]]:
        ep_list = original_get_ep_list(self)
        available = ort.get_available_providers()
        if ROCM_PROVIDER in available:
            rocm_opts = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "miopen_conv_exhaustive_search": True,
                "do_copy_in_default_stream": True,
            }
            ep_list.insert(0, (ROCM_PROVIDER, rocm_opts))
        return ep_list

    OrtInferSession._get_ep_list = _rocm_get_ep_list


def _resolve_model_kwargs(model_dir: Path) -> dict[str, str]:
    """Inspect model directory and map custom ONNX model paths if present."""
    kwargs: dict[str, str] = {}
    if not model_dir.exists():
        return kwargs

    det_candidates = ["ch_PP-OCRv4_det_infer.onnx", "det.onnx"]
    cls_candidates = ["ch_ppocr_mobile_v2.0_cls_infer.onnx", "cls.onnx"]
    rec_candidates = ["ch_PP-OCRv4_rec_infer.onnx", "rec.onnx"]

    for cand in det_candidates:
        if (model_dir / cand).is_file():
            kwargs["det_model_path"] = str(model_dir / cand)
            break
    for cand in cls_candidates:
        if (model_dir / cand).is_file():
            kwargs["cls_model_path"] = str(model_dir / cand)
            break
    for cand in rec_candidates:
        if (model_dir / cand).is_file():
            kwargs["rec_model_path"] = str(model_dir / cand)
            break
    return kwargs


def _convert_quad_to_bbox(quad: list[list[float]], scale: float) -> list[float]:
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
        """Initialize the OCR engine with configuration and provider patch."""
        self._config = cfg
        self._config.apply_rocm_env_overrides()
        _patch_rapidocr_rocm()

        model_kwargs = _resolve_model_kwargs(self._config.ocr_model_dir)
        self._engine = RapidOCR(**model_kwargs)
        self._active_provider = self._detect_active_provider()

    def _detect_active_provider(self) -> str:
        """Detect the currently loaded execution provider from the active session."""
        try:
            providers = self._engine.text_det.infer.session.get_providers()
            return str(providers[0]) if providers else "UnknownExecutionProvider"
        except Exception:
            return "UnknownExecutionProvider"

    @property
    def active_provider(self) -> str:
        """Name of the active ONNX Runtime execution provider."""
        return self._active_provider

    def is_ready(self) -> bool:
        """Verify that the engine and execution provider can perform inference."""
        try:
            dummy_arr = np.ones((64, 64, 3), dtype=np.uint8) * 255
            self._engine(dummy_arr)
            return True
        except Exception:
            return False

    def infer_page(self, page: RenderedPage) -> list[BoundingBox]:
        """Execute OCR detection, orientation classification, and text recognition."""
        img_np = np.array(page.image)
        try:
            raw_results, _ = self._engine(img_np)
        except Exception as exc:
            if self._active_provider == ROCM_PROVIDER or "HIP" in str(exc):
                raise GpuInferenceError(
                    detail="ROCm GPU inference failed",
                    provider_attempted=ROCM_PROVIDER,
                    error_message=str(exc),
                ) from exc
            raise

        if not raw_results:
            return []

        boxes: list[BoundingBox] = []
        for item in raw_results:
            quad, text, score = item[0], str(item[1]), float(item[2])
            bbox = _convert_quad_to_bbox(quad, page.scale)
            boxes.append(BoundingBox(text=text, bbox=bbox, confidence=round(score, 4)))
        return boxes

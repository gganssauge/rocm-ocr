import os
from collections.abc import Generator
from contextlib import contextmanager
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


@contextmanager
def _silence_c_stderr() -> Generator[None, None, None]:
    """Redirect C-level stderr (fd 2) to devnull to suppress MIOpen search logs."""
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr = os.dup(2)
        try:
            os.dup2(null_fd, 2)
            yield
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            os.close(null_fd)
    except OSError:
        yield


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
    original_init = OrtInferSession.__init__
    original_get_ep_list = OrtInferSession._get_ep_list

    def _rocm_init(self: OrtInferSession, config: dict[str, Any]) -> None:
        setattr(self, "_model_path", str(config.get("model_path", "")))  # noqa: B010
        if config.get("intra_op_num_threads", -1) == -1:
            config["intra_op_num_threads"] = 4
        original_init(self, config)

    def _rocm_get_ep_list(self: OrtInferSession) -> list[tuple[str, dict[str, Any]]]:
        ep_list = original_get_ep_list(self)
        available = ort.get_available_providers()
        model_str = getattr(self, "_model_path", "").lower()
        is_rec = (
            "_rec" in model_str
            or "latin" in model_str
            or "rec_" in model_str
            or "rec.onnx" in model_str
        )
        if is_rec:
            # Recognition models execute on CPU (intra_op_num_threads=4) to prevent
            # RDNA3 kernel issues and preserve GPU VRAM for concurrent LLMs (Qwen2.5).
            return [ep for ep in ep_list if ep[0] != ROCM_PROVIDER]

        if ROCM_PROVIDER in available:
            rocm_opts = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "miopen_conv_exhaustive_search": False,
                "do_copy_in_default_stream": True,
            }
            ep_list = [ep for ep in ep_list if ep[0] != ROCM_PROVIDER]
            ep_list.insert(0, (ROCM_PROVIDER, rocm_opts))
        return ep_list

    OrtInferSession.__init__ = _rocm_init
    OrtInferSession._get_ep_list = _rocm_get_ep_list


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
        """Initialize the OCR engine with configuration and provider patch."""
        self._config = cfg
        self._config.apply_rocm_env_overrides()
        _patch_rapidocr_rocm()

        model_kwargs = _resolve_model_kwargs(
            self._config.ocr_model_dir,
            self._config.ocr_rec_model_path,
            self._config.ocr_rec_keys_path,
        )
        self._engine = RapidOCR(**model_kwargs)
        self._active_provider = self._detect_active_provider()
        self._is_ready = False

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
        if self._is_ready:
            return True
        try:
            dummy_arr = np.ones((64, 64, 3), dtype=np.uint8) * 255
            with _silence_c_stderr():
                self._engine(dummy_arr)
            self._is_ready = True
            return True
        except Exception:
            return False

    def infer_page(self, page: RenderedPage) -> list[BoundingBox]:
        """Execute OCR detection, orientation classification, and text recognition."""
        img_np = np.array(page.image)
        try:
            with _silence_c_stderr():
                raw_results, _ = self._engine(img_np)
        except Exception as exc:
            self._is_ready = False
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
            conf = max(0.0, min(1.0, round(score, 4)))
            boxes.append(BoundingBox(text=text, bbox=bbox, confidence=conf))
        return boxes

"""Shared native OCR boundary for GPU-independent unit and HTTP tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def native_rapid(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Keep application logic real while replacing native model execution."""
    rapid = MagicMock(return_value=(None, None))
    for stage in (rapid.text_det, rapid.text_cls):
        stage.infer.session.get_providers.return_value = ["ROCMExecutionProvider"]
    rapid.text_rec.infer.session.get_providers.return_value = ["CPUExecutionProvider"]
    monkeypatch.setattr("rocm_ocr.engine.RapidOCR", MagicMock(return_value=rapid))
    monkeypatch.setattr(
        "rocm_ocr.engine.ort.get_available_providers",
        lambda: ["ROCMExecutionProvider", "CPUExecutionProvider"],
    )
    return rapid

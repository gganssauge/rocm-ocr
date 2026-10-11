"""Tests for the Dockerfile digest and version updater script."""

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure root directory is on sys.path so scripts can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.update_docker_digests import (  # noqa: E402
    AssetInfo,
    _update_instruction_content,
    update_dockerfile,
)

CHK_1 = "sha256:" + "1" * 64
CHK_FAV = "sha256:d16f72cf5a2773eba499cc7f3db1923c6f5f73de990f739e3a05806aafec15b7"
CHK_LAT = "sha256:4fd81948e80f385afc2a9410a51904b70b6ff6dae9e26c2fc54c4bc52d4124d6"
HF_URL = (
    "https://huggingface.co/breezedeus/cnocr-ppocr-latin_PP-OCRv3"
    "/resolve/f6567a4a5d5ed30129b5ff7ffe5b226cd2678644/latin_PP-OCRv3_rec_infer.onnx"
)

SAMPLE_DOCKERFILE = (
    "# syntax=docker/dockerfile:1\n"
    "FROM ubuntu:rolling AS builder\n\n"
    f"ADD --checksum={CHK_1} \\\n"
    "    https://example.com/asset-v1.js \\\n"
    "    /app/src/rocm_ocr/static/scalar.js\n\n"
    f"ADD --checksum={CHK_FAV} \\\n"
    "    https://fastapi.tiangolo.com/img/favicon.png \\\n"
    "    /app/src/rocm_ocr/static/favicon.png\n\n"
    f"ADD --checksum={CHK_LAT} \\\n"
    f"    {HF_URL} \\\n"
    "    /models/latin_PP-OCRv3_rec_infer.onnx\n"
)


def test_update_instruction_content_unchanged() -> None:
    """Verify that matching digest and URL report no change."""
    asset = AssetInfo(
        name="FastAPI favicon",
        destination="/app/src/rocm_ocr/static/favicon.png",
        url="https://fastapi.tiangolo.com/img/favicon.png",
        digest=CHK_FAV.removeprefix("sha256:"),
    )
    updated, changed = _update_instruction_content(SAMPLE_DOCKERFILE, asset)
    assert not changed
    assert updated == SAMPLE_DOCKERFILE


def test_update_instruction_content_changed() -> None:
    """Verify that updated digest and URL modify instruction block correctly."""
    new_digest = "2" * 64
    new_url = "https://example.com/asset-v2.js"
    asset = AssetInfo(
        name="Scalar standalone JS",
        destination="/app/src/rocm_ocr/static/scalar.js",
        url=new_url,
        digest=new_digest,
    )
    updated, changed = _update_instruction_content(SAMPLE_DOCKERFILE, asset)
    assert changed
    assert f"sha256:{new_digest}" in updated
    assert new_url in updated
    assert "/app/src/rocm_ocr/static/scalar.js" in updated


def test_update_dockerfile_check_mode(tmp_path: Path) -> None:
    """Verify update_dockerfile detects drift in check_only mode and returns 1."""
    df_path = tmp_path / "Dockerfile"
    df_path.write_text(SAMPLE_DOCKERFILE, encoding="utf-8")

    fake_assets = [
        AssetInfo(
            name="Scalar standalone JS",
            destination="/app/src/rocm_ocr/static/scalar.js",
            url="https://example.com/asset-v2.js",
            digest="2" * 64,
        ),
        AssetInfo(
            name="FastAPI favicon",
            destination="/app/src/rocm_ocr/static/favicon.png",
            url="https://fastapi.tiangolo.com/img/favicon.png",
            digest=CHK_FAV.removeprefix("sha256:"),
        ),
        AssetInfo(
            name="Latin PP-OCRv3 model",
            destination="/models/latin_PP-OCRv3_rec_infer.onnx",
            url=HF_URL,
            digest=CHK_LAT.removeprefix("sha256:"),
        ),
    ]

    with (
        patch(
            "scripts.update_docker_digests._resolve_scalar",
            return_value=fake_assets[0],
        ),
        patch(
            "scripts.update_docker_digests._resolve_favicon",
            return_value=fake_assets[1],
        ),
        patch(
            "scripts.update_docker_digests._resolve_latin_model",
            return_value=fake_assets[2],
        ),
    ):
        exit_code = update_dockerfile(df_path, check_only=True)
        assert exit_code == 1
        # File should remain unmodified in check mode
        assert df_path.read_text(encoding="utf-8") == SAMPLE_DOCKERFILE

        # Run again without check mode -> updates file and returns 0
        update_code = update_dockerfile(df_path, check_only=False)
        assert update_code == 0
        assert "asset-v2.js" in df_path.read_text(encoding="utf-8")

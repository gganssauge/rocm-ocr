#!/usr/bin/env python3
"""Maintain and update checksums and versions in Dockerfile ADD instructions."""

import argparse
import hashlib
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetInfo:
    """Information about an asset resolved from remote upstream."""

    name: str
    destination: str
    url: str
    digest: str


def _fetch_stream_sha256(url: str) -> str:
    """Download content from URL and compute its hex SHA-256 digest."""
    hasher = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "rocm-ocr-updater"})
    with urllib.request.urlopen(req) as response:
        while chunk := response.read(64 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_scalar() -> AssetInfo:
    """Resolve latest Scalar standalone JS version and digest from jsdelivr."""
    probe_url = (
        "https://cdn.jsdelivr.net/npm/@scalar/api-reference@latest"
        "/dist/browser/standalone.js"
    )
    req = urllib.request.Request(
        probe_url, method="HEAD", headers={"User-Agent": "rocm-ocr-updater"}
    )
    with urllib.request.urlopen(req) as resp:
        version = resp.headers.get("x-jsd-version")
        if not version:
            raise RuntimeError("Could not determine x-jsd-version from jsdelivr")

    pinned_url = (
        f"https://cdn.jsdelivr.net/npm/@scalar/api-reference@{version}"
        "/dist/browser/standalone.js"
    )
    digest = _fetch_stream_sha256(pinned_url)
    return AssetInfo(
        name="Scalar standalone JS",
        destination="/app/src/rocm_ocr/static/scalar.js",
        url=pinned_url,
        digest=digest,
    )


def _resolve_favicon() -> AssetInfo:
    """Resolve FastAPI favicon digest."""
    url = "https://fastapi.tiangolo.com/img/favicon.png"
    digest = _fetch_stream_sha256(url)
    return AssetInfo(
        name="FastAPI favicon",
        destination="/app/src/rocm_ocr/static/favicon.png",
        url=url,
        digest=digest,
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent automatic redirect following to inspect HTTP 302 headers."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def _resolve_recognition_model() -> AssetInfo:
    """Resolve PP-OCRv6 recognition model commit hash and digest."""
    probe_url = (
        "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx"
        "/resolve/main/inference.onnx"
    )
    req = urllib.request.Request(
        probe_url, method="HEAD", headers={"User-Agent": "rocm-ocr-updater"}
    )
    commit = None
    opener = urllib.request.build_opener(_NoRedirectHandler())
    try:
        with opener.open(req) as resp:
            commit = resp.headers.get("x-repo-commit")
    except urllib.error.HTTPError as exc:
        commit = exc.headers.get("x-repo-commit")

    if not commit:
        raise RuntimeError("Could not determine x-repo-commit from Hugging Face")

    pinned_url = (
        f"https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx"
        f"/resolve/{commit}/inference.onnx"
    )
    digest = _fetch_stream_sha256(pinned_url)
    return AssetInfo(
        name="PP-OCRv6 small recognition model",
        destination="/models/PP-OCRv6_small_rec.onnx",
        url=pinned_url,
        digest=digest,
    )


def _update_instruction_content(content: str, asset: AssetInfo) -> tuple[str, bool]:
    """Replace ADD instruction for target destination if url or digest changed."""
    pattern = re.compile(
        r"ADD\s+--checksum=sha256:([0-9a-fA-F]{64})\s+\\\s*\n\s*(\S+)\s+\\\s*\n\s*"
        + re.escape(asset.destination),
        re.MULTILINE,
    )

    match = pattern.search(content)
    if not match:
        raise ValueError(
            f"Could not find ADD instruction for destination: {asset.destination}"
        )

    old_hash, old_url = match.group(1), match.group(2)
    if old_hash == asset.digest and old_url == asset.url:
        return content, False

    new_instruction = (
        f"ADD --checksum=sha256:{asset.digest} \\\n"
        f"    {asset.url} \\\n"
        f"    {asset.destination}"
    )
    updated = content[: match.start()] + new_instruction + content[match.end() :]
    return updated, True


def update_dockerfile(dockerfile_path: Path, check_only: bool = False) -> int:
    """Check or update ADD instructions in Dockerfile to match upstream assets."""
    content = dockerfile_path.read_text(encoding="utf-8")
    resolvers = [_resolve_scalar, _resolve_favicon, _resolve_recognition_model]

    any_changed = False
    for resolve_fn in resolvers:
        asset = resolve_fn()
        content, changed = _update_instruction_content(content, asset)
        if changed:
            any_changed = True
            action = "Needs update" if check_only else "Updated"
            print(
                f"[{action}] {asset.name} ({asset.destination}) -> "
                f"sha256:{asset.digest}"
            )
        else:
            print(f"[Up to date] {asset.name} ({asset.destination})")

    if not any_changed:
        print(f"All assets in {dockerfile_path} are up to date.")
        return 0

    if check_only:
        print(f"Dockerfile {dockerfile_path} has outdated digests or versions.")
        return 1

    dockerfile_path.write_text(content, encoding="utf-8")
    print(f"Successfully updated {dockerfile_path}.")
    return 0


def main() -> int:
    """Parse CLI options and execute digest update."""
    parser = argparse.ArgumentParser(
        description="Update checksums and versions in Dockerfile ADD instructions."
    )
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=Path("Dockerfile"),
        help="Path to the Dockerfile to inspect and update (default: Dockerfile).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only without modifying; exit 1 if changes are needed.",
    )
    args = parser.parse_args()
    return update_dockerfile(args.dockerfile, check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())

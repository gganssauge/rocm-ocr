#!/usr/bin/env python3
"""In-cluster E2E integration test runner for ROCm OCR Kubernetes deployment."""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_NAMESPACE = "ocr"
DEFAULT_SERVICE = "rocm-ocr"
DEFAULT_TIMEOUT_SEC = 30.0
INVOICE_FIXTURE = Path("tests/fixtures/sample_german_invoice.pdf")


def find_free_port() -> int:
    """Find a random available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_endpoint(
    url: str, expected_status: int = 200, timeout: float = 30.0
) -> None:
    """Poll an HTTP endpoint until it returns the expected status code."""
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=3.0)
            if resp.status_code == expected_status:
                return
        except Exception as exc:
            last_err = exc
        time.sleep(1.0)
    raise TimeoutError(
        f"Endpoint {url} did not become ready within {timeout}s: {last_err}"
    )


def verify_inference_result(data: dict[str, Any]) -> None:
    """Validate GPU execution provider, latency, and German character accuracy."""
    assert data["execution_provider"] == "ROCMExecutionProvider", (
        f"Expected ROCMExecutionProvider, got {data.get('execution_provider')}"
    )
    assert data["inference_time_ms"] > 0, "Expected non-zero inference latency"
    assert len(data["pages"]) == 1, f"Expected 1 page, got {len(data['pages'])}"

    first_page = data["pages"][0]
    extracted_text = " ".join(box["text"] for box in first_page["boxes"])

    required_tokens = ["München", "Kurfürstendamm", "Fälligkeitsdatum", "Umsatzsteuer"]
    for token in required_tokens:
        assert token in extracted_text, (
            f"Token '{token}' not found in OCR output: {extracted_text}"
        )


def execute_e2e_predict(
    base_url: str, invoice_path: Path, correlation_id: str | None = None
) -> tuple[dict[str, Any], str | None]:
    """Send neutral German invoice to predict endpoint and return response data."""
    if not invoice_path.is_file():
        raise FileNotFoundError(f"Invoice fixture missing: {invoice_path}")

    files = {"file": (invoice_path.name, invoice_path.read_bytes(), "application/pdf")}
    headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
    resp = httpx.post(
        f"{base_url}/ocr/v1/predict", files=files, headers=headers, timeout=60.0
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Predict failed HTTP {resp.status_code}: {resp.text}")
    return resp.json(), resp.headers.get("X-Correlation-ID")


def parse_args() -> argparse.Namespace:
    """Parse command line flags for in-cluster E2E test runner."""
    parser = argparse.ArgumentParser(description="ROCm OCR Kubernetes E2E Runner")
    parser.add_argument(
        "--namespace", default=DEFAULT_NAMESPACE, help="Kubernetes namespace"
    )
    parser.add_argument(
        "--service", default=DEFAULT_SERVICE, help="Kubernetes service name"
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep k8s deployment after test"
    )
    parser.add_argument(
        "--port", type=int, default=0, help="Local port (default: random)"
    )
    return parser.parse_args()


def run_test_suite(base_url: str, fixture_path: Path) -> None:
    """Execute health checks and prediction verification against cluster service."""
    print(f"Verifying /healthz on {base_url}...")
    wait_for_endpoint(
        f"{base_url}/healthz", expected_status=200, timeout=DEFAULT_TIMEOUT_SEC
    )
    print("✓ /healthz OK")

    print(f"Verifying /readyz on {base_url}...")
    wait_for_endpoint(
        f"{base_url}/readyz", expected_status=200, timeout=DEFAULT_TIMEOUT_SEC
    )
    print("✓ /readyz OK")

    expected_cid = "e2e-cluster-trace-42"
    print(f"Sending inference request with X-Correlation-ID: {expected_cid}...")
    start_time = time.time()
    payload, resp_cid = execute_e2e_predict(
        base_url, fixture_path, correlation_id=expected_cid
    )
    elapsed = (time.time() - start_time) * 1000

    assert resp_cid == expected_cid, f"Correlation ID mismatch: {resp_cid}"
    verify_inference_result(payload)
    print(
        f"✓ Inference successful in {elapsed:.1f}ms "
        f"(reported GPU latency: {payload['inference_time_ms']}ms)"
    )
    print(f"✓ Execution Provider: {payload['execution_provider']}")
    print(f"✓ Correlation ID verified: {resp_cid}")


def main() -> int:
    """Entry point orchestrating port-forward, tests, and cleanup."""
    args = parse_args()
    local_port = args.port or find_free_port()
    base_url = f"http://127.0.0.1:{local_port}"

    print(
        f"Starting kubectl port-forward to svc/{args.service} on port {local_port}..."
    )
    pf_cmd = [
        "kubectl",
        "port-forward",
        f"svc/{args.service}",
        f"{local_port}:8000",
        "-n",
        args.namespace,
    ]
    pf_proc = subprocess.Popen(pf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        run_test_suite(base_url, INVOICE_FIXTURE)
        print("🎉 ALL IN-CLUSTER E2E CHECKS PASSED!")
        return 0
    finally:
        pf_proc.terminate()
        pf_proc.wait(timeout=5.0)
        if not args.keep:
            print("Cleaning up deployment resources (--keep not set)...")
            subprocess.run(["kubectl", "delete", "-k", "deploy/k8s/"], check=False)
            print("Cleanup completed.")


if __name__ == "__main__":
    sys.exit(main())

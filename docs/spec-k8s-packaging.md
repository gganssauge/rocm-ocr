# Specification: Container Packaging & Kubernetes Deployment for ROCm OCR Inference Microservice

## Problem Statement

The ROCm OCR microservice is currently implemented and verified locally on the host machine using Python 3.12 and ROCm 7.x libraries with AMD Radeon RX 7800 XT GPU acceleration. However, it cannot yet be executed in cloud-native or Kubernetes environments because no container image exists, and Kubernetes deployment manifests are missing.

Furthermore, in the target single-node Kubernetes cluster (`k3s` on `helios.gorki120.de`), an existing LLM workload (`ollama`) already occupies the cluster's sole physical AMD GPU via the standard `amd.com/gpu: 1` allocation from the AMD GPU device plugin. Standard device plugin scheduling would cause the OCR pod to remain perpetually `Pending`.

Therefore, the service requires an automated, self-contained container build and Kubernetes deployment mechanism that enables simultaneous GPU access alongside Ollama, together with end-to-end testing against the live Kubernetes service.

## Solution

1. Provide an optimized, multi-stage `Dockerfile` based on `ubuntu:24.04` that includes required ROCm user-space runtime libraries (`librocm-smi64-7`, `libamdhip64-7`, `libhipblas3`, `libhipfft0`), compatibility shims for ONNX Runtime (`librocm_smi64.so.1` and `libroctracer64.so.4`), self-contained OCR models (`ch_PP-OCRv4_det_infer.onnx`, `PP-OCRv6_small_rec.onnx`, `ppocrv6_dict.txt`), and production runtime configuration.
2. Provide GitOps-ready Kubernetes Kustomize manifests under `deploy/k8s/` configured for the `ocr` namespace. To allow simultaneous GPU operation with existing workloads without device-plugin locking, the pod mounts host render nodes (`/dev/kfd` and `/dev/dri`) with GID supplementary groups (`video: 44`, `render: 109`).
3. Provide an end-to-end integration test runner that builds the container image, imports it into the local `k3s` containerd runtime, deploys the service, validates live HTTP endpoints over a port-forwarded connection, verifies `ROCMExecutionProvider` GPU inference with German umlauts, and manages lifecycle teardown.

## User Stories

1. As a cluster operator, I want a self-contained container image for the OCR service, so that I can deploy it in Kubernetes without installing host Python dependencies or external volume mounts for models.
2. As a cluster operator, I want the container image size to remain minimal and fast to pull, so that deployment rollouts and test execution do not incur multithreaded multi-gigabyte download overhead.
3. As a cluster operator, I want the OCR service to run simultaneously on the AMD Radeon RX 7800 XT alongside existing GPU workloads such as Ollama, so that neither service starves or blocks the single physical GPU.
4. As a Kubernetes administrator, I want standard Kubernetes health checks (`/healthz` for liveness and `/readyz` for readiness), so that the orchestrator accurately monitors pod availability and manages traffic routing.
5. As an infrastructure engineer, I want Kustomize-based deployment manifests in the repository, so that they can be directly committed or synced into our FluxCD GitOps homelab repository (`~/homelab`).
6. As a DevOps engineer, I want an automated end-to-end test script, so that I can verify the complete pipeline (image build -> cluster import -> deployment -> live inference -> verification) in a single command.
7. As a quality engineer, I want the Kubernetes E2E test to validate real German business documents, so that character accuracy for umlauts (`ä, ö, ü, Ä, Ö, Ü, ß`) and DIN 5008 layouts is guaranteed in the cluster container runtime.
8. As a site reliability engineer, I want the Kubernetes service to fail fast with HTTP 503 if GPU access is broken or misconfigured inside the container, rather than silently degrading to slow CPU inference.

## Implementation Decisions

1. **Base Image & Dependency Sizing**:
   - The container image uses a multi-stage build starting from `ubuntu:24.04`.
   - The build stage installs `uv` to resolve and install Python 3.12 dependencies (`fastapi`, `uvicorn`, `onnxruntime-rocm>=1.20.0`, `rapidocr-onnxruntime`, `pymupdf`, `pillow`) with dependency overrides excluding CPU `onnxruntime`.
   - The runtime stage installs minimal ROCm runtime packages (`librocm-smi64-7`, `libamdhip64-7`, `libhipblas3`, `libhipfft0`) and essential system utilities, avoiding heavy compiler and dev toolchains.
   - Compatibility shims (`librocm_smi64.so.1` linked to system SMI and a stub `libroctracer64.so.4`) are placed in the application's library search path.

2. **Self-Contained Model Packaging**:
   - Detection model (`ch_PP-OCRv4_det_infer.onnx`), recognition model (`PP-OCRv6_small_rec.onnx`), and multilingual dictionary (`ppocrv6_dict.txt`) are copied into `/models` within the container.
   - Default environment variable `OCR_MODEL_DIR=/models` is baked into the image.

3. **GPU Hardware Access in Kubernetes**:
   - Rather than requesting `resources.limits: { amd.com/gpu: 1 }` which blocks when another pod holds the single physical GPU device counter, the deployment uses direct host device volume mounts:
     - `/dev/kfd` (character device)
     - `/dev/dri` (host directory)
   - The pod security context specifies `supplementalGroups: [44, 109]` matching the host's `video` and `render` groups.
   - Linux kernel AMDGPU/KFD driver performs hardware time-slicing and multiplexing between Ollama and OCR.

4. **Kubernetes Resources & Manifest Layout**:
   - Manifest directory structured as `deploy/k8s/` containing:
     - `namespace.yaml` (`name: ocr`)
     - `deployment.yaml` (`name: rocm-ocr`, port `8000`, `livenessProbe` at `/healthz`, `readinessProbe` at `/readyz`)
     - `service.yaml` (ClusterIP, port `8000`)
     - `kustomization.yaml` aggregating all resources.

5. **Image Import & E2E Test Pipeline**:
   - For local `k3s` execution, the image is saved from Docker and imported into `k3s` containerd using `k3s ctr images import`.
   - An orchestration script executes:
     1. Image build with tag `rocm-ocr:test`.
     2. Streamed import into `k3s` containerd.
     3. Deployment rollout with `kubectl apply -k deploy/k8s/`.
     4. Rollout wait (`kubectl rollout status`).
     5. Port-forward connection establishment.
     6. HTTP E2E test verification (`/healthz`, `/readyz`, `/ocr/v1/predict` on DIN 5008 test invoice).
     7. Teardown of test deployment (with optional `--keep` flag).

## Testing Decisions

- **Test Seam**: The highest possible architectural seam is tested: external HTTP API requests against the deployed Kubernetes `Service` exposed via port-forwarding.
- **Assertion Coverage**:
  - Service reports `status: "healthy"` on `/healthz` and readiness on `/readyz`.
  - Service reports `execution_provider: "ROCMExecutionProvider"` on `/ocr/v1/predict`.
  - Total inference duration per page remains well below baseline thresholds.
  - Extracted document text contains all German special characters (`München`, `Kurfürstendamm`, `Fälligkeitsdatum`).
- **Prior Art**: Follows existing E2E integration test semantics established in `tests/test_e2e.py`.

## Out of Scope

- Multi-GPU cluster scheduling across heterogeneous worker nodes.
- Ingress controller configuration and external TLS certificate management (handled centrally by cluster GitOps).
- Automatic database ingestion or downstream document management integrations (strictly domain-agnostic).

## Further Notes

- The resulting manifests under `deploy/k8s/` are designed for direct inclusion or reference within the operator's `~/homelab` FluxCD repository.

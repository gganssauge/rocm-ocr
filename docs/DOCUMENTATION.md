# ROCm OCR Microservice - Dokumentation

High-Performance OCR-Inferenz-Microservice, hardwarebeschleunigt auf AMD ROCm GPUs (speziell AMD Radeon RX 7800 XT / RDNA3 / `gfx1101`) mittels RapidOCR und ONNX Runtime ROCm Execution Provider.

---

## 1. Übersicht (Overview)

### 1.1 Zweck & Einsatzbereich

`rocm-ocr` stellt einen zustandslosen, domänenunabhängigen HTTP-Microservice für die Texterkennung (OCR) in Dokumenten (PDFs) und Rasterbildern (PNG, JPEG, TIFF, WebP) bereit.
Er dient als performantes OCR-Backend in Linux- und Kubernetes-Umgebungen und teilt sich bei Bedarf eine AMD-GPU mit anderen Diensten (wie z. B. Ollama).

### 1.2 Kernmerkmale

- **AMD ROCm GPU-Beschleunigung:** Text-Detektion (`ch_PP-OCRv4_det_infer.onnx`) und Richtungsklassifikation (`ch_ppocr_mobile_v2.0_cls_infer.onnx`) laufen auf der GPU über `ROCMExecutionProvider`.
- **RDNA3-Optimierung:** Automatische Kompatibilitätspatches für Consumer-Grafikkarten (`HSA_OVERRIDE_GFX_VERSION=11.0.0`, MIOpen Fast-Heuristics und Thread-Optimierung).
- **Hohe Genauigkeit:** Modellgestützte Erkennung mit lateinischem Zeichensatz (`latin_PP-OCRv3_rec_infer.onnx`) für vollständige Unterstützung deutscher Sonderzeichen (ä, ö, ü, ß, Groß-/Kleinschreibung).
- **Zustandslose REST-API:** Asynchrone FastAPI-Implementierung mit Pydantic V2-Datenmodellen.
- **Microservice-Observability:** Volle Unterstützung von `X-Correlation-ID` (`asgi-correlation-id`) zur Traceability über Servicegrenzen hinweg.
- **Stille Logs:** Automatische Filterung routinemäßiger `200 OK`-Healthchecks (`/healthz`, `/readyz`) und C-Level-Dämpfung interner MIOpen-Solver-Meldungen.
- **Fail-Fast & Transparenz:** Hardware- oder GPU-Fehler werden transparent mit HTTP 503 gemeldet; kein unbemerktes, teures Zurückfallen auf CPU.

---

## 2. Implementierung (Architecture & Implementation)

### 2.1 Modulübersicht

```text
src/rocm_ocr/
├── config.py       # Pydantic Settings & ROCm/MIOpen Umgebungskonfiguration
├── concurrency.py  # Dynamischer Concurrency-Limiter (Semaphore)
├── engine.py       # RapidOCR Lifecycle, ROCm Session-Patches & Inferenz
├── logging.py      # HealthCheckFilter & CorrelationAccessFormatter
├── models.py       # Pydantic Schemas (Request/Response API-Verträge)
├── parser.py       # PyMuPDF/Pillow Parsing & 144 DPI Skalierung
└── main.py         # FastAPI App, Lifespan, Middleware & Routen
```

### 2.2 OCR-Pipeline & RDNA3-Hybrid-Inferenz

Die Texterkennung gliedert sich in drei Stufen:

1. **Detektion (Text Detection):**
   - Modell: `ch_PP-OCRv4_det_infer.onnx`
   - Provider: `ROCMExecutionProvider` (AMD Radeon GPU)
   - Findet Textbox-Polygone im Gesamtbild in ca. **160 ms**.
2. **Klassifikation (Angle Classification):**
   - Modell: `ch_ppocr_mobile_v2.0_cls_infer.onnx`
   - Provider: `ROCMExecutionProvider` (GPU)
   - Korrigiert 0°/180°-Zeilenausrichtungen in ca. **100 ms**.
3. **Erkennung (Text Recognition):**
   - Modell: `latin_PP-OCRv3_rec_infer.onnx`
   - Provider: `CPUExecutionProvider` mit `intra_op_num_threads = 4`
   - **Hintergrund:** Ältere PP-OCRv3-Modelle mit `BiasSoftmax`-Operatoren lösen im aktuellen MIOpen-RDNA3-Treiber bei dynamischen Zeilenbreiten einen Kernel-Fehler aus. Das Modell wird isoliert auf 4 CPU-Threads ausgeführt. Dies vermeidet Thread-Contention auf Mehrkernsystemen und liefert Inferenzzeiten von ~1,0 s für 60 Textzeilen.

### 2.3 C-Level Stderr-Dämpfung

MIOpen schreibt Kernel-Lookup-Fehlermeldungen (`Failed to load kernel source: Conv_Winograd_...`) direkt über `std::cerr` (Dateideskriptor 2). In [`engine.py`](../src/rocm_ocr/engine.py) dämpft der Context-Manager `_silence_c_stderr()` mittels `os.dup2(devnull, 2)` diese C-Level-Ausgaben während der Inferenz, ohne Python-Exceptions zu unterdrücken.

---

## 3. Deployment

### 3.1 Docker-Container

Das Multi-Stage [`Dockerfile`](../Dockerfile) basiert auf Ubuntu Rolling mit ROCm 7.1 Runtime-Bibliotheken und vorkonfiguriertem Python 3.12:

```bash
# Docker Image bauen
docker build -t rocm-ocr:local .
```

#### Wichtige Container-Umgebungsvariablen

- `HSA_OVERRIDE_GFX_VERSION="11.0.0"`: Aktiviert ROCm auf Radeon RX 7800 XT / 7000er Serie.
- `MIOPEN_DEBUG_CONV_WINOGRAD="0"`: Unterdrückt nicht-existente GCN-Assembler-Kernel.
- `MIOPEN_FIND_MODE="2"`: Aktiviert Fast-Heuristic Search für dynamische Shapes.
- `OCR_CONCURRENCY="1"`: Maximale gleichzeitige Inferenzanfragen.
- `OCR_MODEL_DIR="/models"`: Pfad zu den ONNX-Modelldateien.

### 3.2 Kubernetes / k3s Deployment

Die Kustomize-Manifeste befinden sich in [`deploy/k8s/`](../deploy/k8s/):

- **Host Device Mounts:** Zugriff auf `/dev/kfd` und `/dev/dri` (Grafiktreiber und Kernel Fusion Driver).
- **Gruppen-IDs:** `supplementalGroups: [44, 109]` (`video`, `render`).
- **Privileged Mode:** Erforderlich für direkten GPU-Hardwarezugriff in k3s/containerd.
- **Probes:**
  - Liveness: `GET /healthz` (Port 8000, 10s Intervall)
  - Readiness: `GET /readyz` (Prüft Engine- und GPU-Bereitschaft)

#### Bereitstellung

```bash
# Image in k3s importieren (falls lokal gebaut)
docker save rocm-ocr:local | sudo k3s ctr images import -

# Kustomize anwenden
kubectl apply -k deploy/k8s/

# Rollout überwachen
kubectl rollout status deployment/rocm-ocr -n ocr
```

### 3.3 End-to-End-Verifikation

Ein in-cluster E2E-Testskript validiert Deployment, Latenz, Correlation-ID und deutsche Umlaute:

```bash
uv run python scripts/test_k8s_e2e.py --keep
```

---

## 4. API-Referenz (REST API Specification)

Basis-URL: `http://<host>:8000`

### 4.1 `GET /healthz`

Liveness-Probe für Kubernetes und Load Balancer.

- **Status Code:** `200 OK`
- **Response:**

  ```json
  {
    "status": "ok"
  }
  ```

### 4.2 `GET /readyz`

Readiness-Probe. Verifiziert die Funktionsfähigkeit der OCR-Engine und des GPU-Providers.

- **Status Codes:**
  - `200 OK`: Engine bereit.
  - `503 Service Unavailable`: GPU oder Engine nicht initialisiert.
- **Response (`200 OK`):**

  ```json
  {
    "status": "ready",
    "provider": "ROCMExecutionProvider"
  }
  ```

### 4.3 `POST /ocr/v1/setup`

Passt die maximale Inferenz-Parallelität (Concurrency) zur Laufzeit dynamisch an.

- **Request Body (`application/json`):**

  ```json
  {
    "concurrency": 2
  }
  ```

- **Response (`200 OK`):**

  ```json
  {
    "status": "updated",
    "concurrency": 2,
    "active_tasks": 0
  }
  ```

### 4.4 `POST /ocr/v1/predict`

Führt die Texterkennung auf einem hochgeladenen Dokument oder Bild durch.

- **Headers (Optional):**
  - `X-Correlation-ID`: Trace-ID. Wird übernommen und in der Response zurückgegeben. Fehlt der Header, generiert der Dienst automatisch eine UUID4.
- **Request (`multipart/form-data`):**
  - `file`: Binärdatei (`application/pdf`, `image/png`, `image/jpeg`, `image/tiff`, `image/webp`).
- **Response (`200 OK`):**

  ```json
  {
    "execution_provider": "ROCMExecutionProvider",
    "inference_time_ms": 1359.1,
    "pages": [
      {
        "page_number": 0,
        "width": 595.3,
        "height": 841.9,
        "boxes": [
          {
            "text": "Rechnung Nr. 2026-0042",
            "bbox": [54.0, 72.5, 230.1, 88.0],
            "confidence": 0.9852
          }
        ]
      }
    ]
  }
  ```

- **Fehlercodes:**
  - `400 Bad Request`: Nicht unterstütztes Dateiformat oder beschädigte Datei.
  - `422 Unprocessable Entity`: Fehlender Multipart-Parameter `file`.
  - `503 Service Unavailable`: Hardware-/Treiberfehler der GPU (inkl. detailliertem JSON-Error).

---

## 5. Lokale Entwicklung & Qualitätssicherung

```bash
# Abhängigkeiten installieren (strictly Python 3.12)
uv sync --all-groups

# Lokaler Dev-Server
uv run uvicorn rocm_ocr.main:app --host 0.0.0.0 --port 8000 --reload

# Tests ausführen
uv run pytest

# Pre-Commit-Prüfungen manuell diagnostizieren (bei Hook-Fehlern)
uv run prek run --all-files
```

---

## 6. CI/CD & Fast-Path Pipeline

Die GitHub Actions Pipeline (`.github/workflows/ci.yaml`) umfasst:

- **Test & SonarCloud Analysis:** Linting (Ruff), Typisierung (`ty check`), Pytest mit XML-Coverage-Report und automatische SonarCloud Quality Gate Überprüfung.
- **Fast-Path Pfad-Erkennung:** Änderungen, die ausschließlich Dokumentation oder Metadaten betreffen (`docs/**`, `*.md`, `renovate.json`, `sonar-project.properties`), überspringen den zeitaufwendigen Docker-Build.
- **Docker Build & Trivy Scan:** Container-relevante Änderungen bauen das Image und prüfen es auf ungefixte HIGH/CRITICAL-Schwachstellen.
- **Automerge:** PRs mit dem Label `automerge` werden nach erfolgreicher Quality-Gate-Prüfung automatisch gesquasht und gemerged.

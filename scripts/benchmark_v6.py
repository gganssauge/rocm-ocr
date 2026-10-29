#!/usr/bin/env python3
"""Benchmark comparing PP-OCRv3 vs PP-OCRv6 (Small & Medium) recognition models.

Evaluates:
- Latency breakdown (Det, Cls, Rec) across CPU and GPU
- Character and umlaut recognition accuracy on test fixtures
- Real-world case study on the Vodafone activation code PDF
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from rocm_ocr.config import settings
from rocm_ocr.engine import create_rapidocr
from rocm_ocr.parser import parse_document

MODELS_DIR = Path("models")
V6_DIR = MODELS_DIR / "ppocrv6"

MODELS_CONFIGS = [
    {
        "name": "PP-OCRv3 Latin (Baseline)",
        "rec_model_path": str(MODELS_DIR / "latin_PP-OCRv3_rec_infer.onnx"),
        "rec_keys_path": str(MODELS_DIR / "latin_dict.txt"),
    },
    {
        "name": "PP-OCRv6 Small",
        "rec_model_path": str(V6_DIR / "PP-OCRv6_small_rec.onnx"),
        "rec_keys_path": str(V6_DIR / "ppocrv6_dict.txt"),
    },
    {
        "name": "PP-OCRv6 Medium",
        "rec_model_path": str(V6_DIR / "PP-OCRv6_medium_rec.onnx"),
        "rec_keys_path": str(V6_DIR / "ppocrv6_dict.txt"),
    },
]

DOCUMENTS = [
    ("German Invoice", Path("tests/fixtures/sample_german_invoice.pdf")),
    ("Bundestagsdrucksache", Path("tests/fixtures/bundestag_drucksache_sample.pdf")),
    (
        "Vodafone Aktivierungscode",
        Path("2025-08-25 Vodafone Vodafone Aktivierungscode.pdf"),
    ),
    (
        "Commerzbank Anlegerprofil",
        Path("2026-02-04 Commerzbank Anlegerprofil.pdf"),
    ),
]


def init_engine(rec_model_path: str, rec_keys_path: str) -> RapidOCR:
    """Initialize RapidOCR with specified models and explicit hybrid providers."""
    settings.apply_rocm_env_overrides()

    det_model = str(MODELS_DIR / "ch_PP-OCRv4_det_infer.onnx")
    return create_rapidocr(
        det_model_path=det_model,
        rec_model_path=rec_model_path,
        rec_keys_path=rec_keys_path,
    )


def run_benchmark_on_page(engine: RapidOCR, img_np: np.ndarray) -> dict[str, Any]:
    """Run inference on a single page image and measure timings and output."""
    start = time.perf_counter()
    results, elapse_list = engine(img_np)
    total_time = (time.perf_counter() - start) * 1000

    boxes = []
    texts = []
    if results:
        for item in results:
            quad, text, score = item[0], str(item[1]), float(item[2])
            boxes.append((quad, text, score))
            texts.append(text)

    # elapse_list is [det_time, cls_time, rec_time] in seconds
    det_ms = (elapse_list[0] if elapse_list and len(elapse_list) > 0 else 0.0) * 1000
    cls_ms = (elapse_list[1] if elapse_list and len(elapse_list) > 1 else 0.0) * 1000
    rec_ms = (elapse_list[2] if elapse_list and len(elapse_list) > 2 else 0.0) * 1000

    return {
        "total_ms": total_time,
        "det_ms": det_ms,
        "cls_ms": cls_ms,
        "rec_ms": rec_ms,
        "box_count": len(boxes),
        "texts": texts,
        "full_text": "\n".join(texts),
    }


def _warmup_engines() -> dict[str, RapidOCR] | None:
    """Warm up configured OCR engines; return None if models are missing."""
    engines = {}
    for cfg in MODELS_CONFIGS:
        if not Path(cfg["rec_model_path"]).is_file():
            print(f"❌ Missing model file: {cfg['rec_model_path']}")
            return None
        print(f"📦 Loading {cfg['name']}...")
        engines[cfg["name"]] = init_engine(cfg["rec_model_path"], cfg["rec_keys_path"])
        dummy = np.ones((64, 64, 3), dtype=np.uint8) * 255
        engines[cfg["name"]](dummy)
    return engines


def _run_model_on_pages(
    engine: RapidOCR,
    rendered_pages: list[Any],
) -> list[dict[str, Any]]:
    """Run warm-up and measured benchmarks across all pages of a document."""
    page_runs = []
    for page in rendered_pages:
        img_np = np.array(page.image)
        _ = run_benchmark_on_page(engine, img_np)
        res = run_benchmark_on_page(engine, img_np)
        page_runs.append(res)
    return page_runs


def _benchmark_document_latency(
    doc_name: str,
    doc_path: Path,
    engines: dict[str, RapidOCR],
) -> dict[str, list[dict[str, Any]]]:
    """Benchmark a single document across all models and display results."""
    print(f"\n📄 Document: {doc_name} ({doc_path.name})")
    doc_bytes = doc_path.read_bytes()
    rendered_pages = parse_document(doc_bytes, "application/pdf")
    print(f"   Pages: {len(rendered_pages)}")

    doc_results: dict[str, list[dict[str, Any]]] = {}
    for cfg in MODELS_CONFIGS:
        m_name = cfg["name"]
        page_runs = _run_model_on_pages(engines[m_name], rendered_pages)
        doc_results[m_name] = page_runs

        avg_total = sum(r["total_ms"] for r in page_runs) / len(page_runs)
        avg_rec = sum(r["rec_ms"] for r in page_runs) / len(page_runs)
        avg_det = sum(r["det_ms"] for r in page_runs) / len(page_runs)
        total_boxes = sum(r["box_count"] for r in page_runs)

        print(
            f"   • {m_name:28s}: Total = {avg_total:7.1f} ms | "
            f"Rec = {avg_rec:7.1f} ms | Det = {avg_det:6.1f} ms | "
            f"Boxes = {total_boxes}"
        )
    return doc_results


def _run_latency_benchmarks(
    engines: dict[str, RapidOCR],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Run latency benchmarks for all available documents."""
    print("\n" + "=" * 80)
    print("⏱️  LATENCY BENCHMARKS")
    print("=" * 80)

    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for doc_name, doc_path in DOCUMENTS:
        if not doc_path.is_file():
            print(f"\n⚠️  Skipping missing doc: {doc_path}")
            continue
        all_results[doc_name] = _benchmark_document_latency(doc_name, doc_path, engines)
    return all_results


def _compare_recognition_quality(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
    umlauts: list[str],
) -> None:
    """Print umlaut and character count statistics per document and model."""
    print("\n" + "=" * 80)
    print("🔍 RECOGNITION QUALITY & UMLAUT COMPARISON")
    print("=" * 80)

    for doc_name, _doc_path in DOCUMENTS:
        if doc_name not in all_results:
            continue
        print(f"\n📄 {doc_name}:")
        for m_name in [c["name"] for c in MODELS_CONFIGS]:
            pages = all_results[doc_name][m_name]
            full_text = " ".join(" ".join(p["texts"]) for p in pages)
            umlaut_counts = {u: full_text.count(u) for u in umlauts if u in full_text}
            print(
                f"   • {m_name:28s}: {len(full_text):5d} chars, "
                f"Umlaute: {umlaut_counts}"
            )


def _compare_vodafone_deltas(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
    umlauts: list[str],
) -> None:
    """Analyze and print text differences on the Vodafone document."""
    v_key = "Vodafone Aktivierungscode"
    if v_key not in all_results:
        return

    print("\n" + "=" * 80)
    print("🔎 VODAFONE DOCUMENT DELTAS (v3 vs v6)")
    print("=" * 80)
    v3_p = all_results[v_key]["PP-OCRv3 Latin (Baseline)"]
    v6s_p = all_results[v_key]["PP-OCRv6 Small"]
    v6m_p = all_results[v_key]["PP-OCRv6 Medium"]

    v3_text = " ".join(" ".join(p["texts"]) for p in v3_p)
    v6_s_text = " ".join(" ".join(p["texts"]) for p in v6s_p)
    v6_m_text = " ".join(" ".join(p["texts"]) for p in v6m_p)

    # Check for umlaut discrepancies
    print("\nUmlaut Occurrences in Vodafone Document:")
    print(f"  v3 Baseline : { {u: v3_text.count(u) for u in umlauts} }")
    print(f"  v6 Small    : { {u: v6_s_text.count(u) for u in umlauts} }")
    print(f"  v6 Medium   : { {u: v6_m_text.count(u) for u in umlauts} }")

    # Sample lines where v6 differs from v3
    v3_boxes = [t for p in v3_p for t in p["texts"]]
    v6s_boxes = [t for p in v6s_p for t in p["texts"]]
    v6m_boxes = [t for p in v6m_p for t in p["texts"]]

    print("\nText differences on detected lines (first 10 differences):")
    diff_count = 0
    for i in range(min(len(v3_boxes), len(v6s_boxes), len(v6m_boxes))):
        t3, ts, tm = v3_boxes[i], v6s_boxes[i], v6m_boxes[i]
        if t3 != ts or t3 != tm:
            diff_count += 1
            if diff_count <= 10:
                print(f"  Line {i + 1:2d}:")
                print(f"    v3 : {t3}")
                print(f"    v6s: {ts}")
                print(f"    v6m: {tm}")


def main() -> None:
    print("=" * 80)
    print("🚀 PP-OCRv3 vs PP-OCRv6 (Small & Medium) Benchmark")
    print("=" * 80)

    engines = _warmup_engines()
    if engines is None:
        return

    all_results = _run_latency_benchmarks(engines)
    umlauts = ["ä", "ö", "ü", "Ä", "Ö", "Ü", "ß"]
    _compare_recognition_quality(all_results, umlauts)
    _compare_vodafone_deltas(all_results, umlauts)


if __name__ == "__main__":
    main()

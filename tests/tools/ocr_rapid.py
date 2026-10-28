#!/usr/bin/env python
"""Offline OCR stub for running the pipeline without PaddleOCR models.

The production pipeline always uses PP-OCRv5 through ``paddleocr`` (see
``scripts/ocr/ocr_paddle.py``). That requires ~85 MB of model weights
downloaded from the Paddle model zoo / Hugging Face on first run.

On sandboxes that can only reach PyPI/GitHub (no model CDN), this helper
substitutes ``rapidocr-onnxruntime`` — which *ships* its PP-OCRv4 ONNX
models inside the wheel — and writes the exact same
``{text, x1, y1, x2, y2, confidence}`` JSON shape the rest of the
pipeline consumes. Output goes to ``<work>/ocr/page_NN.ocr.json`` so
``build_deck.py`` can run unchanged.

This is a TEST/DEV helper only: recognition quality is PP-OCRv4, and the
per-word / per-character boxes PP-OCRv5 emits (``words``, ``char_boxes``)
are absent, so mixed-size and in-line colour heuristics degrade slightly.

Usage:
    python tests/tools/ocr_rapid.py --source-dir <slides/> --work-dir <run/>
    python tests/tools/ocr_rapid.py --source-dir <slides/> --work-dir <run/> --pages 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from image_sources import discover_page_numbers, find_page_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--pages", default=None,
                   help="Comma-separated page numbers (default: all).")
    p.add_argument("--min-conf", type=float, default=0.3)
    return p.parse_args()


def poly_to_bbox(poly) -> tuple[int, int, int, int]:
    xs = [int(pt[0]) for pt in poly]
    ys = [int(pt[1]) for pt in poly]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> int:
    args = parse_args()
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        sys.stderr.write(
            "ERROR: rapidocr-onnxruntime not installed. Run: "
            "pip install rapidocr-onnxruntime\n"
        )
        return 1

    source_dir = Path(args.source_dir)
    work = Path(args.work_dir)
    ocr_dir = work / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    if args.pages:
        pages = [str(p).strip().zfill(2)
                 for p in args.pages.split(",") if p.strip()]
    else:
        pages = discover_page_numbers(source_dir)

    engine = RapidOCR()
    for num in pages:
        try:
            img = find_page_image(source_dir, num)
        except FileNotFoundError:
            sys.stderr.write(f"SKIP page {num}: no image found\n")
            continue
        result, _ = engine(str(img))
        items: list[dict] = []
        for poly, text, score in (result or []):
            if float(score) < args.min_conf or not text or not text.strip():
                continue
            x1, y1, x2, y2 = poly_to_bbox(poly)
            items.append({
                "text": text,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": float(score),
            })
        out = ocr_dir / f"page_{num}.ocr.json"
        out.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"ok page {num}: {len(items)} items -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

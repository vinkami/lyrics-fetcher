"""Tesseract OCR (CPU fallback for OCR).

Kept for well-lit, flat, printed pages; inferior to the vision LLM on real
phone-photo booklets (uneven lighting). Use VLMOcr for production.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytesseract

from .base import BaseOCR


class TesseractOcr(BaseOCR):
    name = "ocr-tesseract"

    def __init__(self, lang: str = "jpn+eng", max_side: int = 2400):
        self.lang = lang
        self.max_side = max_side

    def ocr(self, image: Path) -> str:
        im = Image.open(image).convert("L")
        w, h = im.size
        scale = min(1.0, self.max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return pytesseract.image_to_string(im, lang=self.lang)
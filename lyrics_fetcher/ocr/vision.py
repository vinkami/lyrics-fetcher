"""Vision LLM OCR via a local llama-server (Qwen3.5-9B on RX 9060 XT).

PRODUCTION (2026-08-28): Qwen3.5-9B (Q4_K_M) served on 127.0.0.1:8081 (my own
start-vision; user's ~/AI/start untouched). Uses ~7.6GB VRAM, leaving ~9GB so
whisper-medium can run simultaneously. Gemma-4-12B failed (mojibake); 27B was
worse and used 16.4GB.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import requests
from PIL import Image

from ..config import settings
from ..models import Lyrics
from .base import BaseOCR

# default Qwen vision server (override via constructor or config)
DEFAULT_API = settings.vision_api
DEFAULT_MODEL = settings.vision_model

VLM_PROMPT = (
    "This is a photo of an album lyrics booklet page (Japanese). "
    "Extract ALL the text exactly as printed, preserving line breaks. "
    "Only output the transcribed lyrics text, one line per lyric line. "
    "Do not add commentary, do not translate, do not include furigana readings separately. "
    "If the page contains song title/artist headers, output them as [title] ... style lines."
)

# Post-OCR cleanup prompt: fixes minor VLM slips (dropped particles/morphemes,
# wrong kanji) WITHOUT re-reading the image. Deliberately asks to change as
# little as possible — we trust the transcription, only correct obvious slips.
CLEANUP_PROMPT = (
    "Below is an OCR transcription of Japanese song lyrics from a booklet. "
    "Fix ONLY clear transcription errors: a dropped particle/morpheme (e.g. "
    "missing の/一), a wrong kanji that's obviously a misread, or an awkward "
    "morph that breaks grammar. Keep every line's meaning and line breaks; do "
    "not reformat, do not add/remove lines, do not translate, do not editorialize. "
    "If a line looks fine, output it unchanged. Output only the corrected lines, "
    "one per line, nothing else.\n\n"
    "LYRICS:\n{text}"
)

MAX_SIDE = 1568


class VLMOcr(BaseOCR):
    name = "ocr-vlm"

    def __init__(self, api: str | None = None, model: str | None = None,
                 timeout: int = 600, cache=None, clean: bool = True):
        self.api = api or settings.vision_api
        self.model = model or settings.vision_model
        self.timeout = timeout
        self.cache = cache
        self.clean = clean

    def _chat(self, prompt: str, image: Path | None = None, max_tokens: int = 2048) -> str:
        """Call the llama-server vision/text endpoint; returns assistant text."""
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": self._encode(image)}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        r = requests.post(self.api, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _encode(image: Path) -> str:
        im = Image.open(image).convert("RGB")
        w, h = im.size
        scale = min(1.0, MAX_SIDE / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def cleanup(self, text: str) -> str:
        """Post-OCR cleanup pass: fix minor slips (dropped particles/kanji)."""
        return self._chat(CLEANUP_PROMPT.format(text=text), max_tokens=2048).strip()

    def ocr(self, image: Path) -> str:
        # cache hit by absolute image path
        if self.cache:
            hit = self.cache.get_ocr(image)
            if hit is not None:
                return hit
        text = self._chat(VLM_PROMPT, image=image)
        if self.clean:
            try:
                text = self.cleanup(text)
            except Exception:
                pass  # cleanup is best-effort; keep raw transcription
        if self.cache:
            self.cache.put_ocr(image, text)
        return text

    def fetch(self, image: Path, title: str = "", artist: str = "") -> Lyrics:
        return super().fetch(image, title, artist)
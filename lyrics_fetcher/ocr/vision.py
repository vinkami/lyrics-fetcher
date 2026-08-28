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

from ..models import Lyrics
from .base import BaseOCR

# default Qwen vision server (override via constructor)
DEFAULT_API = "http://127.0.0.1:8081/v1/chat/completions"
DEFAULT_MODEL = "qwen3.5-9b"

VLM_PROMPT = (
    "This is a photo of an album lyrics booklet page (Japanese). "
    "Extract ALL the text exactly as printed, preserving line breaks. "
    "Only output the transcribed lyrics text, one line per lyric line. "
    "Do not add commentary, do not translate, do not include furigana readings separately. "
    "If the page contains song title/artist headers, output them as [title] ... style lines."
)

MAX_SIDE = 1568


class VLMOcr(BaseOCR):
    name = "ocr-vlm"

    def __init__(self, api: str = DEFAULT_API, model: str = DEFAULT_MODEL, timeout: int = 600):
        self.api = api
        self.model = model
        self.timeout = timeout

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

    def ocr(self, image: Path) -> str:
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": self._encode(image)}},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        r = requests.post(self.api, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def fetch(self, image: Path, title: str = "", artist: str = "") -> Lyrics:
        return super().fetch(image, title, artist)
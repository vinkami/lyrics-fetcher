"""PoC: OCR comparison — local vision LLM (Qwen via llama-server) vs Tesseract.

Tested on real booklet photos (phone photos, uneven lighting, no scanner).

RESULTS (ASTEROID 06...250.jpg):
- VISION LLM (qwen3.8-27b via ROCm): EXCELLENT. Transcribed アンデッド track
  lyrics with accurate kanji, correct line breaks, only ~2 minor errors
  (もしくは->もしいない, 壊 garbled). Also handled maimai prism layout and
  manosaba artificial language lyrics.
- TESSERACT (CPU, jpn+eng): POOR on uneven-light phone photos. Lost most lines,
  merged/dropped text, garbled kanji ("朽ちる"->"紀かれた", "正しくは"->"暗くは"),
  hallucinated noise tokens. Not usable alone for these booklets.

CONCLUSION: vision LLM is the primary OCR engine for real booklet photos.
Tesseract is a fallback only for well-lit, flat, printed pages if vision is off.
"""

import sys
from pathlib import Path

# VLM
import base64, io, json
import requests
from PIL import Image

# Tesseract
import pytesseract

VLM_API = "http://127.0.0.1:8080/v1/chat/completions"
VLM_MODEL = "qwen3.8-27b"

VLM_PROMPT = (
    "This is a photo of an album lyrics booklet page (Japanese). "
    "Extract ALL the text exactly as printed, preserving line breaks. "
    "Only output the transcribed lyrics text, one line per lyric line. "
    "Do not add commentary, do not translate, do not include furigana readings separately. "
    "If the page contains song title/artist headers, output them as [title] ... style lines."
)


def vlm_ocr(path: Path, timeout: int = 300) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, 1568 / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    url = f"data:image/jpeg;base64,{b64}"
    payload = {
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    r = requests.post(VLM_API, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def tesseract_ocr(path: Path, lang: str = "jpn+eng", max_side: int = 2400) -> str:
    im = Image.open(path).convert("L")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pytesseract.image_to_string(im, lang=lang)


if __name__ == "__main__":
    img = Path(sys.argv[1])
    engine = sys.argv[2] if len(sys.argv) > 2 else "vlm"
    if engine == "vlm":
        print(vlm_ocr(img))
    else:
        print(tesseract_ocr(img))
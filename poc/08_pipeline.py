"""Capstone PoC: Full pipeline for one song.

Photo of booklet page → Vision LLM OCR (Qwen3.5-9B on RX 9060 XT)
                        → lyric lines
                        → align with whisper.cpp timestamps (GPU, coexists w/ Qwen)
                        → .lrc + .html output

Demonstrates the complete Part 1 (OCR fetcher) + Part 2 (alignment) integration
on a real song: ASTEROID track 1 アンデッド, using the user's booklet photo.

Usage:
    uv run python poc/08_pipeline.py
"""
import base64
import io
import re
import json
import subprocess
import sys
from pathlib import Path

import requests
from PIL import Image

# Qwen vision server (my own start-vision, port 8081; runs ~7.6GB, whisper shares card)
VLM_API = "http://127.0.0.1:8081/v1/chat/completions"
VLM_MODEL = "qwen3.5-9b"

# whisper.cpp (Vulkan on RX 9060 XT, medium = best for synthetic vocals)
WHISPER = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-medium.bin"

MUSIC = Path("/mnt/fnos/storage/Music") / "光収容の倉庫 ASTEROID"
BOOKLET_IMG = MUSIC / "booklet" / "20260828_060250.jpg"
AUDIO = MUSIC / "01 アンデッド.flac"
OUT_DIR = Path(__file__).parent / "out"

OCR_PROMPT = (
    "This is a photo of an album lyrics booklet page (Japanese). "
    "Extract ALL the lyrics exactly as printed, one line per lyric line. "
    "Output ONLY the song lyrics — do NOT include the title, artist, or any header "
    "lines, do NOT translate, do NOT add commentary. If furigana rubi is printed, "
    "keep just the kanji reading the lyrics normally."
)


def ocr(image: Path) -> str:
    """OCR a booklet page via the local Qwen vision model."""
    im = Image.open(image).convert("RGB")
    w, h = im.size
    s = min(1.0, 1568 / max(w, h))
    if s < 1.0:
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    r = requests.post(VLM_API, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def run_whisper(audio: Path, lang="ja", max_len=40) -> list[dict]:
    out = Path("/tmp/_lf_pipeline")
    cmd = [str(WHISPER), "-m", str(WHISPER_MODEL), "-l", lang,
           "-f", str(audio), "-ml", str(max_len), "-oj", "-of", str(out), "--no-prints"]
    subprocess.run(cmd, check=True, capture_output=True)
    data = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    segs = []
    for s in data.get("transcription", []):
        off = s["offsets"]
        segs.append({"from": off["from"], "to": off["to"], "text": s["text"].strip()})
    return segs


def clean(s: str) -> str:
    return re.sub(r"[\s「」『』（）()〈〉【】、。,.!！?？\-]", "", s)


def align_lines(known: list[str], segs: list[dict]) -> list[tuple[float, str]]:
    """Assign whisper segment start times to known lyric lines via fuzzy match."""
    try:
        from thefuzz import fuzz
    except ImportError:
        fuzz = None
    if fuzz is None:
        total = segs[-1]["to"] / 1000 if segs else 0
        n = len(known)
        return [(i * total / max(n, 1), l) for i, l in enumerate(known)]
    times = {}
    for ki, line in enumerate(known):
        cl = clean(line)
        best, best_score = None, 0
        for si, seg in enumerate(segs):
            score = fuzz.ratio(cl, clean(seg["text"]))
            if score > best_score:
                best, best_score = si, score
        times[ki] = segs[best]["from"] / 1000 if best is not None else 0
    return [(times[ki], l) for ki, l in enumerate(known)]


def write_lrc(path: Path, title, artist, album, timed):
    L = [f"[ti:{title}]", f"[ar:{artist}]", f"[al:{album}]", "[by:lyrics-fetcher]", ""]
    for t, text in timed:
        m, s = divmod(t, 60)
        L.append(f"[{int(m):02d}:{s:05.2f}]{text}")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("=== Capstone: OCR -> align -> LRC (ASTEROID アンデッド) ===\n")

    print("[1/3] OCR booklet page via Qwen3.5-9B...")
    raw = ocr(BOOKLET_IMG)
    # filter out any [title] remnants already excluded by prompt; drop empties
    known = [l for l in (x.strip() for x in raw.splitlines()) if l]
    print(f"      OCR'd {len(known)} lyric lines (sample):")
    for l in known[:4]:
        print(f"        {l}")

    print("\n[2/3] Whisper alignment (GPU, coexists with Qwen)...")
    segs = run_whisper(AUDIO)
    print(f"      Whisper produced {len(segs)} segments")

    print("\n[3/3] Matching + writing LRC...")
    timed = align_lines(known, segs)
    out = OUT_DIR / "ASTEROID_01_アンデッド.lrc"
    write_lrc(out, "アンデッド", "光収容の倉庫", "ASTEROID", timed)
    print(f"      Wrote {out} ({len(timed)} lines)\n")

    print("Final LRC:")
    for t, l in timed:
        m, s = divmod(t, 60)
        print(f"  [{int(m):02d}:{s:05.2f}] {l}")


if __name__ == "__main__":
    main()
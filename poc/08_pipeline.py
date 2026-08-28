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


def align_lines(known: list[str], segs: list[dict], min_score: float = 0.0) -> list[tuple[float, str]]:
    """Monotonic forced alignment (DTW-style).

    Maps each known lyric line to a whisper segment such that the line indices
    are NON-DECREASING (line n cannot map to an earlier segment than line n-1).
    Maximizes total fuzzy-match similarity via dynamic programming with
    backtracking. This fixes the repeated-chorus collision the greedy best-match
    version had: identical chorus lines are now kept in correct temporal order.

    Returns list of (start_seconds, line_text) in original line order.
    """
    from thefuzz import fuzz

    n_lines = len(known)
    n_segs = len(segs)
    if n_lines == 0 or n_segs == 0:
        return [(0.0, l) for l in known]

    # similarity[i][j] = match quality of known line i against segment j
    sim = [[0.0] * n_segs for _ in range(n_lines)]
    for i, line in enumerate(known):
        cl = clean(line)
        for j, seg in enumerate(segs):
            sim[i][j] = fuzz.ratio(cl, clean(seg["text"]))

    # DP: dp[i][j] = best total score for first (i+1) lines, line i->seg j.
    # Monotonic constraint: line i-1 maps to seg k with k <= j (non-decreasing).
    # Transition: dp[i][j] = sim[i][j] + max_{k<=j} dp[i-1][k]. We track the argmax
    # of the prefix as `running` to make each row O(n_segs).
    NEG = float("-inf")
    dp = [[NEG] * n_segs for _ in range(n_lines)]
    back = [[-1] * n_segs for _ in range(n_lines)]  # seg index line i-1 used

    for j in range(n_segs):
        dp[0][j] = sim[0][j]
    for i in range(1, n_lines):
        running = NEG
        argk = None
        for j in range(n_segs):
            if dp[i-1][j] > running:  # prefix max of previous row up to j
                running = dp[i-1][j]
                argk = j
            if argk is not None:
                dp[i][j] = running + sim[i][j]
                back[i][j] = argk

    # backtrack: find best end
    best_j = max(range(n_segs), key=lambda j: dp[n_lines-1][j])
    assign = [0] * n_lines
    j = best_j
    for i in range(n_lines - 1, 0, -1):
        assign[i] = j
        j = back[i][j]
    assign[0] = j

    timed = []
    for i, line in enumerate(known):
        seg = segs[assign[i]]
        timed.append((seg["from"] / 1000.0, line))
    return timed


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
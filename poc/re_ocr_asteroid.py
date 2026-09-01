"""poc/re_ocr_asteroid.py — re-OCR ASTEROID booklet photos, realign on vocal stems.

The user re-photographed the booklet (one photo per song, new 20260902_*.jpg).
This:
  1. OCRs each new photo via the vision server (:8081), splitting per-song and
     normalizing the label to the canonical on-disc track title.
  2. Takes the corrected lyrics for each track.
  3. Re-aligns those lyrics against the demucs vocal stem (in _sep_out/*_vocals.wav)
     with the existing WhisperCppAligner.
  4. Writes the re-aligned .lrc to a fresh output dir for review.

Usage: uv run python poc/re_ocr_asteroid.py
"""
from __future__ import annotations

import re
from pathlib import Path

from lyrics_fetcher.aligner.whisper_cpp import WhisperCppAligner
from lyrics_fetcher.models import Lyrics, LyricLine, SongMeta
from lyrics_fetcher.ocr.vision import VLMOcr

ALBUM = Path("/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID")
BOOKLET = ALBUM / "booklet"
# the newly-taken photos (20260902) — one per song
PHOTOS = sorted(BOOKLET.glob("20260902_*.jpg"))
# vocal stems produced by the earlier demucs pass
STEM_DIR = Path("/home/vinkami/Code/lyrics-fetcher/_sep_out")
# fresh output so we don't clobber the user's files while reviewing
OUTDIR = Path("/home/vinkami/Code/lyrics-fetcher/_lrc_re-ocr")
OUTDIR.mkdir(parents=True, exist_ok=True)


def clean_title(s: str) -> str:
    return re.sub(r"[\s「」『』]", "", s)


def main() -> None:
    if not PHOTOS:
        print("No 20260902 photos found in", BOOKLET)
        return

    ocr = VLMOcr()  # points at the default :8081 vision server
    aligner = WhisperCppAligner()

    # audio files (sorted 01..05); stems are named "{audio.stem}_vocals.wav"
    audios = sorted(ALBUM.glob("*.flac"))
    # map canonical metadata-title -> track-number (for positional fallback)
    num_by_title = {}
    for a in audios:
        meta = SongMeta.from_path(a)
        t = (meta.title or a.stem).strip()
        num = re.match(r"^(\d+)", a.stem)
        num_by_title[clean_title(t)] = (num.group(1) if num else None, t)

    print(f"Found {len(PHOTOS)} new photos, {len(num_by_title)} tracks\n")

    for idx, img in enumerate(PHOTOS):
        print(f"=== OCR {img.name} ===")
        try:
            songs = ocr.extract_songs(img, known_titles=[t for _, t in num_by_title.values()])
        except Exception as e:
            print(f"  OCR FAILED: {e}")
            continue
        if not songs:
            print("  (no song blocks extracted)")
            continue
        label, text = next(iter(songs.items()))
        num, canon = num_by_title.get(clean_title(label), (None, None))
        if canon is None:
            # positional fallback (photo idx -> track idx)
            num, canon = list(num_by_title.values())[idx]
            print(f"  ! '{label}' not canonical; positional fallback -> '{canon}'")
        if num is None:
            print(f"  ! no track number for '{canon}'")
            continue
        stem = STEM_DIR / f"{num} {canon}_vocals.wav"
        if not stem.exists():
            print(f"  ! stem not found: {stem.name}")
            continue
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        print(f"  aligning {len(lines)} lines on {stem.name} ...")
        ly = Lyrics(source="ocr-vlm", title=canon, lines=[LyricLine(l) for l in lines])
        timed = aligner.align(stem, ly)
        out = OUTDIR / f"{num} {canon}.lrc"
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"[ti:{canon}]\n[ar:光収容]\n[al:ASTEROID]\n")
            f.write(f"[by:re-ocr vocal-sep]\n\n")
            for tl in timed:
                s = tl.start
                m = int(s // 60)
                f.write(f"[{m:02d}:{s - m * 60:05.2f}]{tl.text}\n")
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
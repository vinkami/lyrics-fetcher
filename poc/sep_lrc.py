"""poc/sep_lrc.py — produce vocal-separated .lrc files for the ASTEROID album.

For each song: demucs-separate the vocal stem, align the known lyrics (extracted
from the album's existing .lrc files) with the existing WhisperCppAligner, and
write a .lrc into a PERSISTENT output dir for comparison against the originals.

Usage: uv run python poc/sep_lrc.py
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import demucs.api
import soundfile as sf

from lyrics_fetcher.aligner.whisper_cpp import WhisperCppAligner
from lyrics_fetcher.models import Lyrics, LyricLine

ALBUM = Path("/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID")
# persistent output (NOT /tmp — wiped on reboot)
OUTDIR = Path("/home/vinkami/Code/lyrics-fetcher/_sep_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEPARATOR = None


def get_separator():
    global SEPARATOR
    if SEPARATOR is None:
        SEPARATOR = demucs.api.Separator(
            model="htdemucs",
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        )
    return SEPARATOR


def split_vocals(audio: Path) -> Path:
    sep = get_separator()
    _, separated = sep.separate_audio_file(str(audio))
    vocals = separated["vocals"]
    out = OUTDIR / f"{audio.stem}_vocals.wav"
    sf.write(str(out), vocals.cpu().numpy().T, sep.samplerate)
    return out


def extract_lyrics_from_lrc(lrc: Path) -> list[str]:
    """Pull lyric lines from an existing .lrc (strip metadata + timestamps)."""
    out = []
    for ln in lrc.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith(("[ti:", "[ar:", "[al:", "[by:")):
            continue
        text = re.sub(r"^(\[[0-9:.\[\]]+\])+", "", ln).strip()
        if text:
            out.append(text)
    return out


def write_lrc(audio: Path, times: list[float], lines: list[str], out: Path) -> None:
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"[ti:{audio.stem}]\n")
        f.write("[ar:光収容]\n[al:ASTEROID]\n[by:lyrics-fetcher (vocal-sep)]\n\n")
        for i, text in enumerate(lines):
            s = times[i]
            m = int(s // 60)
            f.write(f"[{m:02d}:{s - m * 60:05.2f}]{text}\n")


def process(audio: Path) -> None:
    lrc_src = ALBUM / f"{audio.stem}.lrc"
    if not lrc_src.exists():
        print(f"[{audio.stem}] no source .lrc, skipping", flush=True)
        return
    lines = extract_lyrics_from_lrc(lrc_src)
    if not lines:
        print(f"[{audio.stem}] empty lyrics, skipping", flush=True)
        return

    t0 = time.time()
    print(f"[{audio.stem}] separating ({len(lines)} lines)...", flush=True)
    target = split_vocals(audio)
    print(f"  separated in {time.time()-t0:.1f}s", flush=True)

    aligner = WhisperCppAligner()
    ly = Lyrics(source="spike", lines=[LyricLine(text=t) for t in lines])
    timed = aligner.align(target, ly)
    times = [l.start for l in timed]
    out = OUTDIR / f"{audio.stem}_sep.lrc"
    write_lrc(audio, times, lines, out)
    print(f"[{audio.stem}] wrote {out.name}", flush=True)


def main() -> None:
    for audio in sorted(ALBUM.glob("*.flac")):
        process(audio)


if __name__ == "__main__":
    main()
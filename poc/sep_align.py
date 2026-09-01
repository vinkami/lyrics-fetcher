"""poc/sep_align.py — throwaway spike: vocal separation before whisper alignment.

Computes anchor stats for a track aligned from (a) the raw audio and (b) a
separated vocal stem, using the existing WhisperCppAligner. Prints a comparison
so we can judge whether separation helps hallucinating/drifting songs.

Usage:
    uv run python poc/sep_align.py "<audio.flac>" [--engine demucs|audio-separator]
        [--lyrics lyrics.txt] [--outdir /tmp/sep_out]
        [--time-base /0:00.0 ...]   # optional, not used yet

Purposely ragged / not test-covered. Discarded after the spike.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

from lyrics_fetcher.aligner.whisper_cpp import WhisperCppAligner
from lyrics_fetcher.models import Lyrics, LyricLine

MIN_SCORE = 58.0  # anchor_min_score config default
RAW_CACHE = Path("/tmp/sep_raw_cache.json")


def load_lines(lyrics_txt: Path) -> list[str]:
    """Load lyric lines from a plain text file (non-empty lines)."""
    return [ln.strip() for ln in lyrics_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def split_vocals(audio: Path, out_dir: Path, engine: str) -> Path:
    """Separate the vocal stem. Returns path to a vocal .wav."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if engine == "demucs":
        # run in the same venv so torch/rocm is used; model cached on first call
        import demucs.api
        separator = demucs.api.Separator(model="htdemucs", device="cuda" if __import__("torch").cuda.is_available() else "cpu")
        _, separated = separator.separate_audio_file(str(audio))
        vocals = separated["vocals"]
        import soundfile as sf
        target = out_dir / "demucs_vocals.wav"
        sf.write(str(target), vocals.cpu().numpy().T, separator.samplerate)
        result = target
    else:
        from audio_separator.separator import Separator
        sep = Separator(
            output_dir=str(out_dir),
            model_file_dir=str(out_dir / "models"),
            output_single_stem="vocals",
            sample_rate=44100,
        )
        sep.load_model("MDX-Net Model: UVR-MDX-NET Inst HQ 5")
        sep.initialize_model()
        outputs = sep.separate(str(audio))
        # with output_single_stem we expect one vocals file
        result = Path(outputs[0]) if outputs else next(
            (out_dir / f for f in out_dir.glob("*Vocals*") if "vocals" in f.name.lower()))
        print(f"[{engine}] outputs={[str(o) for o in outputs]}")
    print(f"[{engine}] separated in {time.time()-t0:.1f}s -> {result}")
    return result


def anchor_stats(aligner: WhisperCppAligner, audio: Path, lines: list[str]) -> dict:
    segs = aligner._all_segments(audio)
    anchors = aligner._anchor_align(lines, segs, min_score=MIN_SCORE)
    anchored = [i for i, a in enumerate(anchors) if a is not None]
    # build per-line times to inspect even-spread fallback
    from lyrics_fetcher.models import Lyrics
    ly = Lyrics(source="poc-spike", lines=[LyricLine(text=t) for t in lines])
    timed = aligner.align(audio, ly)
    return {
        "total": len(lines),
        "n_segments": len(segs),
        "anchored": len(anchored),
        "anchor_idx": anchored,
        "even_spread": (len(anchored) == 0),
        "first_anchor_time": (timed[anchored[0]].start if anchored else None),
        "times": [round(t.start, 2) for t in timed],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--engine", choices=["demucs", "audio-separator"], default="demucs")
    ap.add_argument("--lyrics", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/sep_out"))
    args = ap.parse_args()

    lines = load_lines(args.lyrics) if args.lyrics else None

    aligner = WhisperCppAligner()
    print(f"=== {args.audio} (engine={args.engine}) ===")
    print(f"device env torch rocm={__import__('torch').cuda.is_available()}")

    if lines:
        # RAW result: compute once, reuse across engines via a JSON cache
        cache = {}
        if RAW_CACHE.exists():
            cache = json.loads(RAW_CACHE.read_text(encoding="utf-8"))
        key = str(args.audio)
        if key not in cache:
            print("\n--- RAW audio (computing) ---")
            raw = anchor_stats(aligner, args.audio, lines)
            cache[key] = raw
            RAW_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        raw = cache[key]
        print(f"--- RAW (cached) ---  anchored {raw['anchored']}/{raw['total']}  "
              f"n_seg={raw['n_segments']}  even_spread={raw['even_spread']}  "
              f"first_anchor={raw['first_anchor_time']}")

        print("\n--- Separated vocal stem ---")
        stem = split_vocals(args.audio, args.outdir, args.engine)
        sep = anchor_stats(aligner, stem, lines)
        print(f"anchored {sep['anchored']}/{sep['total']}  n_seg={sep['n_segments']}  "
              f"even_spread={sep['even_spread']}  first_anchor={sep['first_anchor_time']}")

        print("\nComparator: raw vocab times vs sep times")
        for i, (rv, sv) in enumerate(zip(raw["times"], sep["times"])):
            marker = " <-- EVEN-SPREAD" if raw["even_spread"] and i == 1 else ""
            print(f"  L{i:02d}  raw={rv:7.2f}  sep={sv:7.2f}")
    else:
        # no lyrics: just separate + report segments found by whisper on each
        print("(no --lyrics provided; separating and transcribing segments)")
        stem = split_vocals(args.audio, args.outdir, args.engine)
        for label, p in (("raw", args.audio), ("stem", stem)):
            segs = aligner._all_segments(p)
            print(f"[{label}] {len(segs)} segments")
            for s in segs[:10]:
                print(f"   {s['from']/1000:.2f}-{s['to']/1000:.2f}: {s['text'][:60]}")


if __name__ == "__main__":
    main()
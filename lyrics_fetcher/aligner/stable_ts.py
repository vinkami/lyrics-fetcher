"""stable-ts aligner — forced alignment via stable-whisper (opt-in).

whisper.cpp's anchor DP still has three known failure modes on ASTEROID
(HANDOFF §9): intro hallucination -> 0 anchors -> even-spread, off-by-one
cascades when it merges two lyric lines into one segment, and ~6-kana drift
after a resync. stable-ts (``m.align(audio, known_text)``) forces the KNOWN
lyrics through whisper's own cross-attention and emits word-level timestamps,
so line starts come from the audio directly — no anchors, no interpolation.

Phase 0 gate (2026-09-04, poc/stablets_align.py + poc/out/stablets_results.json):
validated on all 5 ASTEROID songs — 0 monotonic violations, 告げよ intro
anchored at 27.4/30.5/33.8, アンデッド intro anchored and the first repeated
chorus line kept at its first occurrence (43.9s, not the later 152s match).
Runs on our torch 2.11.0+rocm7.2: medium model ~5s load, 6-36s per song,
~3.2 GiB peak VRAM (coexists with whisper.cpp models and the vision server).

stabilize-whisper is a DEV dependency (like demucs — Linux-only env), so
``stable_whisper`` is imported LAZILY inside ``_load``: this module (and all
tests) must work without it installed. Any failure — missing lib, OOM, model
download error — warns and falls back to whisper.cpp so a run never breaks.

HARDWARE NOTE: keep ``device`` on the RX 9060 XT (torch index 0 under ROCm).
The eGPU (RX 6600 XT, hosts the vision server) is torch device index 2 and
HANGS on first compute — do not target it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from ..config import settings
from ..models import Lyrics
from .base import BaseAligner, TimedLine

# sentinel: "no fallback given" -> build WhisperCppAligner() lazily on first need
# (never at construction time, so the CLI can offer --aligner stable-ts cheaply)
_UNSET = "sentinel"


def _norm(s: str) -> str:
    """Normalize a lyric line / whisper word for char-count matching: stable-ts
    word segmentation and the source lyrics differ in whitespace, so we compare
    by stripped char counts, never exact text."""
    return re.sub(r"\s+", "", s or "")


class StableTSAligner(BaseAligner):
    name = "stable-ts"

    def __init__(self, model: str | None = None, lang: str | None = None,
                 device: str | None = None,
                 fallback: BaseAligner | None = _UNSET):
        # resolve against live config at construction time (so --config works)
        self.model = model or settings.stable_ts_model
        self.lang = lang or settings.stable_ts_lang
        self.device = device or settings.stable_ts_device
        self.fallback = None if fallback is _UNSET or fallback is None else fallback
        self._model = None

    # ---- model loading (lazy: stable_whisper is an optional dev dep) ----
    def _load(self):
        if self._model is None:
            import stable_whisper  # lazy: not installed in CI / default deps
            self._model = stable_whisper.load_model(self.model, device=self.device)
        return self._model

    def _fallback_aligner(self) -> BaseAligner:
        """The whisper.cpp fallback, constructed only when actually needed."""
        if self.fallback is None:
            from .whisper_cpp import WhisperCppAligner
            self.fallback = WhisperCppAligner()
        return self.fallback

    # ---- word timestamps -> per-line start times ----
    @staticmethod
    def _line_times(result, lines: list[str]) -> list[float]:
        """Start time for each lyric line, from stable-ts word timestamps.

        Flatten every segment's words (start, normalized-text), then walk the
        word stream once per line, consuming words until their concatenated
        char count reaches the line's char count. The line starts at its first
        consumed word. Matching by char count after whitespace-normalization
        is what makes this robust to whisper splitting words differently (or
        gluing two lyric lines into one word) than the source text. Greedy
        consumption also keeps repeated identical lines in order: line N+1
        starts after line N's words were consumed, so the second chorus lands
        on the second occurrence. (Ported from poc/stablets_align.py.)
        """
        words: list[tuple[float, str]] = []
        for seg in result.segments:
            for w in (seg.words or []):
                words.append((float(w.start), _norm(w.word)))

        times: list[float] = []
        wi = 0
        for line in lines:
            target = _norm(line)
            line_start = None
            acc = ""
            while wi < len(words) and len(acc) < len(target):
                ws, wt = words[wi]
                if line_start is None and wt:
                    line_start = ws
                acc += wt
                wi += 1
                if len(acc) >= len(target):
                    break
            if line_start is None:
                # word stream exhausted (or empty line): hold the previous time
                # rather than inventing one — the clamp below keeps it sane.
                line_start = times[-1] if times else 0.0
            times.append(line_start)
        return times

    # ---- public API ----
    def align(self, audio: Path, lyrics: Lyrics) -> list[TimedLine]:
        """Align known lyrics with stable-ts; ANY failure falls back to
        whisper.cpp so opting in can never break a run (mirrors the
        --separation graceful-degradation contract)."""
        known = [l.text for l in lyrics.lines]
        if not known:
            print("stable-ts: no lyric lines to align", file=sys.stderr)
            return self._fallback_aligner().align(audio, lyrics)
        try:
            model = self._load()
            # regroup="p" = punctuation-level regrouping (the validated Phase 0 setting)
            result = model.align(str(audio), "\n".join(known),
                                 language=self.lang, regroup="p", verbose=False)
            times = self._line_times(result, known)
        except Exception as e:
            fb = self._fallback_aligner()
            print(f"stable-ts alignment failed ({e.__class__.__name__}: {e}); "
                  f"falling back to {fb.name}", file=sys.stderr)
            return fb.align(audio, lyrics)

        timed: list[TimedLine] = []
        prev = 0.0
        for text, t in zip(known, times):
            # cheap safety: LRC players assume non-decreasing starts, and a
            # word-boundary hiccup could hand line N+1 a start slightly below
            # line N's — clamp instead of shipping a broken .lrc.
            t = max(t, prev)
            timed.append(TimedLine(text=text, start=t))
            prev = t
        return timed

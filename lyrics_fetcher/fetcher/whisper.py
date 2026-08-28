"""Whisper-as-fetcher — AI recognition fallback for songs with no other source.

When no web database has a song and there's no booklet, we transcribe the audio
with whisper.cpp and treat the recognized text as the lyrics. This is the last
resort — quality is lower than curated lyrics, but it fills the gap for truly
obscure tracks (e.g. obscure maimai songs not on SilentBlue).

It reuses whisper.cpp (Vulkan on RX 9060 XT). The transcribed segments carry
their own timestamps, so this fetcher returns lyric lines together with timing.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..models import LyricLine, Lyrics
from ..utils import get_session
from .base import BaseFetcher

# Music-note tokens whisper emits when it hears music but no vocals (e.g. an
# instrumental with no singing): "♪~", "♪ ♫", "♪♪", etc. These are NOT lyrics —
# writing them produces a useless .lrc full of "♪~". We drop them so best-effort
# transcription only keeps genuine speech/singing (e.g. spoken-word samples like
# ATLAS RUSH's "Deliver the file").
_MUSIC_NOTE_RE = re.compile(r"[\u2669\u266a\u266b\u266c\u266d♫♪]")
_HAVE_TEXT_RE = re.compile(r"[a-zA-Z0-9\u3040-\u30ff\u3400-\u9fff]")


def _is_lyric_line(text: str) -> bool:
    """True if a transcribed segment is really lyrics/speech (not music notes)."""
    stripped = _MUSIC_NOTE_RE.sub("", text)
    return bool(_HAVE_TEXT_RE.search(stripped))


def filter_whisper_lines(segments: list[dict]) -> list[LyricLine]:
    """Turn whisper segments into lyric lines, dropping noise.

    Two pass-filters, in order:
      1. drop music-note-only / empty segments (instrumental w/ no vocals), and
      2. drop HALLUCINATION LOOPS — whisper repeating one phrase many times over
         a song (告げよ's "メルエリアルリン" ×137, or ATLAS RUSH looping
         "「Santus Crush」"). A real song repeats a chorus a few times; a loop is
         a phrase appearing a large number of times. If one phrase dominates the
         whole track it's a hallucination, not lyrics — treat as no lyrics.

    Returns a flat lyric-line list (empty when nothing real remains, so the
    pipeline treats the track as "no lyrics" and skips it).
    """
    cands = [s for s in segments if _is_lyric_line(s["text"])]
    if not cands:
        return []

    def _norm(t: str) -> str:
        return _MUSIC_NOTE_RE.sub("", t).strip()

    # count every phrase over the WHOLE transcription (before dedup), so a loop
    # that repeats at scattered timestamps is visible.
    freq: dict[str, int] = {}
    for s in cands:
        n = _norm(s["text"])
        freq[n] = freq.get(n, 0) + 1

    # a phrase appearing >= this many times is a hallucination loop. Real songs
    # (even Cryptarithm's 3x chorus) usually repeat < 4 times.
    LOOP_REPEATS = 4
    loop_total = sum(c for c in freq.values() if c >= LOOP_REPEATS)
    if loop_total >= LOOP_REPEATS and loop_total >= 0.5 * len(cands):
        # the track is dominated by one (or a few) repeated hallucinated phrase
        return []

    # remove loop phrases, then collapse exact consecutive repeats
    keep = [s for s in cands if freq[_norm(s["text"])] < LOOP_REPEATS]
    cleaned: list[LyricLine] = []
    prev_norm = None
    for s in keep:
        n = _norm(s["text"])
        if n == prev_norm:
            continue
        cleaned.append(LyricLine(text=s["text"], start=s["from"] / 1000.0))
        prev_norm = n
    return cleaned


class WhisperFetcher(BaseFetcher):
    """Fetch lyrics by transcribing the audio with whisper.cpp.

    Unlike the web fetchers this needs the audio file itself. ``fetch`` takes
    the audio path as ``title`` (a path string) so it fits the BaseFetcher
    interface; the alignment step will already have its own transcription, so
    this is primarily for exposing the recognized text.
    """

    name = "whisper"

    def __init__(self, aligner=None):
        from ..aligner.whisper_cpp import WhisperCppAligner

        self.aligner = aligner or WhisperCppAligner()

    def fetch(self, title: str, artist: str = "") -> Lyrics:
        """Transcribe an audio file (``title`` is the file path)."""
        audio = Path(title)
        if not audio.exists():
            return Lyrics(source=self.name, title=title, artist=artist)
        segs = self.aligner._segments(audio)
        lines = filter_whisper_lines(segs)
        if not lines:
            return Lyrics(source=self.name, title=str(audio.stem), artist=artist)
        return Lyrics(source=self.name, title=str(audio.stem), artist=artist, lines=lines)

    @staticmethod
    def current_lyrics() -> "Lyrics":
        raise NotImplementedError("WhisperFetcher.transcribe is audio-specific")
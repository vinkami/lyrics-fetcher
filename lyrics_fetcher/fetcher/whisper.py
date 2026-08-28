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
        # drop music-note-/empty-only segments (instrumental with no vocals);
        # if nothing real remains the fetch returns empty Lyrics so the pipeline
        # treats the track as "no lyrics" and skips it.
        lines = [
            LyricLine(text=s["text"], start=s["from"] / 1000.0)
            for s in segs if _is_lyric_line(s["text"])
        ]
        if not lines:
            return Lyrics(source=self.name, title=str(audio.stem), artist=artist)
        return Lyrics(source=self.name, title=str(audio.stem), artist=artist, lines=lines)

    @staticmethod
    def current_lyrics() -> "Lyrics":
        raise NotImplementedError("WhisperFetcher.transcribe is audio-specific")
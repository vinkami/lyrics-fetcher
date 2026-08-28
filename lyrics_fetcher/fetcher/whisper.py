"""Whisper-as-fetcher — AI recognition fallback for songs with no other source.

When no web database has a song and there's no booklet, we transcribe the audio
with whisper.cpp and treat the recognized text as the lyrics. This is the last
resort — quality is lower than curated lyrics, but it fills the gap for truly
obscure tracks (e.g. obscure maimai songs not on SilentBlue).

It reuses whisper.cpp (Vulkan on RX 9060 XT). The transcribed segments carry
their own timestamps, so this fetcher returns lyric lines together with timing.
"""
from __future__ import annotations

from pathlib import Path

from ..models import LyricLine, Lyrics
from ..utils import get_session
from .base import BaseFetcher


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
        lines = [LyricLine(text=s["text"], start=s["from"] / 1000.0) for s in segs]
        return Lyrics(source=self.name, title=str(audio.stem), artist=artist, lines=lines)

    @staticmethod
    def current_lyrics() -> "Lyrics":
        raise NotImplementedError("WhisperFetcher.transcribe is audio-specific")
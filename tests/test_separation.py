"""Tests for vocal separation (demucs) integration.

The separation module imports demucs lazily, so these tests run even without
the heavy dependency installed — they mock the separator where needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_make_separator_returns_none_when_demucs_absent(monkeypatch):
    """If demucs can't be imported, --separation degrades to None (no crash)."""
    import lyrics_fetcher.separation as sep

    def boom(device=None, model=None, model_dir=None):
        raise ImportError("no demucs")

    monkeypatch.setattr(sep, "VocalSeparator", boom)
    assert sep.make_separator(device=None) is None


def test_pipeline_align_uses_separator_when_set(tmp_path):
    """Pipeline._align separates the audio and aligns against the stem."""
    from lyrics_fetcher.pipeline import Pipeline
    from lyrics_fetcher.models import Lyrics, LyricLine

    audio = tmp_path / "song.flac"
    audio.write_bytes(b"fake")

    class FakeSep:
        def separate(self, audio, out_dir=None):
            stem = out_dir / "song_vocals.wav"
            stem.write_bytes(b"stem")
            return stem

        def __call__(self, *a, **k):
            return None

    seen = {}

    class FakeAligner:
        def align(self, audio, lyrics):
            seen["audio"] = audio
            return [type("T", (), {"text": "x", "start": 0.0, "end": 0.0})()]

    pipe = Pipeline(aligner=FakeAligner(), separator=FakeSep())
    ly = Lyrics(source="test", lines=[LyricLine("x")])
    pipe._align(audio, ly)
    # aligner must have received the separated stem, not the raw audio
    assert seen["audio"].name == "song_vocals.wav"


def test_pipeline_align_falls_back_on_separator_error(tmp_path):
    """If separation raises, Pipeline falls back to aligning the raw audio."""
    from lyrics_fetcher.pipeline import Pipeline
    from lyrics_fetcher.models import Lyrics, LyricLine

    audio = tmp_path / "song.flac"
    audio.write_bytes(b"fake")

    class BoomSep:
        def separate(self, audio, out_dir=None):
            raise RuntimeError("separation failed")

    seen = {}

    class FakeAligner:
        def align(self, audio, lyrics):
            seen["audio"] = audio
            return []

    pipe = Pipeline(aligner=FakeAligner(), separator=BoomSep())
    ly = Lyrics(source="test", lines=[LyricLine("x")])
    pipe._align(audio, ly)
    # fallback: aligned the raw audio
    assert seen["audio"] == audio
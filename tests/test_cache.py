"""Tests for the SQLite cache — round-trip put/get for lyrics + OCR.

Uses placeholder strings only (no real lyric content).
"""
import tempfile
from pathlib import Path

from lyrics_fetcher.cache import LyricsCache


def _tmp_cache() -> LyricsCache:
    d = tempfile.mkdtemp()
    return LyricsCache(db_path=Path(d) / "test.db")


def test_lyrics_roundtrip():
    c = _tmp_cache()
    payload = {"source": "utaten", "title": "Some Song", "lines": ["a", "b"]}
    c.put_lyrics("utaten", "Some Song", "Artist", payload)
    got = c.get_lyrics("utaten", "Some Song", "Artist")
    assert got is not None
    assert got["title"] == "Some Song"
    assert got["lines"] == ["a", "b"]
    c.close()


def test_lyrics_overwrite_same_key():
    c = _tmp_cache()
    c.put_lyrics("g", "T", "A", {"v": 1})
    c.put_lyrics("g", "T", "A", {"v": 2})
    got = c.get_lyrics("g", "T", "A")
    assert got["v"] == 2
    c.close()


def test_lyrics_missing_returns_none():
    c = _tmp_cache()
    assert c.get_lyrics("nope", "T", "A") is None
    c.close()


def test_lyrics_keyed_by_all_fields():
    c = _tmp_cache()
    c.put_lyrics("g", "Title", "ArtistA", {"x": 1})
    c.put_lyrics("g", "Title", "ArtistB", {"x": 2})
    assert c.get_lyrics("g", "Title", "ArtistA")["x"] == 1
    assert c.get_lyrics("g", "Title", "ArtistB")["x"] == 2
    c.close()


def test_ocr_roundtrip():
    c = _tmp_cache()
    img = Path("/some/absolute/image.jpg")
    c.put_ocr(img, "transcribed line one\nline two")
    assert c.get_ocr(img) == "transcribed line one\nline two"
    c.close()


def test_ocr_missing_returns_none():
    c = _tmp_cache()
    assert c.get_ocr(Path("/nope/missing.jpg")) is None
    c.close()
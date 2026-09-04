"""Tests for FetchOrchestrator: title-variant retry + no negative caching.

Pure logic — fetchers are fakes, no network.
"""
import tempfile
from pathlib import Path

from lyrics_fetcher.cache import LyricsCache
from lyrics_fetcher.fetcher.base import BaseFetcher
from lyrics_fetcher.fetcher.orchestrator import FetchOrchestrator
from lyrics_fetcher.models import LyricLine, Lyrics


class FakeFetcher(BaseFetcher):
    name = "fake"

    def __init__(self, only=None):
        self.calls = []
        self.only = only  # set of titles this source 'has'

    def fetch(self, title, artist=""):
        self.calls.append(title)
        if self.only is None or title in self.only:
            return Lyrics(source=self.name, title=title, artist=artist,
                          lines=[LyricLine(text="la la")])
        return Lyrics(source=self.name)  # empty = not found

    @property
    def source_name(self):
        return self.name


def _orch(cache=None):
    return FetchOrchestrator(sources=[], cache=cache)  # sources injected per-test


def test_variant_retry_finds_bare_title():
    f = FakeFetcher(only={"8番出口"})
    o = FetchOrchestrator(sources=[], cache=None)
    o.sources = [f]
    got = o.fetch_best("8番出口 Short ver", "鏡音リン")
    assert got and f.calls == ["8番出口 Short ver", "8番出口"]


def test_instrumental_gets_single_attempt():
    f = FakeFetcher(only=set())
    o = FetchOrchestrator(sources=[], cache=None)
    o.sources = [f]
    got = o.fetch_best("8番出口 Inst(Full ver)", "x")
    assert not got
    assert f.calls == ["8番出口 Inst(Full ver)"]  # no resurrecting vocal lyrics


def test_negative_not_cached_and_retried_next_run():
    cache = LyricsCache(db_path=Path(tempfile.mkdtemp()) / "t.db")
    f1 = FakeFetcher(only=set())
    o = FetchOrchestrator(sources=[], cache=cache)
    o.sources = [f1]
    assert not o.fetch_best("New Song", "A")
    assert cache.get_lyrics("fake", "New Song", "A") is None  # nothing stored
    # source 'gains' the song later -> next run finds it (no poisoned cache)
    f2 = FakeFetcher(only={"New Song"})
    o.sources = [f2]
    got = o.fetch_best("New Song", "A")
    assert got and f2.calls == ["New Song"]
    assert cache.get_lyrics("fake", "New Song", "A") is not None  # positive stored
    cache.close()


def test_legacy_negative_row_ignored_on_read():
    # old versions cached empty payloads; they must not be treated as hits
    cache = LyricsCache(db_path=Path(tempfile.mkdtemp()) / "t.db")
    cache.put_lyrics("fake", "Song", "A", {"source": "fake", "lines": []})
    f = FakeFetcher(only={"Song"})
    o = FetchOrchestrator(sources=[], cache=cache)
    o.sources = [f]
    got = o.fetch_best("Song", "A")
    assert got and f.calls == ["Song"]  # cache miss -> went to network
    cache.close()

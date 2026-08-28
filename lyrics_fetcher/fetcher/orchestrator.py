"""Fetcher orchestrator — tries multiple sources in order, returns best result.

Integrates an optional SQLite cache so repeated fetches skip the network/OCR.
"""
from __future__ import annotations

from ..cache import LyricsCache
from .base import BaseFetcher
from .genius import GeniusFetcher
from .silentblue import SilentBlueFetcher
from .utaten import UtatenFetcher

DEFAULT_SOURCES = [UtatenFetcher, GeniusFetcher, SilentBlueFetcher]


def _payload(l):
    """Serialize a Lyrics object to a cache-friendly dict."""
    return {
        "source": l.source,
        "source_url": l.source_url,
        "title": l.title,
        "artist": l.artist,
        "lines": [{"text": ln.text, "start": ln.start, "ruby": ln.ruby} for ln in l.lines],
        "ruby_all": l.ruby_all,
    }


def _from_payload(d):
    from ..models import LyricLine, Lyrics

    return Lyrics(
        source=d["source"],
        source_url=d.get("source_url", ""),
        title=d.get("title", ""),
        artist=d.get("artist", ""),
        lines=[LyricLine(text=x["text"], start=x.get("start", 0.0), ruby=x.get("ruby", {})) for x in d.get("lines", [])],
        ruby_all=d.get("ruby_all", {}),
    )


class FetchOrchestrator:
    """Try each fetcher until one returns lyrics, exposing choices on failure."""

    def __init__(self, sources: list[type[BaseFetcher]] | None = None,
                 cache: LyricsCache | None = None):
        self.sources = [s() for s in (sources or DEFAULT_SOURCES)]
        self.cache = cache

    def fetch_all(self, title: str, artist: str = "") -> dict[str, "object"]:
        """Run every source; return {source_name: Lyrics}."""
        results = {}
        for f in self.sources:
            try:
                cached = self.cache.get_lyrics(f.source_name, title, artist) if self.cache else None
                if cached:
                    results[f.source_name] = _from_payload(cached)
                    continue
                l = f.fetch(title, artist)
                if self.cache:
                    self.cache.put_lyrics(f.source_name, title, artist, _payload(l))
                results[f.source_name] = l
            except Exception as e:  # a source failing shouldn't kill others
                results[f.source_name] = f"error: {e}"
        return results

    def fetch_best(self, title: str, artist: str = "") -> "object | None":
        """Return first non-empty Lyrics by source priority, else None."""
        for f in self.sources:
            try:
                cached = self.cache.get_lyrics(f.source_name, title, artist) if self.cache else None
                if cached:
                    return _from_payload(cached)
                l = f.fetch(title, artist)
                if self.cache:
                    self.cache.put_lyrics(f.source_name, title, artist, _payload(l))
                if l:
                    return l
            except Exception:
                continue
        return None

    def source_names(self) -> list[str]:
        return [f.source_name for f in self.sources]
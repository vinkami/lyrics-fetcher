"""Fetcher orchestrator — tries multiple sources in order, returns best result.

Integrates an optional SQLite cache so repeated fetches skip the network/OCR.
"""
from __future__ import annotations

from ..cache import LyricsCache
from ..utils import title_variants
from .base import BaseFetcher
from .genius import GeniusFetcher
from .silentblue import SilentBlueFetcher
from .utaten import UtatenFetcher

DEFAULT_SOURCES = [UtatenFetcher, GeniusFetcher, SilentBlueFetcher]


def _cached_hit(cached):
    """Decide a cache row: return Lyrics on a positive hit, None on miss.

    Empty payloads (negative results) are NOT hits — a source that missed
    once must get a fresh chance (new sources, retagged titles, fixed
    fetchers). Negatives are never written anymore (see _try_source).
    """
    if not cached:
        return None
    l = _from_payload(cached)
    return l if l else None


def _try_source(f: BaseFetcher, title: str, artist: str,
                cache: LyricsCache | None):
    """Fetch one source across title variants; return Lyrics (maybe empty).

    Disc titles often carry version suffixes ('... Short ver') that lyrics
    databases don't index, so on a miss we retry with stripped variants.
    Only non-empty results are cached (negatives used to poison re-runs).
    """
    variants = title_variants(title)
    last = None
    for t in variants:
        if cache:
            hit = _cached_hit(cache.get_lyrics(f.source_name, t, artist))
            if hit is not None:
                return hit
        try:
            l = f.fetch(t, artist)
        except Exception:
            raise
        if l:
            if cache:
                cache.put_lyrics(f.source_name, t, artist, _payload(l))
            return l
        last = l
    return last  # empty; caller records the error/miss


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
                results[f.source_name] = _try_source(f, title, artist, self.cache)
            except Exception as e:  # a source failing shouldn't kill others
                results[f.source_name] = f"error: {e}"
        return results

    def fetch_best(self, title: str, artist: str = "") -> "object | None":
        """Return first non-empty Lyrics by source priority, else None."""
        for f in self.sources:
            try:
                l = _try_source(f, title, artist, self.cache)
            except Exception:
                continue
            if l:
                return l
        return None

    def source_names(self) -> list[str]:
        return [f.source_name for f in self.sources]
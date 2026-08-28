"""Fetcher orchestrator — tries multiple sources in order, returns best result."""
from __future__ import annotations

from .base import BaseFetcher
from .genius import GeniusFetcher
from .silentblue import SilentBlueFetcher
from .utaten import UtatenFetcher

DEFAULT_SOURCES = [UtatenFetcher, GeniusFetcher, SilentBlueFetcher]


class FetchOrchestrator:
    """Try each fetcher until one returns lyrics, exposing choices on failure."""

    def __init__(self, sources: list[type[BaseFetcher]] | None = None):
        self.sources = [s() for s in (sources or DEFAULT_SOURCES)]

    def fetch_all(self, title: str, artist: str = "") -> dict[str, "object"]:
        """Run every source; return {source_name: Lyrics}."""
        results = {}
        for f in self.sources:
            try:
                results[f.source_name] = f.fetch(title, artist)
            except Exception as e:  # a source failing shouldn't kill others
                results[f.source_name] = f"error: {e}"
        return results

    def fetch_best(self, title: str, artist: str = "") -> "object | None":
        """Return first non-empty Lyrics by source priority, else None."""
        for f in self.sources:
            try:
                r = f.fetch(title, artist)
                if r:
                    return r
            except Exception:
                continue
        return None

    def source_names(self) -> list[str]:
        return [f.source_name for f in self.sources]
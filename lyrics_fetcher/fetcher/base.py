"""Abstract provider interface for fetching lyrics from an online source."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Lyrics


class BaseFetcher(ABC):
    """Any source that produces plain-text lyrics for a song.

    Subclasses implement :meth:`fetch`, returning a :class:`Lyrics` container
    (possibly empty / falsy when the source has no match).
    """

    #: human-readable source name used in Lyrics.source and CLI output
    name = "base"

    @abstractmethod
    def fetch(self, title: str, artist: str = "") -> Lyrics:
        """Return lyrics for the song, or an empty Lyrics if not found."""

    @property
    def source_name(self) -> str:
        return self.name
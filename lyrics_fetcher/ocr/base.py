"""OCR.fetcher.base — abstract OCR provider (image -> Lyrics).

OCR is treated as a lyrics "fetcher": given booklet image(s), it produces
plain-text lyrics for a specific song, just like the web fetchers do.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import LyricLine, Lyrics


class BaseOCR(ABC):
    name = "ocr-base"

    @abstractmethod
    def ocr(self, image: Path) -> str:
        """Return the raw transcribed lyrics text from one booklet image."""

    def fetch(self, image: Path, title: str = "", artist: str = "") -> Lyrics:
        """OCR an image and wrap it as Lyrics.

        A leading bracketed title header (e.g. ``[Song Title]``) printed on
        the page is treated as a header, not a lyric line, and dropped.
        """
        text = self.ocr(image).strip()
        lines = [l for l in (x.strip() for x in text.splitlines()) if l]
        # drop a leading header line like [Song Title] or a bare 4-6 char title
        if len(lines) > 1:
            first = lines[0]
            if first.startswith("[") and first.endswith("]"):
                lines = lines[1:]
            elif len(first) <= 8 and " " not in first and not any(
                c in first for c in "、。！？"  # not a full sentence
            ):
                lines = lines[1:]
        return Lyrics(
            source=self.name,
            title=title,
            artist=artist,
            lines=[LyricLine(t) for t in lines],
        )
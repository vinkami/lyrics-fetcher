"""Abstract aligner interface: known lyrics + audio -> timed lyric lines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..models import Lyrics


@dataclass
class TimedLine:
    """A lyric line with an assigned start (and optional end) time in seconds."""

    text: str
    start: float
    end: float = 0.0


class BaseAligner(ABC):
    name = "base"

    @abstractmethod
    def align(self, audio: Path, lyrics: Lyrics) -> list[TimedLine]:
        """Return timed lyric lines from known lyrics + the audio file."""
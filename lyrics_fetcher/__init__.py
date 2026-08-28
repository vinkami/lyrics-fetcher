"""lyrics-fetcher — fetch, align, and write lyric files for songs.

Pipeline: read audio metadata -> fetch lyrics (web or OCR) -> align to
timestamps with whisper -> write .lrc (+ HTML companion).
"""

__version__ = "0.1.0"

from .models import LyricLine, Lyrics, SongMeta  # noqa: F401
"""Domain models shared across lyrics-fetcher.

These are the contract between the three pipeline stages:
  fetch (Part 1) -> align (Part 2) -> output (Part 3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SongMeta:
    """Metadata read from an audio file's tags."""

    title: str = ""
    artist: str = ""
    album: str = ""
    albumartist: str = ""
    tracknumber: str = ""
    date: str = ""
    musicbrainz_id: str = ""

    @classmethod
    def from_path(cls, audio: Path) -> "SongMeta":
        """Read metadata from an audio file (FLAC/MP3 via mutagen)."""
        from mutagen import File as MutagenFile

        audio_file = MutagenFile(str(audio))
        meta = cls()
        if audio_file is None:
            return meta
        # Mutagen exposes tags via .get; FLAC uses uppercase Vorbis keys,
        # MP3 uses ID3 (Title/Artist/Album). We probe several spellings.
        def _g(*keys: str):
            for k in keys:
                v = audio_file.get(k)
                if v:
                    if isinstance(v, list):
                        v = v[0]
                    s = str(v)
                    if s:
                        return s
            return ""

        meta.title = _g("title", "Title", "TIT2")
        meta.artist = _g("artist", "Artist", "TPE1")
        meta.album = _g("album", "Album", "TALB")
        meta.albumartist = _g("albumartist", "AlbumArtist", "ALBUMARTIST", "TPE2")
        meta.tracknumber = _g("tracknumber", "Tracknumber", "track", "TRCK")
        meta.date = _g("date", "Date", "TDRC")
        meta.musicbrainz_id = _g("musicbrainz_trackid", "MusicBrainz_TrackID", "MusicBrainz Album Id")
        return meta


@dataclass
class LyricLine:
    """A single lyric line with an optional start time (seconds)."""

    text: str
    start: float = 0.0
    # optional furigana: nanase runs -> reading, filled only when the source had them
    ruby: dict[str, str] = field(default_factory=dict)


@dataclass
class Lyrics:
    """Container for fetched lyrics before timing is assigned."""

    source: str  # e.g. "utaten", "genius", "silentblue", "ocr-vlm", "whisper"
    source_url: str = ""
    title: str = ""
    artist: str = ""

    lines: list[LyricLine] = field(default_factory=list)
    # plain text convenience (utaten stores full ruby pairs for furigana output)
    ruby_all: dict[str, str] = field(default_factory=dict)

    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    def __bool__(self) -> bool:
        return bool(self.lines)
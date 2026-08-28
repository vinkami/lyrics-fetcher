"""Batch processor — process an entire album folder (or arbitrary audio).

Strategy:
  - Discover audio files (flac/mp3/m4a/ogg) in the album dir.
  - Discover booklet page images (jpg/png/webp) in a ``booklet`` subdir
    (or the same dir when --booklet given).
  - For each booklet page: OCR it, try to identify which track it belongs to
    by matching the first-line ``[title]`` header against each audio file's
    metadata title. Pages whose header matches a track are assigned to it.
  - Process each audio: fetch lyrics (OCR-first when a matching page exists,
    else web) -> align with whisper -> write to Jellyfin layout (next to audio)
    or an out_dir.

Matching is by METADATA title so filename/number differences don't matter.
Unmatched pages are left unprocessed and reported.
"""
from __future__ import annotations

import re
from pathlib import Path

from .models import LyricLine, Lyrics, SongMeta
from .pipeline import Pipeline
from .utils import MUSIC_DIR, slugify


class BookletMapper:
    """Assocaite booklet pages to audio tracks by OCR'd title headers."""

    #: audio extensions we recognize
    AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".ogg", ".wav"}

    def __init__(self, audio_dir: Path, ocr):
        self.audio_dir = audio_dir
        self.ocr = ocr
        self._ocr_cache: dict[Path, str] = {}

    def ocr_page(self, img: Path) -> str:
        """OCR a page, caching the result so mapping+fetch don't OCR twice."""
        if img not in self._ocr_cache:
            self._ocr_cache[img] = self.ocr.ocr(img)
        return self._ocr_cache[img]

    # ---- discovery ----
    def discover_audio(self) -> list[Path]:
        return sorted(
            p for p in self.audio_dir.rglob("*")
            if p.suffix.lower() in self.AUDIO_EXTS and not p.name.startswith(".")
        )

    def discover_booklet(self, booklet_dir: Path | None = None) -> list[Path]:
        if booklet_dir and booklet_dir.exists():
            base = booklet_dir
        else:
            base = self.audio_dir / "booklet"
            if not base.exists():
                return []
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return sorted(p for p in base.rglob("*") if p.suffix.lower() in exts)

    # ---- header -> track mapping ----
    @staticmethod
    def _extract_title(text: str) -> str | None:
        """Pull a song title from OCR'd page text.

        Accepts a bracketed header "[命を振り回せ]" or the first non-empty line.
        """
        lines = [l.strip() for l in (x.strip() for x in text.splitlines()) if l]
        if not lines:
            return None
        first = lines[0]
        m = re.match(r"^\[([^\]]+)\]$", first)
        if m:
            return m.group(1).strip()
        # fall back to first line if it's short and looks like a title (no spaces/kanji slug)
        if len(first) <= 30 and not re.search(r"[、。！？\s。]", first):
            return first
        return None

    def map_pages_to_tracks(self, images: list[Path],
                            tracks: list[Path],
                            prefer_ocr_for_unmatched: bool = False) -> dict[Path, Path | None]:
        """Return {page: track_path} using OCR'd title headers.

        A page with no match maps to None (reported separately).
        """
        # build track lookup: slugified title -> path
        title_index: dict[str, list[Path]] = {}
        for t in tracks:
            meta = SongMeta.from_path(t)
            if meta.title:
                title_index.setdefault(slugify(meta.title), []).append(t)

        mapping: dict[Path, Path | None] = {}
        for img in images:
            try:
                text = self.ocr.ocr(img)
            except Exception:
                mapping[img] = None
                continue
            title = self._extract_title(text)
            match = None
            if title:
                matches = title_index.get(slugify(title))
                if matches:
                    match = matches[0]
            mapping[img] = match

        # positional fallback: booklet pages canonically map one-per-track, in
        # order. Any page with no title match is paired with the next unmatched
        # track in filename order. This handles pages whose OCR dropped the
        # title header (VLM output is nondeterministic), and pages whose title
        # differs from the metadata title while order still holds.
        matched_tracks = {t for t in mapping.values() if t is not None}
        unmatched_pages = [i for i, t in mapping.items() if t is None]
        unmatched_tracks = [t for t in tracks if t not in matched_tracks]
        for page, track in zip(unmatched_pages, unmatched_tracks):
            mapping[page] = track
        return mapping


def batch_album(
    album_dir: Path,
    out_dir: Path | None = None,
    booklet_dir: Path | None = None,
    jellyfin: bool = False,
    pipeline: Pipeline | None = None,
    prefer_ocr: bool = True,
    write_html: bool = True,
    verbose: bool = False,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Process every track in an album.

    Returns (processed_lrc_paths, [(page, reason) unmatched]).
    """
    pipe = pipeline or Pipeline()
    mapper = BookletMapper(album_dir, pipe.ocr)

    tracks = mapper.discover_audio()
    images = mapper.discover_booklet(booklet_dir)

    if verbose:
        print(f"tracks: {len(tracks)}  booklet pages: {len(images)}")

    processed: list[Path] = []
    unmatched: list[tuple[Path, str]] = [(i, "") for i in images]  # placeholder

    # map pages -> tracks
    page_to_track = mapper.map_pages_to_tracks(images, tracks)

    # process each audio, using its matched page if any
    for audio in tracks:
        meta = SongMeta.from_path(audio)
        # find a page mapped to this audio
        page = next((p for p, t in page_to_track.items() if t == audio), None)
        try:
            # If a page is mapped to this track, reuse its cached OCR text as
            # pre-acquired lyrics (authoritative for the album, avoids re-OCR).
            pre_lyrics = None
            if page is not None:
                text = mapper.ocr_page(page)
                lines = [LyricLine(l) for l in (x.strip() for x in text.splitlines()) if l]
                if lines:
                    # skip a leading [title] header line as a lyric
                    if len(lines) > 1 and lines[0].text.startswith("["):
                        lines = lines[1:]
                    pre_lyrics = Lyrics(
                        source="ocr-vlm", title=meta.title or "",
                        artist=meta.artist or "", lines=lines,
                    )
            result = pipe.run(
                audio=audio,
                out_dir=out_dir,
                image=None if pre_lyrics else page,
                write_html=write_html,
                prefer_ocr=True,
                jellyfin=jellyfin,
                lyrics=pre_lyrics,
            )
            processed.append(result.lrc_path)
            if verbose:
                print(f"  OK  {audio.name}: {result.lyrics_source} ({result.lines} lines)")
        except Exception as e:
            if verbose:
                print(f"  ERR {audio.name}: {e}")

    # unmatched pages
    unmatched = [(i, "no matching track") for i, t in page_to_track.items()
                 if t is None]
    return processed, unmatched
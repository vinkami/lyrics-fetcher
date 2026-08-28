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

from pathlib import Path

from .models import LyricLine, Lyrics, SongMeta
from .pipeline import Pipeline
from .utils import MUSIC_DIR, _norm_ja


def _fuzz_ratio(a: str, b: str) -> int:
    from thefuzz import fuzz

    return fuzz.ratio(a, b)


class BookletMapper:
    """Associate booklet pages to audio tracks via multi-song VLM extraction.

    A booklet page usually contains MORE THAN ONE song (e.g. PRiSM photo A holds
    プリズム + the first half of RondeauX). Instead of treating one page as one
    song (which leaks adjacent songs' lyrics into each other), we ask the VLM to
    split each page ``{song_title: lyrics}``, then:
      - normalize each block's title to a canonical on-disc track,
      - reassemble a song that spans multiple pages by concatenating its blocks
        in page order,
      - drop "phantom" blocks (titles matching no track) and report them.
    """

    #: audio extensions we recognize
    AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".ogg", ".wav"}

    def __init__(self, audio_dir: Path, ocr):
        self.audio_dir = audio_dir
        self.ocr = ocr

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

    # ---- multi-song aggregation ----
    @staticmethod
    def _is_title_only_block(text: str, label: str) -> bool:
        """True if a VLM block is just a title/header, not real lyrics.

        Index/title pages (e.g. a booklet section listing song names) come back
        as blocks whose content is only the printed song title — a single line
        that equals the title. Treating that as lyrics produces a useless
        ``[00:00.00]QUIQ`` line for a song that has no lyrics. Returns True when
        the block is short and its cleaned text matches the label (title) itself.
        """
        lines = [l for l in (x.strip() for x in text.splitlines()) if l]
        if not lines:
            return False
        if len(lines) >= 3:
            return False  # real lyric pages have multiple lines
        lbl = _norm_ja(label)
        if not lbl:
            return False
        joined = _norm_ja("".join(lines))
        # whole block == the title, or title dominates it (header/echo)
        return bool(joined) and (lbl in joined or joined in lbl
                                 or _fuzz_ratio(joined, lbl) >= 75)

    def collect_album_songs(self, images: list[Path],
                            tracks: list[Path]
                            ) -> tuple[dict[Path, Lyrics], list[tuple[str, str]]]:
        """Aggregate per-song lyrics across all booklet pages.

        Returns ``(per_track_lyrics, phantoms)`` where:
          - ``per_track_lyrics[track_path]`` = Lyrics for that track (concatenated
            across every page that contained it, in page order), or absent if the
            track has no lyrics on any page;
          - ``phantoms`` = [(page_name, song_label), ...] for blocks that matched
            no on-disc track (e.g. a song-section header the VLM split out, like
            "Sanctus" / "レクイエム" inside RondeauX), or that are title-only
            headers (index/title pages), which are not lyrics.
        """
        # index canonical track title -> path
        by_title: dict[str, Path] = {}
        for t in tracks:
            title = (SongMeta.from_path(t).title or t.stem).strip()
            if title and title not in by_title:
                by_title[title] = t

        known = list(by_title)  # hints for the VLM + normalization targets

        # gather each page's blocks
        per_track_blocks: dict[Path, list[str]] = {t: [] for t in tracks}
        phantoms: list[tuple[str, str]] = []
        for img in images:
            blocks = self.ocr.extract_songs(img, known_titles=known)
            for label, text in blocks.items():
                track = by_title.get(label)
                if track is not None and self._is_title_only_block(text, label):
                    # a title-only block (index page) is NOT lyrics for this
                    # track; report it so it isn't silently turned into a line
                    phantoms.append((img.name, f"{label} (title-only)"))
                    continue
                if track is not None:
                    per_track_blocks[track].append(text)
                else:
                    phantoms.append((img.name, label))

        # build Lyrics per track
        per_track: dict[Path, Lyrics] = {}
        for track, blocks in per_track_blocks.items():
            if not blocks:
                continue
            meta = SongMeta.from_path(track)
            lines = [
                LyricLine(l)
                for block in blocks for l in block.splitlines() if l.strip()
            ]
            per_track[track] = Lyrics(
                source=self.ocr.name, title=meta.title or "",
                artist=meta.artist or "", lines=lines,
            )
        return per_track, phantoms


def batch_album(
    album_dir: Path,
    out_dir: Path | None = None,
    booklet_dir: Path | None = None,
    jellyfin: bool = False,
    pipeline: Pipeline | None = None,
    prefer_ocr: bool = True,
    write_html: bool = True,
    verbose: bool = False,
    best_effort: bool = False,
) -> tuple[list[Path], list[tuple[Path, str]], list[Path]]:
    """Process every track in an album.

    Returns (processed_lrc_paths, [(page, reason) unmatched], skipped_tracks).
    Tracks with no lyrics are SKIPPED by default (no .lrc written); with
    ``best_effort=True`` whisper transcribes them as a fallback.
    """
    pipe = pipeline or Pipeline()
    if best_effort:
        pipe.use_whisper_fallback = True
    mapper = BookletMapper(album_dir, pipe.ocr)

    tracks = mapper.discover_audio()
    images = mapper.discover_booklet(booklet_dir)

    if verbose:
        print(f"tracks: {len(tracks)}  booklet pages: {len(images)}")

    # multi-song OCR: aggregate per-track lyrics across pages
    per_track, phantoms = mapper.collect_album_songs(images, tracks)

    processed: list[Path] = []
    skipped: list[Path] = []
    for audio in tracks:
        meta = SongMeta.from_path(audio)
        pre = per_track.get(audio)
        try:
            result = pipe.run(
                audio=audio,
                out_dir=out_dir,
                image=None,
                write_html=write_html,
                prefer_ocr=prefer_ocr,
                jellyfin=jellyfin,
                lyrics=pre,
            )
            processed.append(result.lrc_path)
            if verbose:
                print(f"  OK  {audio.name}: {result.lyrics_source} ({result.lines} lines)")
        except RuntimeError as e:
            if "No lyrics found" in str(e):
                skipped.append(audio)
                if verbose:
                    print(f"  SKIP {audio.name}: no lyrics")
            else:
                if verbose:
                    print(f"  ERR {audio.name}: {e}")
        except Exception as e:
            if verbose:
                print(f"  ERR {audio.name}: {e}")

    # unmatched booklet content: phantom song blocks (no matching track)
    unmatched = [(img, f"unmatched song block: {label}") for img, label in phantoms]
    return processed, unmatched, skipped
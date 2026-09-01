"""Pipeline orchestrator — end-to-end: fetch (web or OCR) -> align -> write.

This is the single entry point for producing .lrc and .html for a song.
It wires the per-stage pieces (fetchers/OCR, aligner, writers) together without
any of them knowing about the others.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import Lyrics, SongMeta
from .fetcher.base import BaseFetcher
from .fetcher.orchestrator import FetchOrchestrator
from .ocr.base import BaseOCR
from .ocr.vision import VLMOcr
from .aligner.base import BaseAligner
from .aligner.whisper_cpp import WhisperCppAligner
from .output.writers import HtmlWriter, LrcWriter


@dataclass
class PipelineResult:
    lyrics_source: str
    lrc_path: Path
    html_path: Path | None
    lines: int


class Pipeline:
    """Produce .lrc (+companion .html) for one song.

    Sources for lyrics are tried in priority order: web fetchers first, then
    OCR (VLM) on a booklet image if provided. Alignment uses whisper.cpp.
    """

    def __init__(
        self,
        fetcher: BaseFetcher | FetchOrchestrator | None = None,
        ocr: BaseOCR | None = None,
        aligner: BaseAligner | None = None,
        use_whisper_fallback: bool = False,
        separator=None,
    ):
        # default: web fetcher orchestrator + whisper aligner; OCR optional
        self.fetcher = fetcher if fetcher is not None else FetchOrchestrator()
        self.ocr = ocr
        self.aligner = aligner if aligner is not None else WhisperCppAligner()
        # Optional vocal separator (demucs). When set, the audio is separated to
        # a vocal stem before alignment — improves intro timing on BGM-dense
        # songs (see separation.py). Never default (can shift already-good songs).
        self.separator = separator
        # Default OFF: whisper is poor at Japanese singing, so tracks with no
        # lyrics are skipped (no .lrc) unless the user opts into best-effort.
        self.use_whisper_fallback = use_whisper_fallback
        self._current_audio: Path | None = None

    # ---- lyrics acquisition (web or OCR) ----
    def _fetch_lyrics(self, meta: SongMeta, image: Path | None = None,
                      prefer_ocr: bool = False, lyrics: Lyrics | None = None) -> Lyrics:
        """Get lyrics for a song.

        If ``lyrics`` is given it is used directly (caller already acquired
        them, e.g. batch OCR). Otherwise tries web fetchers first; when a
        booklet ``image`` is provided AND ``prefer_ocr`` is true, OCR is tried
        first (it's authoritative for that exact album), web as fallback.
        """
        if lyrics:
            return lyrics
        # optional OCR-first mode (used when the user explicitly gives a booklet)
        if prefer_ocr and self.ocr and image and image.exists():
            ocr_lyrics = self.ocr.fetch(image, meta.title, meta.artist)
            if ocr_lyrics:
                return ocr_lyrics

        # web fetchers (orchestrator, or a single fetcher)
        if isinstance(self.fetcher, FetchOrchestrator):
            lyrics = self.fetcher.fetch_best(meta.title, meta.artist)
            if lyrics:
                return lyrics
        elif self.fetcher is not None:
            lyrics = self.fetcher.fetch(meta.title, meta.artist)
            if lyrics:
                return lyrics

        # web-first fallback: OCR if no web match
        if self.ocr and image and image.exists():
            l = self.ocr.fetch(image, meta.title, meta.artist)
            if l:
                return l

        # last resort: transcribe the audio with whisper as the lyrics
        if self.use_whisper_fallback:
            audio = self._current_audio
            if audio:
                from .fetcher.whisper import WhisperFetcher
                wl = WhisperFetcher(self.aligner).fetch(str(audio), meta.artist)
                if wl:
                    return wl

        return Lyrics(source="none", title=meta.title, artist=meta.artist)

    # ---- alignment (optionally on a separated vocal stem) ----
    def _align(self, audio: Path, lyrics: Lyrics) -> list[TimedLine]:
        """Align known lyrics to the audio.

        When ``self.separator`` is set, first separate a dry-vocal stem and align
        against that (better intro timing on BGM-dense songs). If separation
        fails, fall back to aligning the raw audio so a separator problem never
        breaks a run.
        """
        target = audio
        if self.separator is not None:
            try:
                import tempfile

                scratch = Path(tempfile.mkdtemp(prefix="lf_sep_"))
                target = self.separator.separate(audio, out_dir=scratch)
            except Exception as e:
                # separation is best-effort; never let it break alignment
                print(f"warn: vocal separation failed for {audio.name}: {e}")
                target = audio
        return self.aligner.align(target, lyrics)

    # ---- main entry ----
    def run(
        self,
        audio: Path,
        out_dir: Path | None = None,
        image: Path | None = None,
        write_html: bool = True,
        prefer_ocr: bool = False,
        jellyfin: bool = False,
        lyrics: Lyrics | None = None,
    ) -> PipelineResult:
        """Process one song.

        Args:
            audio: path to the audio file.
            out_dir: directory for outputs. Ignored when ``jellyfin`` is True.
            image: optional booklet photo (uses OCR).
            write_html: also write a companion .html.
            prefer_ocr: try OCR before web when an image is given.
            jellyfin: write the .lrc next to the audio file (same stem) so
                Jellyfin/media players pick it up; also the .html alongside.
            lyrics: pre-acquired lyrics (avoids re-OCR in batch).
        """
        meta = SongMeta.from_path(audio)
        self._current_audio = audio
        lyrics = self._fetch_lyrics(meta, image, prefer_ocr=prefer_ocr, lyrics=lyrics)
        if not lyrics:
            raise RuntimeError(
                f"No lyrics found for {meta.title} (web fetchers failed and no OCR image given)"
            )

        timed = self._align(audio, lyrics)

        lrc_writer = LrcWriter()
        html_writer = HtmlWriter(lyrics)

        # Jellyfin layout: sibling of the audio file, same basename (.flac -> .lrc).
        # Otherwise write into out_dir.
        target_dir = audio.parent if jellyfin else out_dir
        if not target_dir:
            raise ValueError("out_dir is required when jellyfin=False")
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = audio.stem

        lrc_path = lrc_writer.write(
            target_dir / f"{stem}.lrc", meta.title or lyrics.title,
            meta.artist or lyrics.artist, meta.album, timed,
        )

        html_path = None
        if write_html:
            html_path = html_writer.write(
                target_dir / f"{stem}.html", meta.title or lyrics.title,
                meta.artist or lyrics.artist, meta.album, timed,
            )

        return PipelineResult(
            lyrics_source=lyrics.source,
            lrc_path=lrc_path,
            html_path=html_path,
            lines=len(timed),
        )
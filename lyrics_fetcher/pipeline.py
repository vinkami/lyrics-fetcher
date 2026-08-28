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
    ):
        # default: web fetcher orchestrator + whisper aligner; OCR optional
        self.fetcher = fetcher if fetcher is not None else FetchOrchestrator()
        self.ocr = ocr
        self.aligner = aligner if aligner is not None else WhisperCppAligner()

    # ---- lyrics acquisition (web or OCR) ----
    def _fetch_lyrics(self, meta: SongMeta, image: Path | None = None,
                      prefer_ocr: bool = False) -> Lyrics:
        """Get lyrics for a song.

        By default web fetchers are tried first. When a booklet ``image`` is
        provided AND ``prefer_ocr`` is true, OCR is tried first (it's
        authoritative for that exact album); web fetchers are a fallback.
        """
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
            return self.ocr.fetch(image, meta.title, meta.artist)
        return Lyrics(source="none", title=meta.title, artist=meta.artist)

    # ---- main entry ----
    def run(
        self,
        audio: Path,
        out_dir: Path,
        image: Path | None = None,
        write_html: bool = True,
        prefer_ocr: bool = False,
    ) -> PipelineResult:
        meta = SongMeta.from_path(audio)
        lyrics = self._fetch_lyrics(meta, image, prefer_ocr=prefer_ocr)
        if not lyrics:
            raise RuntimeError(
                f"No lyrics found for {meta.title} (web fetchers failed and no OCR image given)"
            )

        timed = self.aligner.align(audio, lyrics)

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = audio.stem
        lrc_writer = LrcWriter()
        lrc_path = lrc_writer.write(
            out_dir / f"{stem}.lrc", meta.title or lyrics.title,
            meta.artist or lyrics.artist, meta.album, timed,
        )

        html_path = None
        if write_html:
            html_path = HtmlWriter(lyrics).write(
                out_dir / f"{stem}.html", meta.title or lyrics.title,
                meta.artist or lyrics.artist, meta.album, timed,
            )

        return PipelineResult(
            lyrics_source=lyrics.source,
            lrc_path=lrc_path,
            html_path=html_path,
            lines=len(timed),
        )
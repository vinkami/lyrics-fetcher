"""CLI entry point — `lyrics-fetcher` with subcommands.

Subcommands:
  fetch   — fetch lyrics for a song / from a source; print or save to text
  ocr     — OCR a booklet image (vlm / tesseract) to stdout
  compile — align known lyrics against an audio file -> .lrc (+ .html)
  full    — end-to-end: metadata + fetch/OCR + align + write (recommended)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import Lyrics
from .pipeline import Pipeline
from .fetcher.orchestrator import FetchOrchestrator
from .fetcher.utaten import UtatenFetcher
from .fetcher.genius import GeniusFetcher
from .fetcher.silentblue import SilentBlueFetcher
from .ocr.vision import VLMOcr
from .ocr.tesseract import TesseractOcr
from .aligner.whisper_cpp import DEFAULT_BIN as DEFAULT_WHISPER_BIN
from .aligner.whisper_cpp import DEFAULT_MODEL as DEFAULT_WHISPER_MODEL
from .aligner.whisper_cpp import WhisperCppAligner
from .output.writers import LrcWriter

SOURCE_FACTORY = {
    "utaten": UtatenFetcher,
    "genius": GeniusFetcher,
    "silentblue": SilentBlueFetcher,
}


def _cmd_fetch(args: argparse.Namespace) -> int:
    if args.source:
        factory = SOURCE_FACTORY.get(args.source)
        if not factory:
            print(f"Unknown source '{args.source}'. Choose from: {', '.join(SOURCE_FACTORY)}")
            return 2
        res = factory().fetch(args.title, args.artist)
        results = {res.source: res}
    else:
        orch = FetchOrchestrator()
        results = orch.fetch_all(args.title, args.artist)

    for name, res in results.items():
        if isinstance(res, Lyrics) and res:
            print(f"[{name}] found ({len(res.lines)} lines)" + (f"  {res.source_url}" if res.source_url else ""))
            if args.verbose:
                print(res.text())
        elif isinstance(res, str):
            print(f"[{name}] {res}")
        else:
            print(f"[{name}] not found")
    return 0


def _cmd_ocr(args: argparse.Namespace) -> int:
    path = Path(args.image)
    ocr_engine = VLMOcr(api=args.api, model=args.model) if args.engine == "vlm" \
        else TesseractOcr(lang=args.language)
    try:
        print(ocr_engine.ocr(path))
    except Exception as e:
        print(f"OCR failed: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    audio = Path(args.audio)
    if args.lyrics_file and Path(args.lyrics_file).exists():
        lyrics_text = Path(args.lyrics_file).read_text(encoding="utf-8")
    elif args.lyrics_file:
        lyrics_text = args.lyrics_file
    else:
        print("compile needs a lyrics_file path or inline text", file=sys.stderr)
        return 2

    lyric_lines = [l for l in (x.strip() for x in lyrics_text.splitlines()) if l]
    from .models import LyricLine
    lyrics = Lyrics(source="text", title=args.title or audio.stem, artist=args.artist or "")
    lyrics.lines = [LyricLine(t) for t in lyric_lines]

    aligner = WhisperCppAligner(binary=Path(args.binary) if args.binary else DEFAULT_WHISPER_BIN,
                                model=Path(args.model_whisper) if args.model_whisper else DEFAULT_WHISPER_MODEL)
    timed = aligner.align(audio, lyrics)

    out = Path(args.output) if args.output else audio.with_suffix(".lrc")
    out.parent.mkdir(parents=True, exist_ok=True)
    LrcWriter().write(out, lyrics.title, lyrics.artist, args.album or "", timed)
    print(f"Wrote {out}")
    return 0


def _cmd_full(args: argparse.Namespace) -> int:
    audio = Path(args.audio)
    image = Path(args.image) if args.image else None
    # OCR available only if --image given (vision server must be reachable)
    ocr_engine = VLMOcr(api=args.api, model=args.model) if image else None
    pipe = Pipeline(
        aligner=WhisperCppAligner(
            binary=Path(args.binary) if args.binary else DEFAULT_WHISPER_BIN,
            model=Path(args.model_whisper) if args.model_whisper else DEFAULT_WHISPER_MODEL,
        ),
        ocr=ocr_engine,
    )
    result = pipe.run(audio=audio, out_dir=Path(args.out_dir), image=image,
                      write_html=not args.no_html,
                      prefer_ocr=bool(image) and not args.web_first)
    print(f"source: {result.lyrics_source}")
    print(f"lrc:   {result.lrc_path}")
    if result.html_path:
        print(f"html:  {result.html_path}")
    print(f"lines: {result.lines}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lyrics-fetcher",
                                description="Fetch and align lyrics for music files.")
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fetch", help="Fetch lyrics for a song")
    pf.add_argument("title")
    pf.add_argument("--artist", default="")
    pf.add_argument("--source", choices=list(SOURCE_FACTORY), default=None,
                    help="Use only this source")
    pf.add_argument("-v", "--verbose", action="store_true", help="Print full lyrics")
    pf.set_defaults(func=_cmd_fetch)

    po = sub.add_parser("ocr", help="OCR a booklet image to text")
    po.add_argument("image")
    po.add_argument("--engine", choices=["vlm", "tesseract"], default="vlm")
    po.add_argument("--api", default="http://127.0.0.1:8081/v1/chat/completions")
    po.add_argument("--model", default="qwen3.5-9b")
    po.add_argument("--language", default="jpn+eng")
    po.set_defaults(func=_cmd_ocr)

    pc = sub.add_parser("compile", help="Align lyrics text against an audio file")
    pc.add_argument("audio")
    pc.add_argument("lyrics_file", nargs="?", default=None,
                    help="path to a lyrics .txt/.lrc, or literal text")
    pc.add_argument("-o", "--output", default=None)
    pc.add_argument("--title", default="")
    pc.add_argument("--artist", default="")
    pc.add_argument("--album", default="")
    pc.add_argument("--binary", default=None, help="path to whisper-cli")
    pc.add_argument("--model-whisper", default=None, help="path to ggml model")
    pc.set_defaults(func=_cmd_compile)

    pfull = sub.add_parser("full", help="End-to-end: metadata+fetch/OCR+align+write")
    pfull.add_argument("audio")
    pfull.add_argument("--image", default=None, help="booklet photo for OCR fallback")
    pfull.add_argument("-o", "--out-dir", default="out")
    pfull.add_argument("--api", default="http://127.0.0.1:8081/v1/chat/completions")
    pfull.add_argument("--model", default="qwen3.5-9b")
    pfull.add_argument("--binary", default=None)
    pfull.add_argument("--model-whisper", default=None)
    pfull.add_argument("--no-html", action="store_true", help="skip HTML companion")
    pfull.add_argument("--web-first", action="store_true",
                       help="when --image given, try web fetchers before OCR "
                            "(default: OCR first, it's authoritative for the album)")
    pfull.set_defaults(func=_cmd_full)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compile" and not Path(args.audio).exists():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
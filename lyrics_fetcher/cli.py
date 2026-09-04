"""CLI entry point — `lyrics-fetcher` with subcommands.

Subcommands:
  fetch   — fetch lyrics for a song / from a source; print or save to text
  ocr     — OCR a booklet image (vlm / tesseract) to stdout
  compile — align known lyrics against an audio file -> .lrc (+ .html)
  full    — end-to-end: metadata + fetch/OCR + align + write (recommended)
  cross-check — compare whisper vs Qwen3 timings; flag drifted lines
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
from .aligner.whisper_cpp import WhisperCppAligner
from .output.writers import LrcWriter
from .cache import LyricsCache
from .manual_align import _cmd_manual
from .config import load, settings


def _make_whisper_aligner(args, default_extra_turbo: bool = False,
                          no_extra: bool = False) -> "WhisperCppAligner":
    """Resolve --binary/--model-whisper/--extra-model into a WhisperCppAligner
    (single source of truth: used by the default branch AND as the stable-ts
    fallback, so those flags reach the fallback when it engages)."""
    from .aligner.whisper_cpp import WhisperCppAligner  # late: mockable, no cost
    binary = Path(args.binary) if getattr(args, "binary", None) else settings.whisper_bin
    model = Path(args.model_whisper) if getattr(args, "model_whisper", None) else settings.whisper_model
    extra = tuple(Path(x) for x in (getattr(args, "extra_model", None) or ())) if getattr(args, "extra_model", None) else None
    if no_extra:  # cross-check: single lean model for clean comparison
        return WhisperCppAligner(binary=binary, model=model, extra_models=())
    if default_extra_turbo and (getattr(args, "extra_model", None) is None):
        return WhisperCppAligner(binary=binary, model=model, extra_models=None)
    return WhisperCppAligner(binary=binary, model=model, extra_models=extra)


def _make_aligner(args, default_extra_turbo: bool = False) -> object:
    """Build an aligner by name (--aligner {whisper,qwen3,stable-ts}).

    ``default_extra_turbo``: if True and extra models are not explicitly given,
    include the configured extra models (default large-v3-turbo).
    """
    name = getattr(args, "aligner", "whisper")
    if name == "qwen3":
        from .aligner.qwen3_forced_aligner import Qwen3ForcedAligner
        qm = getattr(args, "qwen3_model", None)
        return Qwen3ForcedAligner(model_dir=Path(qm) if qm else None)
    if name == "stable-ts":
        # lazy import: stable_whisper is an optional dev dep (like demucs)
        from .aligner.stable_ts import StableTSAligner
        # pass the whisper.cpp fallback built from the SAME CLI flags, so
        # --binary/--model-whisper/--extra-model are not lost when it engages
        # (construction is cheap: settings resolution only, no subprocess)
        return StableTSAligner(fallback=_make_whisper_aligner(args, default_extra_turbo))

    return _make_whisper_aligner(args, default_extra_turbo)

SOURCE_FACTORY = {
    "utaten": UtatenFetcher,
    "genius": GeniusFetcher,
    "silentblue": SilentBlueFetcher,
}


def _make_cache(use_cache: bool) -> LyricsCache | None:
    return LyricsCache() if use_cache else None


def _maybe_separator(sep: bool):
    """Build a VocalSeparator when ``--separation`` is requested, else None."""
    if not sep:
        return None
    from .separation import make_separator
    return make_separator()


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

    winner = next((r for r in results.values()
                   if isinstance(r, Lyrics) and r), None)
    if args.output:
        if winner is None:
            print("no source matched — nothing written", file=sys.stderr)
            return 1
        text = winner.text()
        if args.output == "-":
            print(text)
        else:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
            print(f"[saved] {len(winner.lines)} lines -> {args.output}")
        return 0
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
    ocr_engine = (VLMOcr(api=args.api, model=args.model, api_key=args.api_key)
                  if args.engine == "vlm" else TesseractOcr(lang=args.language))
    try:
        text = ocr_engine.ocr(path)
    except Exception as e:
        print(f"OCR failed: {e}", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"[saved] {len(text.splitlines())} lines -> {args.output}")
    else:
        print(text)
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

    aligner = _make_aligner(args)
    timed = aligner.align(audio, lyrics)

    out = Path(args.output) if args.output else audio.with_suffix(".lrc")
    out.parent.mkdir(parents=True, exist_ok=True)
    LrcWriter().write(out, lyrics.title, lyrics.artist, args.album or "", timed)
    print(f"Wrote {out}")
    return 0


def _cmd_full(args: argparse.Namespace) -> int:
    audio = Path(args.audio)
    image = Path(args.image) if args.image else None
    cache = _make_cache(not args.no_cache)
    # OCR available only if --image given (vision server must be reachable)
    ocr_engine = VLMOcr(api=args.api, model=args.model, api_key=args.api_key, cache=cache) if image else None
    pipe = Pipeline(
        aligner=_make_aligner(args),
        ocr=ocr_engine,
        fetcher=FetchOrchestrator(cache=cache),
        use_whisper_fallback=args.whisper_fallback,
        separator=_maybe_separator(getattr(args, "separation", False)),
    )
    result = pipe.run(audio=audio, out_dir=Path(args.out_dir), image=image,
                      write_html=not args.no_html,
                      prefer_ocr=bool(image) and not args.web_first,
                      jellyfin=args.jellyfin)
    print(f"source: {result.lyrics_source}")
    print(f"lrc:   {result.lrc_path}")
    if result.html_path:
        print(f"html:  {result.html_path}")
    print(f"lines: {result.lines}")
    return 0


def _cmd_album(args: argparse.Namespace) -> int:
    """Process every track in an album folder (batch)."""
    from .batch import batch_album

    album_dir = Path(args.album_dir)
    if not album_dir.is_dir():
        print(f"Album dir not found: {album_dir}", file=sys.stderr)
        return 2
    booklet_dir = Path(args.booklet) if args.booklet else None

    cache = _make_cache(not args.no_cache)
    # OCR engine always available for batch (multi-song page splitting needs it;
    # tracks with no booklet page simply get no OCR lyrics)
    ocr_engine = VLMOcr(api=args.api, model=args.model, api_key=args.api_key, cache=cache)
    pipe = Pipeline(
        aligner=_make_aligner(args, default_extra_turbo=True),
        ocr=ocr_engine,
        fetcher=FetchOrchestrator(cache=cache),
        use_whisper_fallback=args.whisper_fallback,
        separator=_maybe_separator(getattr(args, "separation", False)),
    )

    processed, unmatched, skipped = batch_album(
        album_dir=album_dir,
        out_dir=Path(args.out_dir) if not args.jellyfin else None,
        booklet_dir=booklet_dir,
        jellyfin=args.jellyfin,
        pipeline=pipe,
        prefer_ocr=not args.web_first,
        write_html=not args.no_html,
        verbose=not args.quiet,
        best_effort=args.whisper_fallback,
    )

    print(f"\nProcessed {len(processed)} songs -> LRC:")
    for p in processed:
        print(f"  {p}")
    if skipped:
        print(f"\n{len(skipped)} skipped (no lyrics):")
        for s in skipped:
            print(f"  {s.name}")
    if unmatched:
        print(f"\n{len(unmatched)} booklet blocks unmatched:")
        for page, reason in unmatched:
            print(f"  {page}: {reason}")
    return 0


def _cmd_crosscheck(args: argparse.Namespace) -> int:
    """Run the chosen aligners on a song and report lines they disagree on."""
    from .crosscheck import run_cross_check, format_report
    from .models import LyricLine

    audio = Path(args.audio)
    if args.lyrics_file and Path(args.lyrics_file).exists():
        lyrics_text = Path(args.lyrics_file).read_text(encoding="utf-8")
    elif args.lyrics_file:
        lyrics_text = args.lyrics_file
    else:
        print("cross-check needs a lyrics_file path or inline text", file=sys.stderr)
        return 2

    lyric_lines = [l for l in (x.strip() for x in lyrics_text.splitlines()) if l]
    if not lyric_lines:
        print("no lyric lines to compare", file=sys.stderr)
        return 2
    lyrics = Lyrics(source="text", title=args.title or audio.stem, artist=args.artist or "")
    lyrics.lines = [LyricLine(t) for t in lyric_lines]

    names = args.engines or ["whisper", "qwen3"]  # classic default pairing
    def _qwen3():
        from .aligner.qwen3_forced_aligner import Qwen3ForcedAligner
        return Qwen3ForcedAligner(
            model_dir=Path(args.qwen3_model) if args.qwen3_model else None)

    def _stable():
        from .aligner.stable_ts import StableTSAligner
        return StableTSAligner(fallback=_make_whisper_aligner(args))

    builders = {
        "whisper": lambda: _make_whisper_aligner(args, no_extra=True),
        "qwen3": _qwen3,
        "stable-ts": _stable,
    }
    engines = [(n, builders[n]()) for n in names]
    report = run_cross_check(engines, audio, lyrics, tolerance=args.tolerance)
    print(format_report(report, verbose=args.verbose))
    # non-zero exit when anything needs attention so scripts can react
    return 1 if (report.drifted or report.missing) else 0


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
    pf.add_argument("-o", "--output", default=None,
                    help="write the accepted lyrics text to this .txt file "
                         "(use '-' for stdout; default: status lines only)")
    pf.add_argument("--config", default=None, help="path to a config TOML file")
    pf.set_defaults(func=_cmd_fetch)

    po = sub.add_parser("ocr", help="OCR a booklet image to text")
    po.add_argument("image")
    po.add_argument("--engine", choices=["vlm", "tesseract"], default="vlm")
    po.add_argument("--api", default=None, help="chat-completions URL (default: [vision] config)")
    po.add_argument("--model", default=None, help="vision model name (default: [vision] config)")
    po.add_argument("--api-key", default=None, help="bearer token (default: VISION_API_KEY env)")
    po.add_argument("--language", default="jpn+eng")
    po.add_argument("-o", "--output", default=None,
                    help="write transcription to this .txt file "
                         "(default: stdout; keeps -v-style raw output too)")
    po.add_argument("--config", default=None, help="path to a config TOML file")
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
    pc.add_argument("--aligner", choices=["whisper", "qwen3", "stable-ts"], default="whisper",
                    help="alignment engine: whisper (cpp), qwen3 (ForcedAligner), or stable-ts (forced alignment, dev extra; falls back to whisper on failure)")
    pc.add_argument("--qwen3-model", default=None, help="path to Qwen3-ForcedAligner model dir")
    pc.add_argument("--config", default=None, help="path to a config TOML file")
    pc.set_defaults(func=_cmd_compile)

    px = sub.add_parser("cross-check",
                        help="Compare whisper vs Qwen3 timings; flag drifted lines")
    px.add_argument("audio")
    px.add_argument("lyrics_file", nargs="?", default=None,
                    help="path to a lyrics .txt/.lrc, or literal text")
    px.add_argument("--engines", nargs="+", choices=["whisper", "qwen3", "stable-ts"],
                    default=None, metavar="ENGINE",
                    help="aligners to compare, 2+ recommended "
                         "(default: whisper qwen3)")
    px.add_argument("--tolerance", type=float, default=2.5,
                    help="spread (max-min seconds) above which a line counts as "
                         "drifted (default 2.5)")
    px.add_argument("--title", default="")
    px.add_argument("--artist", default="")
    px.add_argument("--binary", default=None, help="path to whisper-cli")
    px.add_argument("--model-whisper", default=None, help="path to ggml model")
    px.add_argument("--qwen3-model", default=None, help="path to Qwen3-ForcedAligner model dir")
    px.add_argument("-v", "--verbose", action="store_true",
                    help="show every line (default: drifted/missing lines only)")
    px.add_argument("--config", default=None, help="path to a config TOML file")
    px.set_defaults(func=_cmd_crosscheck)

    pfull = sub.add_parser("full", help="End-to-end: metadata+fetch/OCR+align+write")
    pfull.add_argument("audio")
    pfull.add_argument("--image", default=None, help="booklet photo for OCR fallback")
    pfull.add_argument("-o", "--out-dir", default="out")
    pfull.add_argument("--api", default=None, help="chat-completions URL (default: [vision] config)")
    pfull.add_argument("--model", default=None, help="vision model name (default: [vision] config)")
    pfull.add_argument("--api-key", default=None, help="bearer token (default: VISION_API_KEY env)")
    pfull.add_argument("--binary", default=None)
    pfull.add_argument("--model-whisper", default=None)
    pfull.add_argument("--extra-model", action="append", default=None,
                       help="extra whisper model(s) to transcribe with for anchor "
                            "merging; repeatable. Helps hallucinating songs.")
    pfull.add_argument("--aligner", choices=["whisper", "qwen3", "stable-ts"], default="whisper",
                       help="alignment engine: whisper (cpp), qwen3 (ForcedAligner), or stable-ts (forced alignment, dev extra; falls back to whisper on failure)")
    pfull.add_argument("--qwen3-model", default=None, help="path to Qwen3-ForcedAligner model dir")
    pfull.add_argument("--config", default=None, help="path to a config TOML file")
    pfull.add_argument("--no-html", action="store_true", help="skip HTML companion")
    pfull.add_argument("--jellyfin", action="store_true",
                       help="write .lrc next to the audio file (Jellyfin layout)")
    pfull.add_argument("--no-cache", action="store_true", help="disable SQLite cache")
    pfull.add_argument("--web-first", action="store_true",
                       help="when --image given, try web fetchers before OCR "
                            "(default: OCR first, it's authoritative for the album)")
    pfull.add_argument("--whisper-fallback", action="store_true",
                       help="best-effort: whisper-transcribe if no lyrics found "
                            "(default: report no lyrics; whisper is poor at Japanese)")
    pfull.add_argument("--separation", action="store_true",
                       help="separate a dry-vocal stem (demucs) before aligning — "
                            "improves intro timing on BGM-dense songs; slower")
    pfull.set_defaults(func=_cmd_full)

    pal = sub.add_parser("album", help="Batch-process every track in an album folder")
    pal.add_argument("album_dir")
    pal.add_argument("--booklet", default=None,
                     help="path to a booklet image folder (default: <album>/booklet)")
    pal.add_argument("-o", "--out-dir", default="out")
    pal.add_argument("--api", default=None, help="chat-completions URL (default: [vision] config)")
    pal.add_argument("--model", default=None, help="vision model name (default: [vision] config)")
    pal.add_argument("--api-key", default=None, help="bearer token (default: VISION_API_KEY env)")
    pal.add_argument("--binary", default=None)
    pal.add_argument("--model-whisper", default=None)
    pal.add_argument("--extra-model", action="append", default=None,
                     help="extra whisper model(s); default large-v3-turbo")
    pal.add_argument("--aligner", choices=["whisper", "qwen3", "stable-ts"], default="whisper",
                     help="alignment engine: whisper (cpp), qwen3 (ForcedAligner), or stable-ts (forced alignment, dev extra; falls back to whisper on failure)")
    pal.add_argument("--qwen3-model", default=None, help="path to Qwen3-ForcedAligner model dir")
    pal.add_argument("--config", default=None, help="path to a config TOML file")
    pal.add_argument("--no-html", action="store_true")
    pal.add_argument("--jellyfin", action="store_true",
                     help="write .lrc next to each audio file (Jellyfin layout)")
    pal.add_argument("--no-cache", action="store_true", help="disable SQLite cache")
    pal.add_argument("--whisper-fallback", action="store_true",
                     help="best-effort: whisper-transcribe tracks with no lyrics "
                          "(default: skip them; whisper is poor at Japanese singing)")
    pal.add_argument("--web-first", action="store_true",
                     help="try web fetchers before OCR for each track")
    pal.add_argument("--separation", action="store_true",
                     help="separate a dry-vocal stem (demucs) before aligning each "
                          "track — improves intro timing on BGM-dense songs; slower")
    pal.add_argument("-q", "--quiet", action="store_true")
    pal.set_defaults(func=_cmd_album)

    pman = sub.add_parser("manual", help="Manually time lyrics by tapping a key per line")
    pman.add_argument("audio")
    pman.add_argument("lyrics", help="path to lyrics .txt")
    pman.add_argument("-o", "--output", default=None)
    pman.add_argument("--title", default="")
    pman.add_argument("--artist", default="")
    pman.add_argument("--album", default="")
    pman.add_argument("--config", default=None, help="path to a config TOML file")
    pman.set_defaults(func=_cmd_manual)
    return p


def main(argv: list[str] | None = None) -> int:
    # load config (auto-detect) before any command runs so settings are available
    load()
    parser = build_parser()
    args = parser.parse_args(argv)
    # if --config was given, (re)load explicitly from it
    cfg = getattr(args, "config", None)
    if cfg:
        load(Path(cfg))
    if args.command == "compile" and not Path(args.audio).exists():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
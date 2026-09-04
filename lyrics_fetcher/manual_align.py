"""Manual aligner — tap a key when each lyric line begins.

Plays the audio (ffplay, headless) and you press RETURN each time a new lyric
line begins; the tool records wall-clock elapsed time relative to the moment
playback started and assigns each line its start timestamp. Writes a .lrc.

This is the fallback for songs that defeat automatic alignment
letting you provide the line start times by ear.

Usage:
    lyrics-fetcher manual <audio> <lyrics.txt> -o out.lrc

Controls (after "GO"):
    RETURN / SPACE  — mark current playback moment as the start of the NEXT line
    b               — go back one line (re-mark it)
    s               — skip the current line (set 0.0)
    q               — quit and write the .lrc so far
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from .aligner.base import TimedLine
from .output.writers import LrcWriter


def read_lyrics(path: Path) -> list[str]:
    return [l for l in (x.strip() for x in path.read_text(encoding="utf-8").splitlines()) if l]


def manual_align(audio: Path, lines: list[str], countdown: float = 3.0) -> list[TimedLine]:
    """Interactive keyboard-tap alignment. Returns list[TimedLine].

    In a non-interactive context (no TTY) it returns all-0 timestamps rather
    than hanging on input().
    """
    if not sys.stdin.isatty():
        return [TimedLine(l, 0.0) for l in lines]

    times: list[float | None] = [None] * len(lines)
    idx = 0  # next line to mark

    print("\nPlayback starting. When each line begins, press RETURN.", file=sys.stderr)
    print(f"Reading lyrics: {len(lines)} lines. Keys: b=back s=skip q=quit\n", file=sys.stderr)
    for i in range(int(countdown), 0, -1):
        print(f"  {i}...", file=sys.stderr)
        time.sleep(1)
    print("  GO  (press RETURN on each line start)\n", file=sys.stderr)

    start_wall = time.monotonic()
    proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-i", str(audio)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )

    def elapsed() -> float:
        return time.monotonic() - start_wall

    def show_line():
        cur = lines[idx] if idx < len(lines) else "(done)"
        print(f"\r  [{idx}/{len(lines)}] t={elapsed():6.1f}s | NEXT: {cur[:34]}",
              file=sys.stderr, end="", flush=True)

    show_line()
    try:
        while idx < len(lines) and proc.poll() is None:
            k = (input() or " ").strip().lower()
            t = elapsed()
            if k == "q":
                break
            elif k == "b":
                if idx > 0:
                    idx -= 1
                    times[idx] = None
                show_line()
                continue
            elif k == "s":
                times[idx] = 0.0
                idx += 1
                if idx < len(lines):
                    show_line()
                continue
            # normal: mark current line's start at playback position
            times[idx] = t
            idx += 1
            if idx < len(lines):
                show_line()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        proc.terminate()

    # fill unmarked lines with the previous marked time (or the last known time)
    timed: list[TimedLine] = []
    prev = 0.0
    for i, l in enumerate(lines):
        t = times[i]
        t = t if t is not None else prev
        prev = t
        timed.append(TimedLine(l, max(t, 0.0)))
    return timed


def _cmd_manual(args) -> int:
    audio = Path(args.audio)
    lines = read_lyrics(Path(args.lyrics))
    if not lines:
        print("No lyric lines in the lyrics file", file=sys.stderr)
        return 2

    timed = manual_align(audio, lines)
    out = Path(args.output) if args.output else audio.with_suffix(".lrc")
    out.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or audio.stem
    LrcWriter().write(out, title, args.artist, args.album or "", timed)
    print(f"\nWrote {out}")
    print("Sample:")
    for tl in timed[:5]:
        m, s = divmod(tl.start, 60)
        print(f"  [{int(m):02d}:{s:05.2f}] {tl.text}")
    return 0
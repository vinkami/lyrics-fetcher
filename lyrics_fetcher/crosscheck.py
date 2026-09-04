"""Cross-check mode — compare two alignment engines on the same song.

The two aligners (whisper.cpp and Qwen3-ForcedAligner) are fully independent
timing sources. Running both on one song and comparing per-line start times
lets the user spot lines where the automatic alignment drifts from the true
timing (the original goal of this tool) and hand-fix ONLY those via ``manual``,
instead of trusting a single engine blindly.

Design notes:
- Both aligners consume the SAME ``Lyrics`` and map every lyric line (by index)
  to a ``TimedLine``, so we compare line ``i``'s ``start`` across engines.
- ``delta`` = whisper start - qwen3 start. ``|delta| > tolerance`` => "drift".
- If an engine returns FEWER lines than another, the missing tail lines are
  reported as "missing" from that source rather than fabricating a delta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import Lyrics
from .aligner.base import BaseAligner, TimedLine


@dataclass
class CrossCheckLine:
    """Per-line cross-check result: both engines' starts, delta, and status."""

    text: str
    # None => that engine produced no timing for this line (truncated output)
    whisper_start: float | None = None
    qwen3_start: float | None = None
    delta: float | None = None
    status: str = "ok"  # "ok" | "drift" | "missing"


@dataclass
class CrossCheckReport:
    """Aggregate cross-check result for a song."""

    lines: list[CrossCheckLine] = field(default_factory=list)
    tolerance: float = 2.5
    # engine names actually run (in case one source was skipped/errored)
    engines: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def drifted(self) -> int:
        return sum(1 for l in self.lines if l.status == "drift")

    @property
    def missing(self) -> int:
        return sum(1 for l in self.lines if l.status == "missing")

    @property
    def total(self) -> int:
        return len(self.lines)


def run_cross_check(
    engines: list[tuple[str, BaseAligner]],
    audio: Path,
    lyrics: Lyrics,
    tolerance: float = 2.5,
) -> CrossCheckReport:
    """Run ``engines`` ([(name, aligner), ...]) and compare their outputs.

    Returns a report pairing line i's start across all engines. ``tolerance``
    is the |seconds| threshold above which a line is flagged as "drift".
    """
    report = CrossCheckReport(tolerance=tolerance)
    outputs: dict[str, list[TimedLine]] = {}
    for name, aligner in engines:
        report.engines.append(name)
        try:
            outputs[name] = aligner.align(audio, lyrics)
        except Exception as e:  # noqa: BLE001 - decode/IO failures shouldn't abort
            report.errors[name] = str(e)

    # Sources must be comparable: all aligners share the same lyric order, so we
    # need at least two successful engines to diff anything.
    ok = [n for n in report.engines if n in outputs]
    if len(ok) < 2:
        # fewer than two engines ran (or they errored) -> nothing to diff.
        # Mark every line as missing so the user sees the gap rather than a
        # silently empty report.
        report.lines = [CrossCheckLine(l.text, status="missing") for l in lyrics.lines]
        return report

    # Canonical pairing with known engine names ("whisper" / "qwen3"). If the
    # caller named engines differently, fall back to first-two positional.
    wname = "whisper" if "whisper" in ok else ok[0]
    qname = "qwen3" if "qwen3" in ok else next((n for n in ok if n != wname), None)

    for i, l in enumerate(lyrics.lines):
        ws = _get_start(outputs[wname], i)
        qs = _get_start(outputs[qname], i) if qname else None
        if ws is None or qs is None:
            status = "missing"
            delta = None
        else:
            delta = ws - qs
            status = "drift" if abs(delta) > tolerance else "ok"
        report.lines.append(
            CrossCheckLine(
                text=l.text,
                whisper_start=ws,
                qwen3_start=qs,
                delta=delta,
                status=status,
            )
        )
    return report


def _get_start(timed: list[TimedLine], i: int) -> float | None:
    """Return line i's start time, or None if the engine didn't time it."""
    if i < len(timed):
        return float(timed[i].start)
    return None


def _fmt(t: float | None) -> str:
    return f"{t:6.1f}s" if t is not None else "    —  "


def format_report(report: CrossCheckReport, verbose: bool = False) -> str:
    """Render a cross-check report as human-readable text."""
    out: list[str] = []
    engines = " vs ".join(report.engines) or "n/a"
    out.append(f"Cross-check: {engines} (tolerance |Δ| > {report.tolerance:.1f}s)")

    if report.errors:
        for name, err in report.errors.items():
            out.append(f"  [error] {name}: {err}")
            out.append(f"  [error] {name} produced no timing — lines 'missing'")

    shown = [l for l in report.lines if l.status != "ok"] if not verbose else report.lines
    for l in shown:
        tag = {"drift": "DRIFT", "missing": "MISSING", "ok": ""}[l.status]
        delta = f"Δ={l.delta:+6.1f}s" if l.delta is not None else f"Δ=    --"
        out.append(
            f"  {l.text!r:<40} whisper={_fmt(l.whisper_start)} "
            f"qwen3={_fmt(l.qwen3_start)} {delta:>10} {tag}"
        )
    if not shown:
        out.append("  (no drifted lines)")

    tallies = [f"{report.total} lines"]
    if report.engines and len(report.engines) >= 2:
        tallies.append(f"{report.drifted} drifted")
    if report.missing:
        tallies.append(f"{report.missing} missing")
    out.append("Summary: " + ", ".join(tallies))
    return "\n".join(out)
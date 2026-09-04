"""Cross-check mode — compare ANY two-or-more alignment engines on one song.

The aligners (whisper.cpp, stable-ts, Qwen3-ForcedAligner) are independent
timing sources. Running several on one song and comparing per-line start
times lets the user spot lines where alignment drifts from the true timing
and hand-fix ONLY those via ``manual``, instead of trusting one engine.

Design notes:
- Every aligner consumes the SAME ``Lyrics`` and maps line ``i`` (by index)
  to a ``TimedLine``, so we diff line i's ``start`` across all engines.
- ``delta`` = max(starts) - min(starts) across engines; ``delta > tolerance``
  => "drift". With 2 engines this is |a-b| as before.
- A line any engine lacks => "missing" (no fabricated delta).
- An engine that ERRORS is reported in ``report.errors`` and excluded from
  the diff; with 3 engines selected you still get a 2-engine comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import Lyrics
from .aligner.base import BaseAligner, TimedLine


@dataclass
class CrossCheckLine:
    """Per-line cross-check result: every engine's start, spread, status."""

    text: str
    # engine name -> start time; an absent key => that engine didn't time
    # this line (truncated output) -> reported via `missing_from`
    starts: dict[str, float] = field(default_factory=dict)
    # max - min across present engines (None if <2 timed this line)
    delta: float | None = None
    status: str = "ok"  # "ok" | "drift" | "missing"

    @property
    def missing_from(self) -> list[str]:
        """Engines that have output for this song but skipped this line."""
        return self._missing_from or []

    _missing_from: list[str] = field(default_factory=list, repr=False)


@dataclass
class CrossCheckReport:
    """Aggregate cross-check result for a song."""

    lines: list[CrossCheckLine] = field(default_factory=list)
    tolerance: float = 2.5
    # engine names run, in call order (errored ones stay listed; see errors)
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

    # Spread across engines: line drifts when max(start) - min(start) > tol.
    # With the classic 2 engines this equals |whisper - qwen3| as before.
    # An engine whose whole run errored is absent from `outputs` -> it is
    # listed per-line in missing_from only if at least one engine timed the
    # line; engines with no successful output at all don't mark lines missing.
    for i, l in enumerate(lyrics.lines):
        starts: dict[str, float] = {}
        for name in ok:
            t = _get_start(outputs[name], i)
            if t is not None:
                starts[name] = t
        vals = list(starts.values())
        if len(vals) < len(ok):
            # some engine(s) ran successfully but stopped before this line
            status = "missing"
            delta = None
            missing_from = [n for n in ok if n not in starts]
        else:
            delta = (max(vals) - min(vals)) if len(vals) >= 2 else None
            status = "drift" if delta is not None and delta > tolerance else "ok"
            missing_from = []
        report.lines.append(
            CrossCheckLine(text=l.text, starts=starts, delta=delta,
                           status=status, _missing_from=missing_from)
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
    out.append(f"Cross-check: {engines} (tolerance spread > {report.tolerance:.1f}s)")

    if report.errors:
        for name, err in report.errors.items():
            out.append(f"  [error] {name}: {err}")
            out.append(f"  [error] {name} excluded from the diff")

    shown = [l for l in report.lines if l.status != "ok"] if not verbose else report.lines
    width = max((len(n) for n in report.engines), default=6) + 1
    for l in shown:
        tag = {"drift": "DRIFT", "missing": "MISSING", "ok": ""}[l.status]
        delta = f"spread={l.delta:5.1f}s" if l.delta is not None else "spread=   --"
        cols = " ".join(
            f"{n + '=':<{width}}{_fmt(l.starts.get(n))}" for n in report.engines
            if n not in report.errors
        )
        extra = f" missing:{','.join(l.missing_from)}" if l.missing_from else ""
        out.append(f"  {l.text!r:<40} {cols} {delta}{extra} {tag}")
    if not shown:
        out.append("  (no drifted lines)")

    tallies = [f"{report.total} lines"]
    if report.engines and len(report.engines) >= 2:
        tallies.append(f"{report.drifted} drifted")
    if report.missing:
        tallies.append(f"{report.missing} missing")
    out.append("Summary: " + ", ".join(tallies))
    return "\n".join(out)
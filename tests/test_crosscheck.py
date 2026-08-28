"""Tests for cross-check mode — comparing two aligners' timings.

Uses SYNTHETIC placeholder lyric text and FAKE aligners that return fixed
timings. No GPU, no whisper, no Qwen3, no real lyrics — pure logic only.
"""
from pathlib import Path

from lyrics_fetcher.models import Lyrics, LyricLine
from lyrics_fetcher.aligner.base import BaseAligner, TimedLine
from lyrics_fetcher.crosscheck import (
    CrossCheckLine,
    CrossCheckReport,
    format_report,
    run_cross_check,
)


class FakeAligner(BaseAligner):
    """Returns a fixed list of TimedLine starts."""

    name = "fake"

    def __init__(self, name, starts):
        self.name = name
        self.starts = list(starts)

    def align(self, audio: Path, lyrics: Lyrics) -> list[TimedLine]:
        return [
            TimedLine(text=l.text, start=st)
            for l, st in zip(lyrics.lines, self.starts)
        ]


def _lyrics(lines):
    ly = Lyrics(source="text", title="t", artist="a")
    ly.lines = [LyricLine(t) for t in lines]
    return ly


def _run(whisper_starts, qwen3_starts, lines=None, tolerance=2.5, extra=None):
    lines = lines or [f"line {i}" for i in range(len(whisper_starts))]
    engines = [("whisper", FakeAligner("whisper", whisper_starts))]
    engines += [("qwen3", FakeAligner("qwen3", qwen3_starts))]
    if extra:
        engines += extra
    return run_cross_check(engines, Path("song.flac"), _lyrics(lines), tolerance)


# ---- core comparison ----
def test_identical_timings_all_ok():
    r = _run([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert r.drifted == 0
    assert r.total == 3
    assert all(l.status == "ok" for l in r.lines)
    assert all(l.delta == 0.0 for l in r.lines)


def test_drift_detected_when_exceeds_tolerance():
    # line 1: whisper 20s vs qwen3 28s => delta -8, over 2.5 => drift
    r = _run([10.0, 20.0, 30.0], [10.0, 28.0, 30.0], tolerance=2.5)
    assert r.drifted == 1
    assert [l.status for l in r.lines] == ["ok", "drift", "ok"]
    assert r.lines[1].delta == -8.0


def test_drift_respects_tolerance_boundary():
    # delta exactly == tolerance is NOT drift (strict >)
    r = _run([10.0], [10.0 + 2.5], tolerance=2.5)
    assert r.drifted == 0
    # just over the boundary IS drift
    r2 = _run([10.0], [10.0 + 2.5001], tolerance=2.5)
    assert r2.drifted == 1


def test_shorter_engine_yields_missing_lines():
    # qwen3 only timed the first 2 of 3 lines -> line 2 is "missing"
    r = _run([10.0, 20.0, 30.0], [10.0, 20.0], lines=["a", "b", "c"])
    assert r.missing == 1
    assert r.lines[2].status == "missing"
    assert r.lines[2].qwen3_start is None
    assert r.lines[2].delta is None
    # the timed lines still compare fine
    assert r.lines[0].status == "ok"
    assert r.lines[1].status == "ok"


def test_empty_lyrics_produces_empty_report():
    r = _run([], [])
    assert r.total == 0
    assert r.drifted == 0


# ---- resilience ----
def test_erroring_engine_marked_missing():
    class Boom(FakeAligner):
        def align(self, audio, lyrics):
            raise RuntimeError("boom")

    lines = _lyrics(["a", "b"])
    engines = [("whisper", FakeAligner("whisper", [1.0, 2.0])),
               ("qwen3", Boom("qwen3", []))]
    ck = run_cross_check(engines, Path("x.flac"), lines, 2.5)
    assert "qwen3" in ck.errors
    # fewer than two successful engines -> everything missing, clear gap
    assert ck.drifted == 0
    assert all(l.status == "missing" for l in ck.lines)


def test_single_engine_gives_no_false_drift():
    lines = _lyrics(["a", "b"])
    engines = [("whisper", FakeAligner("whisper", [1.0, 2.0]))]
    ck = run_cross_check(engines, Path("x.flac"), lines, 2.5)
    assert len(ck.engines) == 1
    assert all(l.status == "missing" for l in ck.lines)
    assert ck.drifted == 0


def test_engines_named_differently_still_pair():
    # if the caller uses non-canonical names, pairwise timing still works
    class A1(BaseAligner):
        name = "engine_a"
        def align(self, audio, lyrics):
            return [TimedLine(l.text, 10.0 + i) for i, l in enumerate(lyrics.lines)]

    class A2(BaseAligner):
        name = "engine_b"
        def align(self, audio, lyrics):
            return [TimedLine(l.text, 50.0 + i) for i, l in enumerate(lyrics.lines)]

    ck = run_cross_check(
        [("alpha", A1()), ("beta", A2())], Path("x.flac"),
        _lyrics(["a", "b", "c"]), tolerance=2.5,
    )
    assert ck.drifted == 3  # every line wildly different


# ---- formatting ----
def test_format_report_mentions_drift_and_summary():
    r = _run([10.0, 20.0], [10.0, 30.0], lines=["hello world", "second verse"])
    s = format_report(r, verbose=False)
    assert "Cross-check:" in s
    assert "DRIFT" in s
    assert "2 lines" in s
    assert "1 drifted" in s


def test_format_report_verbose_shows_all():
    r = _run([10.0, 20.0], [10.0, 20.0], lines=["a", "b"])
    s = format_report(r, verbose=True)
    # ok lines shown in verbose mode
    assert "a" in s and "b" in s
    assert "no drifted lines" not in s  # not collapsed


def test_format_report_no_drift_collapsed_without_verbose():
    r = _run([10.0, 20.0], [10.0, 20.0], lines=["a", "b"])
    s = format_report(r, verbose=False)
    assert "(no drifted lines)" in s


def test_delta_sign_convention_whisper_minus_qwen():
    # delta > 0 => whisper start is LATER than qwen3
    r = _run([25.0], [20.0], lines=["x"])
    assert r.lines[0].delta == 5.0
    # delta < 0 => whisper is EARLIER
    r2 = _run([20.0], [25.0], lines=["x"])
    assert r2.lines[0].delta == -5.0


# ---- CLI wiring (no GPU: aligner classes replaced with fakes) ----
def test_crosscheck_cli_reads_file_runs_both_and_exit_codes(
    monkeypatch, capsys, tmp_path
):
    from lyrics_fetcher import cli

    class FakeWhisper(BaseAligner):
        name = "whisper"

        def __init__(self, *a, **k):
            pass

        def align(self, audio, lyrics):
            return [TimedLine(l.text, 10.0 + i) for i, l in enumerate(lyrics.lines)]

    class FakeQwen(BaseAligner):
        name = "qwen3"

        def __init__(self, *a, **k):
            pass

        def align(self, audio, lyrics):
            # every line drifted vs whisper
            return [TimedLine(l.text, 50.0 + i) for i, l in enumerate(lyrics.lines)]

    monkeypatch.setattr(
        "lyrics_fetcher.aligner.whisper_cpp.WhisperCppAligner", FakeWhisper
    )
    monkeypatch.setattr(
        "lyrics_fetcher.aligner.qwen3_forced_aligner.Qwen3ForcedAligner", FakeQwen
    )

    lyr = tmp_path / "lyrics.txt"
    lyr.write_text("first line\nsecond verse\n\nthird line\n", encoding="utf-8")

    rc = cli.main(["cross-check", "song.flac", str(lyr), "--tolerance", "2.5"])
    out = capsys.readouterr().out
    # both engines ran, lines drifted -> exit non-zero
    assert rc == 1
    assert "Cross-check: whisper vs qwen3" in out
    assert "3 lines" in out
    assert "3 drifted" in out


def test_crosscheck_cli_exit_zero_when_agree(monkeypatch, capsys, tmp_path):
    from lyrics_fetcher import cli

    class FakeWhisper(BaseAligner):
        name = "whisper"

        def __init__(self, *a, **k):
            pass

        def align(self, audio, lyrics):
            return [TimedLine(l.text, 10.0 + i) for i, l in enumerate(lyrics.lines)]

    class FakeQwen(BaseAligner):
        name = "qwen3"

        def __init__(self, *a, **k):
            pass

        def align(self, audio, lyrics):
            return [TimedLine(l.text, 10.0 + i) for i, l in enumerate(lyrics.lines)]

    monkeypatch.setattr(
        "lyrics_fetcher.aligner.whisper_cpp.WhisperCppAligner", FakeWhisper
    )
    monkeypatch.setattr(
        "lyrics_fetcher.aligner.qwen3_forced_aligner.Qwen3ForcedAligner", FakeQwen
    )

    lyr = tmp_path / "lyrics.txt"
    lyr.write_text("only line\n", encoding="utf-8")
    rc = cli.main(["cross-check", "song.flac", str(lyr)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no drifted lines)" in out
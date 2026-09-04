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
    # line 1: whisper 20s vs qwen3 28s => spread 8, over 2.5 => drift
    r = _run([10.0, 20.0, 30.0], [10.0, 28.0, 30.0], tolerance=2.5)
    assert r.drifted == 1
    assert [l.status for l in r.lines] == ["ok", "drift", "ok"]
    assert r.lines[1].delta == 8.0  # spread is unsigned
    assert r.lines[1].starts == {"whisper": 20.0, "qwen3": 28.0}


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
    assert "qwen3" not in r.lines[2].starts
    assert r.lines[2].missing_from == ["qwen3"]
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


def test_delta_is_unsigned_spread_either_direction():
    # spread = max - min, independent of engine order
    r = _run([25.0], [20.0], lines=["x"])
    assert r.lines[0].delta == 5.0
    r2 = _run([20.0], [25.0], lines=["x"])
    assert r2.lines[0].delta == 5.0


# ---- multi-engine (the point of --engines) ----
def test_three_engine_spread_flags_worst_pair():
    lines = ["a", "b"]
    engines = [
        ("whisper", FakeAligner("whisper", [10.0, 20.0])),
        ("qwen3", FakeAligner("qwen3", [11.0, 20.5])),
        ("stable-ts", FakeAligner("stable-ts", [10.5, 35.0])),
    ]
    r = run_cross_check(engines, Path("x.flac"), _lyrics(lines), 2.5)
    assert r.engines == ["whisper", "qwen3", "stable-ts"]
    # line 0: spread 10..11 = 1.0 ok; line 1: 20..35 = 15 => drift
    assert [l.status for l in r.lines] == ["ok", "drift"]
    assert r.lines[1].delta == 15.0
    assert r.lines[1].starts == {"whisper": 20.0, "qwen3": 20.5, "stable-ts": 35.0}


def test_third_engine_error_still_compares_the_other_two():
    class Boom(FakeAligner):
        def align(self, audio, lyrics):
            raise RuntimeError("model OOM")

    engines = [
        ("whisper", FakeAligner("whisper", [10.0, 20.0])),
        ("stable-ts", FakeAligner("stable-ts", [10.2, 30.0])),
        ("qwen3", Boom("qwen3", [])),
    ]
    r = run_cross_check(engines, Path("x.flac"), _lyrics(["a", "b"]), 2.5)
    assert "qwen3" in r.errors
    assert r.drifted == 1               # whisper vs stable-ts still diffed
    assert r.lines[1].starts == {"whisper": 20.0, "stable-ts": 30.0}
    assert r.missing == 0               # errored engine doesn't mark missing


def test_format_report_shows_all_engine_columns():
    lines = ["a"]
    engines = [
        ("whisper", FakeAligner("whisper", [10.0])),
        ("stable-ts", FakeAligner("stable-ts", [20.0])),
    ]
    r = run_cross_check(engines, Path("x.flac"), _lyrics(lines), 2.5)
    s = format_report(r)
    assert "whisper vs stable-ts" in s
    assert "whisper=" in s and "stable-ts=" in s and "spread=" in s


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

def test_crosscheck_cli_engines_selection(monkeypatch, capsys, tmp_path):
    """--engines whisper stable-ts runs exactly those two builders."""
    from lyrics_fetcher import cli

    ran = []

    class FakeAligner2(BaseAligner):
        def __init__(self, engine, *a, **k):
            self.engine = engine
            self.name = engine

        def align(self, audio, lyrics):
            ran.append(self.name)
            step = 10.0 if self.name == "whisper" else 10.5
            return [TimedLine(l.text, step + i) for i, l in enumerate(lyrics.lines)]

    monkeypatch.setattr(
        "lyrics_fetcher.aligner.whisper_cpp.WhisperCppAligner",
        lambda *a, **k: FakeAligner2("whisper"))
    monkeypatch.setattr(
        "lyrics_fetcher.aligner.stable_ts.StableTSAligner",
        lambda *a, **k: FakeAligner2("stable-ts"))

    lyr = tmp_path / "lyrics.txt"
    lyr.write_text("a\nb\n", encoding="utf-8")
    rc = cli.main(["cross-check", "song.flac", str(lyr),
                   "--engines", "whisper", "stable-ts"])
    out = capsys.readouterr().out
    assert sorted(ran) == ["stable-ts", "whisper"]
    assert "qwen3" not in "".join(ran)
    assert "whisper vs stable-ts" in out
    assert rc == 0  # spreads of 0.5 within tolerance


def test_crosscheck_cli_single_engine_errors_early(monkeypatch, capsys, tmp_path):
    # run_cross_check needs >=2 successful engines; a lone engine must still
    # produce a (all-missing) report with non-zero exit, not crash
    from lyrics_fetcher import cli

    class FakeW(BaseAligner):
        name = "whisper"
        def __init__(self, *a, **k):
            pass
        def align(self, audio, lyrics):
            return [TimedLine(l.text, 5.0) for l in lyrics.lines]

    monkeypatch.setattr(
        "lyrics_fetcher.aligner.whisper_cpp.WhisperCppAligner", FakeW)
    lyr = tmp_path / "l.txt"
    lyr.write_text("a\n", encoding="utf-8")
    rc = cli.main(["cross-check", "song.flac", str(lyr), "--engines", "whisper"])
    assert rc == 1

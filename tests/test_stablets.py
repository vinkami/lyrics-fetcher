"""Tests for the stable-ts aligner (opt-in forced alignment).

stable_whisper is an optional DEV dependency (not installed in CI), and these
tests never touch it, the GPU, or a model download: the per-line timing logic
runs on FAKE stable-ts result objects (segments -> words with .start/.word),
and error paths use stubbed _load/model. Same fake-injection pattern as
tests/test_aligner.py and tests/test_separation.py.
"""
from __future__ import annotations

import argparse
import builtins

import pytest

from lyrics_fetcher.aligner.base import BaseAligner, TimedLine
from lyrics_fetcher.aligner.stable_ts import StableTSAligner
from lyrics_fetcher.config import settings
from lyrics_fetcher.models import LyricLine, Lyrics


# ---- fake stable-ts result objects ----
class FakeWord:
    def __init__(self, start, word):
        self.start = start
        self.word = word


class FakeSeg:
    def __init__(self, words):
        # words: list[(start, text)] or None (real stable-ts can have seg.words=None)
        self.words = [FakeWord(s, w) for s, w in words] if words is not None else None


class FakeResult:
    def __init__(self, segments):
        self.segments = [FakeSeg(w) for w in segments]


def lyrics_of(*lines):
    ly = Lyrics(source="test", title="t", artist="a")
    ly.lines = [LyricLine(t) for t in lines]
    return ly


class FakeFallback(BaseAligner):
    name = "fake-fallback"

    def __init__(self):
        self.calls = []

    def align(self, audio, lyrics):
        self.calls.append((audio, lyrics))
        return [TimedLine(text=l.text, start=99.0) for l in lyrics.lines]


# ---- lazy-import contract: module must work without stable_whisper ----
def test_construct_and_align_without_stable_whisper(monkeypatch, tmp_path):
    """Construction and the fallback path never import stable_whisper, so the
    CLI/CI works even with the dev extra absent."""
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "stable_whisper" or name.startswith("stable_whisper."):
            raise ImportError("stable_whisper not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    al = StableTSAligner()  # must not touch the lib
    assert al.name == "stable-ts"
    assert al.fallback is None  # whisper.cpp fallback built lazily, only on need
    with pytest.raises(ImportError):
        al._load()  # the ONLY place the lib is imported
    fb = FakeFallback()
    al2 = StableTSAligner(fallback=fb)
    monkeypatch.setattr(al2, "_load", lambda: (_ for _ in ()).throw(ImportError("nope")))
    out = al2.align(tmp_path / "s.wav", lyrics_of("line one"))
    assert fb.calls and out[0].start == 99.0


# ---- settings resolution at construction (so --config works) ----
def test_settings_resolution(monkeypatch):
    monkeypatch.setattr(settings, "stable_ts_model", "small")
    monkeypatch.setattr(settings, "stable_ts_lang", "en")
    monkeypatch.setattr(settings, "stable_ts_device", "cpu")
    al = StableTSAligner()
    assert (al.model, al.lang, al.device) == ("small", "en", "cpu")
    # explicit args win over settings
    al2 = StableTSAligner(model="large-v3")
    assert al2.model == "large-v3"


# ---- _line_times: the validated PoC algorithm ----
def test_line_times_exact_three_lines():
    res = FakeResult([
        [(0.0, "あいう")],
        [(10.0, "えおか"), (11.0, "きく")],
        [(20.0, "けこ")],
    ])
    times = StableTSAligner._line_times(res, ["あいう", "えおかきく", "けこ"])
    assert times == [0.0, 10.0, 20.0]


def test_line_times_ignores_whitespace():
    # words carry leading spaces (whisper style) and lines carry spaces/tabs:
    # matching is by whitespace-stripped CHAR COUNTS, not exact text
    res = FakeResult([[(0.0, " hello"), (1.0, " world "), (5.0, "bye")]])
    times = StableTSAligner._line_times(res, ["hello   world", "bye"])
    assert times == [0.0, 5.0]


def test_line_times_repeated_line_gets_successive_times():
    # identical text twice -> the greedy walk must NOT reuse the first
    # occurrence's words; the second line lands on the second occurrence
    res = FakeResult([
        [(0.0, "lalala"), (5.0, "xx"), (9.0, "lalala"), (12.0, "yy")],
    ])
    times = StableTSAligner._line_times(res, ["lalala", "xx", "lalala", "yy"])
    assert times == [0.0, 5.0, 9.0, 12.0]


def test_line_times_overshoot_word_does_not_steal_next_start():
    # whisper glued "abcdefg" into one word; line 0 owns it (overshoot "g"
    # is consumed, never rewound), so line 1 starts at its NEXT word — it
    # must not be given word 0's earlier start, and must not skip forward.
    res = FakeResult([[(0.0, "abcdefg"), (4.0, "hij")]])
    times = StableTSAligner._line_times(res, ["abcdef", "ghij"])
    assert times == [0.0, 4.0]


def test_line_times_empty_and_no_words():
    # seg.words=None is tolerated; exhausted stream holds the previous time
    res = FakeResult([[(2.0, "ab")], None])
    times = StableTSAligner._line_times(res, ["ab", "cd"])
    assert times == [2.0, 2.0]


# ---- align() success path + monotonic clamp ----
class FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def align(self, audio, text, **kw):
        self.calls.append((audio, text, kw))
        return self.result


def test_align_passes_args_and_returns_timed_lines(tmp_path):
    al = StableTSAligner(lang="ja", model="medium", device="cuda")
    model = FakeModel(FakeResult([[(1.5, "one"), (7.0, "two")]]))
    al._load = lambda: model
    timed = al.align(tmp_path / "s.flac", lyrics_of("one", "two"))
    audio, text, kw = model.calls[0]
    assert audio == str(tmp_path / "s.flac")
    assert text == "one\ntwo"
    assert kw["language"] == "ja" and kw["regroup"] == "p" and kw["verbose"] is False
    assert [(t.text, t.start) for t in timed] == [("one", 1.5), ("two", 7.0)]


def test_align_clamps_non_monotonic_times(tmp_path):
    # a word hiccup handing line 2 an earlier start must be clamped, never
    # shipped (LRC players assume non-decreasing starts)
    al = StableTSAligner()
    al._load = lambda: FakeModel(FakeResult([
        [(0.0, "ab"), (5.0, "cd"), (3.0, "ef")],
    ]))
    timed = al.align(tmp_path / "s.wav", lyrics_of("ab", "cd", "ef"))
    assert [t.start for t in timed] == [0.0, 5.0, 5.0]


# ---- align() failure paths fall back to the injected/default aligner ----
def test_align_falls_back_when_load_raises(tmp_path, capsys):
    fb = FakeFallback()
    al = StableTSAligner(fallback=fb)

    def boom():
        raise RuntimeError("CUDA out of memory")

    al._load = boom
    timed = al.align(tmp_path / "s.wav", lyrics_of("hello there"))
    assert fb.calls, "fallback aligner must have been used"
    assert timed[0].start == 99.0
    err = capsys.readouterr().err
    assert "out of memory" in err and "falling back" in err


def test_align_falls_back_when_model_align_raises(tmp_path, capsys):
    fb = FakeFallback()
    al = StableTSAligner(fallback=fb)

    class BadModel:
        def align(self, *a, **k):
            raise RuntimeError("model download failed")

    al._load = lambda: BadModel()
    timed = al.align(tmp_path / "s.wav", lyrics_of("a b"))
    assert fb.calls and timed[0].start == 99.0
    assert "falling back" in capsys.readouterr().err


def test_align_empty_lyrics_falls_back(tmp_path, capsys):
    fb = FakeFallback()
    al = StableTSAligner(fallback=fb)
    al.align(tmp_path / "s.wav", lyrics_of())
    assert fb.calls
    assert "no lyric lines" in capsys.readouterr().err


def test_default_fallback_is_whisper_cpp():
    from lyrics_fetcher.aligner.whisper_cpp import WhisperCppAligner
    al = StableTSAligner()
    assert al.fallback is None  # not built at construction
    assert isinstance(al._fallback_aligner(), WhisperCppAligner)


# ---- CLI wiring ----
def test_cli_make_aligner_routes_stable_ts():
    from lyrics_fetcher.aligner.stable_ts import StableTSAligner
    from lyrics_fetcher.cli import _make_aligner
    # constructing must work even if stable_whisper were absent (lazy imports)
    al = _make_aligner(argparse.Namespace(aligner="stable-ts"))
    assert isinstance(al, StableTSAligner)


def test_cli_parser_accepts_stable_ts_everywhere_but_crosscheck():
    from lyrics_fetcher.cli import build_parser
    p = build_parser()
    for argv in (["compile", "a.flac", "b.txt", "--aligner", "stable-ts"],
                 ["full", "a.flac", "--aligner", "stable-ts"],
                 ["album", "dir", "--aligner", "stable-ts"]):
        assert p.parse_args(argv).aligner == "stable-ts"
    # cross-check is hardcoded whisper+qwen3 and stays that way
    with pytest.raises(SystemExit):
        p.parse_args(["cross-check", "a.flac", "b.txt", "--aligner", "stable-ts"])

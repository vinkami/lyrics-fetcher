"""Tests for the stable-ts aligner (opt-in forced alignment).

stable-ts is a dev-group dep: CI installs it (plain `uv sync` includes the
dev group), but `--no-dev` production installs lack the extra — the lazy
import keeps the module importable there, and these tests never touch
stable_whisper, the GPU, or a model download: the per-line timing logic
runs on FAKE stable-ts result objects (segments -> words with .start/.word),
and error paths use stubbed _load/model. Same fake-injection pattern as
tests/test_aligner.py and tests/test_separation.py.
"""
from __future__ import annotations

import argparse
import builtins
import unicodedata

import pytest

from lyrics_fetcher.aligner.base import BaseAligner, TimedLine
from lyrics_fetcher.aligner.stable_ts import _InsufficientCoverage, StableTSAligner
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
    module works in a --no-dev / production install without the extra."""
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


def test_line_times_empty_and_punct_only_lines_hold_previous():
    # Regression: blank lines (utaten emits them via bare splitlines) and
    # punctuation-only lines (「！！」norms to "") consume no words. They must
    # HOLD the previous start — a None would sail out of _line_times and the
    # monotonic clamp in align() runs outside the try -> uncaught TypeError,
    # breaking the "any failure falls back" contract.
    res = FakeResult([[(0.0, "ab"), (5.0, "cd")]])
    assert StableTSAligner._line_times(res, ["ab", "", "cd"]) == [0.0, 0.0, 5.0]
    assert StableTSAligner._line_times(res, ["ab", "！！", "cd"]) == [0.0, 0.0, 5.0]
    # leading empty line holds 0.0, does not anchor at a bogus word time
    assert StableTSAligner._line_times(res, ["", "ab", "cd"]) == [0.0, 0.0, 5.0]


def test_align_punct_only_line_does_not_crash(tmp_path):
    # end-to-end guard for the same hole through align()'s clamp
    class _M:
        def align(self, *a, **k):
            return FakeResult([[(1.0, "ab"), (6.0, "cd")]])

    al = StableTSAligner()
    al._model = _M()
    lyrics = Lyrics(source="test", lines=[LyricLine(text="ab"), LyricLine(text="！！"),
                           LyricLine(text="cd")])
    timed = al.align(tmp_path / "song.wav", lyrics)
    assert [t.start for t in timed] == [1.0, 1.0, 6.0]


# ---- _line_times: normalization parity (FIX B) ----
def test_line_times_words_omit_punctuation_and_brackets():
    # stable-ts segmentation can drop 、。！ and 「」; _norm must strip the
    # SAME character class as whisper.cpp's _clean on both sides, or the
    # mis-counted target over-consumes words and drift cascades downstream
    res = FakeResult([[(0.0, "こんにちは"), (5.0, "世界"), (9.0, "さよなら")]])
    times = StableTSAligner._line_times(res, ["こんにちは、世界！", "「さよなら」"])
    assert times == [0.0, 9.0]


def test_line_times_nfd_target_matches_nfc_word_stream():
    # JIS-sourced lyrics can be NFD-decomposed (ハ+゛…) while stable-ts emits
    # NFC; without NFKC the target counts extra combining marks
    nfd = unicodedata.normalize("NFD", "パーティー")  # 6 code points, 4 NFC
    res = FakeResult([[(0.0, "パーティー"), (6.0, "ok")]])
    times = StableTSAligner._line_times(res, [nfd, "ok"])
    assert times == [0.0, 6.0]


# ---- _line_times: frozen-tail coverage guard (FIX A) ----
def test_line_times_raises_on_long_frozen_tail():
    # 2-word stream vs 10 lines: 8 lines would freeze at the same timestamp
    res = FakeResult([[(0.0, "aa"), (1.0, "bb")]])
    lines = ["aa", "bb"] + ["zz"] * 8
    with pytest.raises(_InsufficientCoverage):
        StableTSAligner._line_times(res, lines)


def test_line_times_short_frozen_tail_stays_best_effort():
    # a legit silent outro (<=3 uncovered trailing lines) keeps holding the
    # previous time instead of bailing to the fallback
    res = FakeResult([[(0.0, "aa")]])
    times = StableTSAligner._line_times(res, ["aa", "bb", "cc"])
    assert times == [0.0, 0.0, 0.0]


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
    res = FakeResult([[(0.0, "ab"), (5.0, "cd"), (3.0, "ef")]])
    # _line_times stays RAW — the decrease (3.0 after 5.0) is visible here,
    # proving the clamp lives at the align() level, not inside _line_times
    assert StableTSAligner._line_times(res, ["ab", "cd", "ef"]) == [0.0, 5.0, 3.0]
    al._load = lambda: FakeModel(res)
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


def test_align_falls_back_on_insufficient_coverage(tmp_path, capsys):
    # FIX A: a 2-word stream for a 10-line Lyrics is a truncated alignment,
    # not a silent outro — must warn on stderr and use the fallback instead
    # of shipping 8 lines frozen at one timestamp
    fb = FakeFallback()
    al = StableTSAligner(fallback=fb)
    res = FakeResult([[(0.0, "l0"), (1.0, "l1")]])
    al._load = lambda: FakeModel(res)
    timed = al.align(tmp_path / "s.wav", lyrics_of(*[f"l{i}" for i in range(10)]))
    assert fb.calls, "fallback aligner must have been used"
    assert all(t.start == 99.0 for t in timed)
    err = capsys.readouterr().err
    assert "InsufficientCoverage" in err and "unanchored" in err and "falling back" in err


def test_load_failure_not_retried(monkeypatch):
    # FIX E: after a failed load the instance must not re-attempt (an offline
    # album run would pay ~5 s per track); and the ImportError names the fix
    al = StableTSAligner()
    attempts = []
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "stable_whisper":
            attempts.append(name)
            raise ImportError("No module named 'stable_whisper'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"install the dev extra: `uv sync --dev`"):
        al._load()
    with pytest.raises(RuntimeError, match="not retrying"):
        al._load()
    assert attempts == ["stable_whisper"]  # the 2nd call never re-imported


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


def test_cli_stable_ts_fallback_gets_whisper_flags(tmp_path):
    # FIX D: --binary/--model-whisper/--extra-model must reach the whisper.cpp
    # fallback, not just the default branch
    from lyrics_fetcher.aligner.stable_ts import StableTSAligner
    from lyrics_fetcher.aligner.whisper_cpp import WhisperCppAligner
    from lyrics_fetcher.cli import _make_aligner
    ns = argparse.Namespace(aligner="stable-ts",
                            binary=str(tmp_path / "whisper-cli"),
                            model_whisper=str(tmp_path / "ggml-medium.bin"),
                            extra_model=[str(tmp_path / "ggml-turbo.bin")])
    al = _make_aligner(ns)
    assert isinstance(al, StableTSAligner)
    fb = al.fallback
    assert isinstance(fb, WhisperCppAligner)
    assert fb.binary == tmp_path / "whisper-cli"
    assert fb.model == tmp_path / "ggml-medium.bin"
    assert fb.extra_models == [tmp_path / "ggml-turbo.bin"]


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

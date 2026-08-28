"""Tests for the alignment logic (monotonic DP + anchors).

Uses SYNTHETIC segments and dummy placeholder text — no real lyrics, no GPU.
These exercise the pure algorithm, not whisper transcription.
"""
from lyrics_fetcher.aligner.whisper_cpp import WhisperCppAligner


def clean(s):
    return WhisperCppAligner._clean(s)


def seg(text, from_ms=0, to_ms=5000):
    return {"from": from_ms, "to": to_ms, "text": text}


# ---- _align monotonic DP ----
def test_align_assigns_matching_segment():
    known = ["hello world", "goodbye moon"]
    segs = [seg("hello world"), seg("goodbye moon")]
    assign, _sim = WhisperCppAligner._align(known, segs)
    assert assign == [0, 1]


def test_align_is_monotonic_with_repeated_lines():
    # identical lines in two chorus blocks must map to increasing segments
    known = ["chorus line A", "bridge text", "chorus line A"]
    segs = [seg("chorus line A"), seg("bridge text"), seg("chorus line A")]
    assign, _sim = WhisperCppAligner._align(known, segs)
    # line 0 -> seg 0, line 1 -> seg 1, line 2 -> seg 2 (not reused seg 0)
    assert assign == [0, 1, 2]
    assert assign[0] < assign[1] < assign[2]


# ---- _anchor_align ----
def test_anchor_align_marks_confident_matches():
    known = ["exact match line", "totally different"]
    segs = [seg("exact match line"), seg("yyy zzz guess")]  # 2nd is unrelated but fuzzy
    anchors = WhisperCppAligner._anchor_align(known, segs, min_score=80.0)
    # first line matches exactly -> anchor; second is garbage -> not anchored
    assert anchors[0] is not None
    # second line has no strong match so it stays unanchored (None)
    assert anchors[1] is None


def test_anchor_align_no_anchors_when_all_garbled():
    # whisper hallucinated -> nothing matches -> all None
    known = ["real lyric line one", "real lyric line two"]
    segs = [
        seg("meow meow gibberish xyz"),
        seg("woof quack nonsense"),
    ]
    anchors = WhisperCppAligner._anchor_align(known, segs, min_score=80.0)
    assert all(a is None for a in anchors)


# ---- _clean ----
def test_clean_removes_punctuation():
    assert clean("こんにちは、世界！") == "こんにちは世界"
    # parens and dash are removed; spaces are not (but harmless for matching)
    assert clean("A (B) - C") == "ABC"


def test_align_preserves_order_when_segments_out_of_text_sequence():
    # ensure DP keeps known order even if fuzzy matches are not perfectly ordered
    known = ["first line", "second line", "third line"]
    segs = [seg("second line"), seg("first line"), seg("third line")]
    assign, _sim = WhisperCppAligner._align(known, segs)
    assert assign[0] <= assign[1] <= assign[2]
"""Tests for whisper-fetcher noise filtering (music notes + hallucination loops).

Pure logic on synthetic segments — no whisper, no GPU.
"""
from lyrics_fetcher.fetcher.whisper import _is_lyric_line, filter_whisper_lines


def seg(text, from_ms=0):
    return {"from": from_ms, "to": from_ms + 1000, "text": text}


# ---- _is_lyric_line ----
def test_music_note_only_is_not_lyric():
    assert not _is_lyric_line("♪~")
    assert not _is_lyric_line("♪ ♫ ♪")
    assert not _is_lyric_line("♪~♪~♪~")


def test_real_text_is_lyric_even_with_music_note():
    # a genuine line can contain a leading music note marker
    assert _is_lyric_line("♪ こんにちは世界")
    assert _is_lyric_line("Deliver the file")


# ---- filter_whisper_lines: music notes ----
def test_filters_out_pure_music_note_segments():
    segs = [seg("♪~"), seg("♪ ♫"), seg("こんにちは世界"), seg("♪〜")]
    lines = filter_whisper_lines(segs)
    assert [l.text for l in lines] == ["こんにちは世界"]


def test_all_music_notes_return_empty():
    assert filter_whisper_lines([seg("♪~"), seg("♪~")]) == []


# ---- filter_whisper_lines: hallucination loops ----
def test_clears_repeated_single_phrase():
    # whisper looping one phrase across a lyric-less song (ATLAS RUSH case)
    segs = [seg("「Santus Crush」 (Santus Crush)", from_ms=n * 1000) for n in range(8)]
    segs.append(seg("本日もご視聴有難う御座いました。", from_ms=8000))
    lines = filter_whisper_lines(segs)
    # dominated by one repeated phrase -> treated as no lyrics
    assert lines == []


def test_keeps_genuine_diverse_lyrics():
    segs = [seg("one line here", from_ms=n) for n in [0, 1000]]
    segs += [seg("a different line", from_ms=5000)]
    lines = filter_whisper_lines(segs)
    # "one line here" appears 2x (<4, not a loop) but consecutive dup collapses;
    # both distinct phrases survive
    assert [l.text for l in lines] == ["one line here", "a different line"]


def test_drops_consecutive_duplicate_lines():
    # exact consecutive repeats collapse but diverse content survives
    segs = [seg("chorus A", from_ms=0), seg("chorus A", from_ms=1000),
            seg("verse one", from_ms=2000)]
    lines = filter_whisper_lines(segs)
    assert [l.text for l in lines] == ["chorus A", "verse one"]
"""Tests for multi-song VLM OCR: JSON parsing, label normalization, fetch split.

Pure logic only — no vision server, no network, no GPU. Uses synthetic text and
monkeypatched ``_chat`` to inject canned VLM responses.
"""
from pathlib import Path

from lyrics_fetcher.ocr.vision import VLMOcr
from lyrics_fetcher.ocr.base import BaseOCR


def make_ocr(chat_impl=None):
    ocr = VLMOcr(api="http://localhost:1/v1/chat/completions", model="fake")
    if chat_impl is not None:
        ocr._chat = chat_impl
    return ocr


# ---- _parse_json_response ----
def test_parse_plain_json():
    assert VLMOcr._parse_json_response('{"a": "b"}') == {"a": "b"}


def test_parse_fenced_json():
    s = '```json\n{"songs": {"x": "y"}}\n```'
    assert VLMOcr._parse_json_response(s) == {"songs": {"x": "y"}}


def test_parse_json_with_trailing_text():
    # model often appends prose after the JSON
    s = '{"x": "y"} That is all.'
    assert VLMOcr._parse_json_response(s) == {"x": "y"}


def test_parse_invalid_returns_empty():
    assert VLMOcr._parse_json_response("no json here") == {}


# ---- _normalize_song_labels ----
def test_normalize_matches_canonical_title():
    songs = {"プリズム△▽リズム（Long ver.）": "a line"}
    known = ["プリズム△▽リズム (Long ver.)", "Edelweiss"]
    out = VLMOcr._normalize_song_labels(songs, known)
    # slug-normalized to the exact disc title
    assert "プリズム△▽リズム (Long ver.)" in out


def test_normalize_keeps_phantom_label():
    # "Sanctus" / "レクイエム" are song SECTIONS inside RondeauX, not real tracks
    songs = {"Sanctus": "lyrics", "RondeauX of RagnaroQ -Crescendo of Cataclysm-": "r"}
    known = ["RondeauX of RagnaroQ -Crescendo of Cataclysm-", "Edelweiss"]
    out = VLMOcr._normalize_song_labels(songs, known)
    # RondeauX matches the canonical title; Sanctus stays raw (phantom)
    assert "RondeauX of RagnaroQ -Crescendo of Cataclysm-" in out
    assert "Sanctus" in out


def test_normalize_no_known_titles_passthrough():
    songs = {"Some Title": "lyrics"}
    assert VLMOcr._normalize_song_labels(songs, []) == songs


# ---- extract_songs ----
def test_extract_songs_parses_and_normalizes(monkeypatch):
    ocr = make_ocr()
    responses = iter([
        '{"songs": {"プリズム△▽リズム（Long ver.）": "line1\\nline2", '
        '"RondeauX of RagnaroQ -Crescendo of Cataclysm-": "rline"}}'
    ])
    def fake_chat(prompt, image=None, max_tokens=2048):
        return next(responses)
    ocr._chat = fake_chat
    known = ["プリズム△▽リズム (Long ver.)", "RondeauX of RagnaroQ -Crescendo of Cataclysm-"]
    out = ocr.extract_songs(Path("page.jpg"), known_titles=known)
    assert "プリズム△▽リズム (Long ver.)" in out
    assert out["プリズム△▽リズム (Long ver.)"] == "line1\nline2"
    # cache hit -> chat not called again
    ocr._chat = lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache miss"))
    out2 = ocr.extract_songs(Path("page.jpg"), known_titles=known)
    assert out2 == out


def test_extract_songs_drops_empty_blocks():
    ocr = make_ocr()
    def fake_chat(prompt, image=None, max_tokens=2048):
        return '{"songs": {"A": "real", "B": "  "}}'
    ocr._chat = fake_chat
    out = ocr.extract_songs(Path("page.jpg"))
    assert "A" in out and "B" not in out


# ---- fetch: per-song split fixes bleed ----
def test_fetch_returns_only_matching_song():
    ocr = make_ocr()
    def fake_chat(prompt, image=None, max_tokens=2048):
        return ('{"songs": {"プリズム△▽リズム (Long ver.)": "p1\\np2", '
                '"RondeauX of RagnaroQ -Crescendo of Cataclysm-": "r1"}}')
    ocr._chat = fake_chat
    img = Path("page.jpg")
    # requesting プリズム returns only プリズム's lines
    lyr = ocr.fetch(img, "プリズム△▽リズム (Long ver.)")
    assert [l.text for l in lyr.lines] == ["p1", "p2"]
    # requesting Edelweiss (not on this page) returns empty -> no bleed
    lyr2 = ocr.fetch(img, "Edelweiss")
    assert not lyr2.lines


def test_fetch_returns_empty_when_no_song_matches():
    ocr = make_ocr()
    ocr._chat = lambda *a, **k: '{"songs": {"Other Song": "x"}}'
    lyr = ocr.fetch(Path("page.jpg"), "Target")
    assert not lyr.lines
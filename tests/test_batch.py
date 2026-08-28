"""Tests for multi-song album aggregation (BookletMapper.collect_album_songs).

Uses a FAKE OCR object that returns canned per-song dicts — no vision server,
no GPU.
"""
from pathlib import Path

from lyrics_fetcher.batch import BookletMapper
from lyrics_fetcher.utils import MUSIC_DIR


class FakeOcr:
    """Returns a fixed per-page {song_label: lyrics} mapping."""

    name = "fake-ocr"

    def __init__(self, pages):
        # {page_basename: {song_title: lyrics}}
        self.pages = pages

    def extract_songs(self, image: Path, known_titles=None):
        # known_titles ignored here — pages return pre-normalized canonical titles
        return self.pages.get(image.name, {})


def _make_track(tmp_path, canonical_title, ext=".dat"):
    # Use the file stem == the canonical metadata title so collect_album_songs's
    # title->path index (SongMeta.from_path().title or stem) matches the VLM
    # block labels. Non-audio extension so mutagen returns None gracefully.
    p = tmp_path / f"{canonical_title}{ext}"
    p.touch()
    return p


def test_collect_normalizes_and_aggregates_across_pages(tmp_path):
    # photo A: プリズム + RondeauX (first half); photo B: RondeauX (second half)
    ocr = FakeOcr({
        "a.jpg": {
            "プリズム△▽リズム (Long ver.)": "p1\np2",
            "RondeauX of RagnaroQ -Crescendo of Cataclysm-": "r1\nr2",
        },
        "b.jpg": {
            "RondeauX of RagnaroQ -Crescendo of Cataclysm-": "r3\nr4",
        },
    })
    prism = _make_track(tmp_path, "プリズム△▽リズム (Long ver.)")
    rondo = _make_track(tmp_path, "RondeauX of RagnaroQ -Crescendo of Cataclysm-")
    mapper = BookletMapper(tmp_path, ocr)

    per_track, phantoms = mapper.collect_album_songs(
        [Path("a.jpg"), Path("b.jpg")], [prism, rondo]
    )

    # プリズム got only its own lines (no RondeauX leak)
    assert [l.text for l in per_track[prism].lines] == ["p1", "p2"]
    # RondeauX stitched across both pages, in page order
    assert [l.text for l in per_track[rondo].lines] == ["r1", "r2", "r3", "r4"]
    assert phantoms == []


def test_collect_drops_phantom_blocks_and_reports(tmp_path):
    # Sanctus is a section header the VLM split out; it's not a real track
    ocr = FakeOcr({
        "a.jpg": {
            "RondeauX of RagnaroQ -Crescendo of Cataclysm-": "r1",
            "Sanctus": "section lyrics",
            "レクイエム": "more section lyrics",
        },
    })
    rondo = _make_track(tmp_path, "RondeauX of RagnaroQ -Crescendo of Cataclysm-")
    edel = _make_track(tmp_path, "Edelweiss")  # no lyrics on the page
    mapper = BookletMapper(tmp_path, ocr)

    per_track, phantoms = mapper.collect_album_songs([Path("a.jpg")], [rondo, edel])

    # RondeauX got its real lyrics; phantoms reported, not attached
    assert [l.text for l in per_track[rondo].lines] == ["r1"]
    assert edel not in per_track  # no lyrics -> absent
    assert ("a.jpg", "Sanctus") in phantoms
    assert ("a.jpg", "レクイエム") in phantoms


def test_collect_track_with_no_lyrics_absent(tmp_path):
    ocr = FakeOcr({
        "a.jpg": {"Only Song": "line one\nline two"},
    })
    a = _make_track(tmp_path, "Only Song")
    b = _make_track(tmp_path, "Instrumental Track")
    mapper = BookletMapper(tmp_path, ocr)
    per_track, _phantoms = mapper.collect_album_songs([Path("a.jpg")], [a, b])
    assert a in per_track
    assert b not in per_track  # skipped — no lyrics


def test_discover_audio_and_booklet(tmp_path):
    (tmp_path / "booklet").mkdir()
    (tmp_path / "01 a.flac").touch()
    (tmp_path / "02 b.mp3").touch()
    (tmp_path / "03 c.txt").touch()  # not audio
    (tmp_path / "booklet" / "p1.jpg").touch()
    (tmp_path / "booklet" / "p2.png").touch()
    mapper = BookletMapper(tmp_path, FakeOcr({}))
    audio = mapper.discover_audio()
    pages = mapper.discover_booklet()
    assert {p.name for p in audio} == {"01 a.flac", "02 b.mp3"}
    assert {p.name for p in pages} == {"p1.jpg", "p2.png"}
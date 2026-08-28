"""Tests for lyrics_fetcher.utils — title/artist matching, normalization, slugs.

Uses ONLY song titles and synthetic strings — no copyrighted lyric text. Song
titles are factual data, not copyrightable content.
"""
from lyrics_fetcher.utils import (
    _norm_ja,
    artist_match,
    best_match_index,
    norm_for_search,
    slugify,
    title_match,
)


# ---- normalization ----
def test_norm_ja_strips_parentheticals():
    assert _norm_ja("天ノ弱 (Amanojaku)") == "天ノ弱"
    assert _norm_ja("NOIZY BOUNCE (Extended)") == "noizybounce"


def test_norm_ja_strips_separators():
    # slash is stripped; middle dot (・) is a real char and retained (by design)
    assert _norm_ja("A/B") == "ab"
    assert "・" in _norm_ja("A・B")


def test_norm_for_search_strips_track_number():
    assert norm_for_search("01 アンデッド") == "アンデッド"
    # strips leading digits + following space, leaves any trailing dash/word
    assert norm_for_search("07 Title") == "Title"


# ---- title_match ----
def test_title_match_exact():
    assert title_match("アンデッド", "アンデッド")


def test_title_match_with_romanization_suffix():
    # candidate has a (Amanojaku) hint that must be ignored
    assert title_match("天ノ弱", "天ノ弱 (Amanojaku)")


def test_title_match_fuzzy_similar():
    assert title_match("パンダヒーロー", "パンダヒーロー (full ver.)")


def test_title_match_rejects_unrelated():
    assert not title_match("命を振り回せ", "黒い目")


# ---- artist_match ----
def test_artist_match_empty_requested_is_ok():
    # no requested artist -> never reject
    assert artist_match("", "anything")


def test_artist_match_exact():
    assert artist_match("光収容", "光収容")


def test_artist_match_substring():
    assert artist_match("ハチ", "ハチ feat. GUMI")


def test_artist_match_rejects_wrong():
    assert not artist_match("光収容", "164")


# ---- best_match_index ----
def test_best_match_index_picks_right_song():
    candidates = [
        ("アンデッド", "魔王魂"),   # wrong artist
        ("命を振り回せ", "光収容"), # wrong title
        ("アンデッド", "光収容"),   # correct
    ]
    idx = best_match_index(candidates, "アンデッド", "光収容")
    assert idx == 2


def test_best_match_index_returns_none_when_no_match():
    candidates = [("何か全く別の曲", "別アーティスト")]
    assert best_match_index(candidates, "アンデッド", "光収容") is None


# ---- slugify ----
def test_slugify_basic():
    assert slugify("Hello World") == "Hello_World"


def test_slugify_safe_for_spaces_and_jp():
    s = slugify("01 アンデッド")
    # keeps JP chars, safe on filesystems
    assert "/" not in s and "\\" not in s
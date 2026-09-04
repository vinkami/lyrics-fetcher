"""Common helpers: HTTP client, matching/verification, path utilities."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import requests

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
)


def get_session() -> requests.Session:
    """A requests.Session with a desktop browser UA (bypasses naive blocks)."""
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA, "Accept-Language": "ja,en;q=0.8"})
    return s


def get_json(url: str, params: dict | None = None, timeout: int = 20) -> dict:
    """GET a URL and return parsed JSON."""
    with get_session() as s:
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()


def get_html(url: str, params: dict | None = None, timeout: int = 20) -> str:
    """GET a URL and return its text."""
    with get_session() as s:
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text


def is_html_cloudflare(text: str) -> bool:
    """Heuristic: did the server return a Cloudflare challenge page?"""
    low = text[:2000].lower()
    return "just a moment" in low or ("cloudflare" in low and "captcha" in low)


def norm_for_search(title: str) -> str:
    """Strip leading track numbers like '01 ' and collapse whitespace."""
    return re.sub(r"^\d+\s+", "", title).strip()


# ---------------------------------------------------------------------------
# Song matching / verification
# ---------------------------------------------------------------------------
_NORM_RE = re.compile(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", re.UNICODE)


def _norm_ja(s: str) -> str:
    """Lowercase and drop separators/punctuation for comparing titles/artists."""
    # drop parentheticals (e.g. "天ノ弱 (Amanojaku)" -> "天ノ弱") — these are
    # often romanization hints that differ across sources
    s = re.sub(r"\([^)]*\)|（[^）]*）", "", s or "")
    return _NORM_RE.sub("", s.lower())


def title_match(requested: str, candidate: str, min_ratio: float = 50.0) -> bool:
    """True if candidate title matches the requested title well enough."""
    from thefuzz import fuzz

    r, c = _norm_ja(requested), _norm_ja(candidate)
    if not r or not c:
        return False
    if r in c or c in r or fuzz.ratio(r, c) >= min_ratio:
        return True
    return fuzz.partial_ratio(r, c) >= min_ratio


def artist_match(requested: str, candidate: str, min_ratio: float = 40.0) -> bool:
    """True if candidate artist matches requested artist. Lenient for JP."""
    from thefuzz import fuzz

    r, c = _norm_ja(requested), _norm_ja(candidate)
    if not r:  # no requested artist -> don't reject
        return True
    if not c:
        return False
    if r in c or c in r or fuzz.partial_ratio(r, c) >= min_ratio:
        return True
    return fuzz.ratio(r, c) >= min_ratio


def best_match_index(candidates: list[tuple[str, str]], title: str, artist: str = "",
                     min_total: float = 55.0) -> int | None:
    """Pick the index among candidates '(candidate_title, candidate_artist)'
    whose title+artist best match the requested song.

    Returns the best index or None if nothing matches well enough.
    Scoring rewards both title and artist agreeing.
    """
    from thefuzz import fuzz

    best_i, best_score = None, 0.0
    for i, (ct, ca) in enumerate(candidates):
        tr = fuzz.ratio(_norm_ja(title), _norm_ja(ct))
        if not artist:
            score = tr
        else:
            ar = fuzz.ratio(_norm_ja(artist), _norm_ja(ca))
            # require artist agreement to contribute; title must be close
            if ar < 30:
                continue
            score = tr * 0.6 + ar * 0.4
        if score > best_score:
            best_i, best_score = i, score
    return best_i if best_score >= min_total else None


# ---------------------------------------------------------------------------
# Path / slug helpers
# ---------------------------------------------------------------------------
def slugify(title: str) -> str:
    """A filesystem-safe slug from a song title."""
    safe = "".join(c if (c.isalnum() or c in "._- ") else "-" for c in title)
    safe = "_".join(safe.split())
    return safe or "untitled"


def quote_via(title: str) -> str:
    return quote(title, safe="")
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


#: trailing version/arrangement markers a pressing adds to a song title
#: (e.g. "8番出口 Short ver", "Edelweiss (Long ver.)",
#: "アマツキツネ (10th Anniversary ver.)"). Lyrics databases index the
#: *bare* song title, so searches with these suffixes miss.
_VERSION_WORDS = (
    r"ver(?:sion)?|mix|short|full|long|extended|radio(?:\s*edit)?|edit"
    r"|album|single|instrumental|inst|karaoke|movie|tv|size|extra"
    r"|remaster(?:ed)?|anniversary|edition"
    r"|off[\s-]*vocal|vocal[\s-]*off|\d+\s*分耐久|[+\-]\d*\s*key"
)
#: never strip: language versions genuinely have different lyric text,
#: piano arrangements have no vocals — resurrecting the vocal song's lyrics
#: for them would align wrong text to the audio
_UNSTRIPPABLE_RE = re.compile(
    r"english|japanese|\beng\b|\bjp\b|日本語|英語|piano",
    re.IGNORECASE,
)
#: trailing parenthetical holding a version marker: "(Long ver.)", "(Full)",
#: "(10th Anniversary ver.)", "(+1key)", "（日本語 ver）"
_PAREN_SUFFIX_RE = re.compile(
    rf"\s*[（(](?![^（）()]*{_UNSTRIPPABLE_RE.pattern})[^（）()]*"
    rf"(?:{_VERSION_WORDS})[^（）()]*[）)]\s*$",
    re.IGNORECASE,
)
#: trailing bare marker(s) outside parens: "Short ver", "Inst", "ver",
#: "30分耐久ver". (?<!\w) stops "Deliver"->"Deli" style mutilation; a marker
#: glued to a word (e.g. 鏡音レンver) is left alone — it may be meaningful.
_BARE_SUFFIX_RE = re.compile(
    rf"(?:\s|-)*(?<!\w)(?:{_VERSION_WORDS})(?:\s*[-\.]?\s*(?:ver(?:sion)?))?\.?\s*$",
    re.IGNORECASE,
)
#: titles that ARE instrumental/karaoke versions — their "lyrics" would be
#: the original song's text against music with no vocals. Never suggest a
#: variant for them; they should not match a lyrics page at all.
_INSTRUMENTAL_RE = re.compile(
    rf"\b(?:inst(?:rumental)?|karaoke|off[\s-]*vocal|vocal[\s-]*off)\b"
    rf"|BGM|サウンドトラック",
    re.IGNORECASE,
)


def title_variants(title: str) -> list[str]:
    """Candidate search titles for a disc track title, best first.

    Strips trailing version/arrangement markers iteratively (parentheticals
    first, then bare words):
      '8番出口 Short ver(+1key)' -> ['8番出口 Short ver(+1key)',
                                     '8番出口 Short ver', '8番出口']
    Stops when nothing more strips or the core is < 2 chars; drops dupes.
    The input title is always the first candidate. Instrumental/karaoke
    titles get NO variants: they have no lyrics and must stay that way.
    """
    title = title.strip()
    out = [title]
    if not title or _INSTRUMENTAL_RE.search(title):
        return out
    cur = title
    while True:
        stripped = _PAREN_SUFFIX_RE.sub("", cur).strip()
        if stripped == cur:
            stripped = _BARE_SUFFIX_RE.sub("", cur).strip()
        if (stripped == cur or len(stripped) < 2 or stripped in out
                or _INSTRUMENTAL_RE.search(stripped)):
            break
        out.append(stripped)
        cur = stripped
    return out


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
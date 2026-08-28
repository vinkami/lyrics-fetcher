"""utaten.com fetcher — vocaloid lyrics with furigana.

WORKS (2026-08-28): use https://utaten.com (NOT www — www.utaten.com fails DNS).
Search: GET /lyric/search?title=<query> -> result links /lyric/<id>/
Lyrics live in div.lyricBody. Contains span.ruby > span.rb (kanji) + span.rt (reading).
=> utaten provides FURIGANA for free. Removing .rt spans yields clean lyrics text.

DISAMBIGUATION (2026-08-28): utaten search returns many songs sharing the title.
We parse each candidate page's real title + artist (from <title> "…歌詞 <artist> ふりがな付"
and a.artistName) and pick the one matching the *requested* title/artist, instead of
blindly taking the first hit (which returned the wrong アンデッド vs ASTEROID's track).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from ..models import LyricLine, Lyrics
from ..utils import best_match_index, get_session, get_html
from .base import BaseFetcher

SEARCH_URL = "https://utaten.com/lyric/search"
MAX_CANDIDATES = 8


@dataclass
class _Candidate:
    url: str
    title: str = ""
    artist: str = ""
    text: str = ""
    ruby: dict[str, str] = None  # type: ignore[assignment]


def _extract_song_meta(soup: BeautifulSoup) -> tuple[str, str]:
    """Return (title, artist) from an utaten lyric page's <title> and artist links."""
    title = ""
    artist = ""
    t = soup.title.get_text(strip=True) if soup.title else ""
    m = re.match(r"^(.*?)\s*歌詞\s*(.*?)\s*ふりがな付", t)
    if m:
        title = m.group(1).strip()
        artist = m.group(2).strip()
    # fall back to artist link if title regex missed
    if not artist:
        for a in soup.select('a[href*="/artist/"]'):
            artist = a.get_text(strip=True)
            if artist:
                break
    return title, artist


def _parse_lyrics(soup: BeautifulSoup) -> tuple[list[str], dict[str, str]]:
    """Return (clean lyric lines, ruby pairs {kanji: reading})."""
    body = soup.select_one(".lyricBody")
    if not body:
        return [], {}
    ruby: dict[str, str] = {}
    for span in body.select("span.ruby"):
        rb = span.select_one(".rb")
        rt = span.select_one(".rt")
        if rb and rt:
            ruby[rb.get_text(strip=True)] = rt.get_text(strip=True)
    for rt in body.select("span.rt"):
        rt.decompose()
    lines = [l.strip() for l in body.get_text().splitlines() if l.strip()]
    return lines, ruby


class UtatenFetcher(BaseFetcher):
    name = "utaten"

    def fetch(self, title: str, artist: str = "") -> Lyrics:
        found = Lyrics(source=self.name, title=title, artist=artist)
        with get_session() as s:
            r = s.get(SEARCH_URL, params={"title": title}, timeout=15)
            if r.status_code != 200:
                return found
            links = sorted(set(re.findall(r'href="(/lyric/[a-z]{2}\d{6,}/)"', r.text)))
            if not links:
                return found

            # fetch up to MAX_CANDIDATES candidate pages and extract meta
            candidates: list[_Candidate] = []
            for path in links[:MAX_CANDIDATES]:
                try:
                    r2 = s.get(f"https://utaten.com{path}", timeout=15)
                    if r2.status_code != 200:
                        continue
                    soup = BeautifulSoup(r2.text, "html.parser")
                    cand_title, cand_artist = _extract_song_meta(soup)
                    lines, ruby = _parse_lyrics(soup)
                    if not lines:
                        continue
                    candidates.append(_Candidate(
                        url=f"https://utaten.com{path}",
                        title=cand_title, artist=cand_artist,
                        text="\n".join(lines), ruby=ruby,
                    ))
                except Exception:
                    continue

        if not candidates:
            return found

        # pick the candidate matching the requested song
        idx = best_match_index(
            [(c.title, c.artist) for c in candidates], title, artist
        )
        if idx is None:
            return found  # no candidate matches well -> treat as not found

        best = candidates[idx]
        found.source_url = best.url
        found.title = best.title or title
        found.artist = best.artist or artist
        found.ruby_all = best.ruby or {}
        found.lines = [LyricLine(text=l, ruby=best.ruby or {}) for l in best.text.splitlines()]
        return found
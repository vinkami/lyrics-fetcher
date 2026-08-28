"""utaten.com fetcher — vocaloid lyrics with furigana.

WORKS (2026-08-28): use https://utaten.com (NOT www — www.utaten.com fails DNS).
Search: GET /lyric/search?title=<query> -> result links /lyric/<id>/
Lyrics live in div.lyricBody. Contains span.ruby > span.rb (kanji) + span.rt (reading).
=> utaten provides FURIGANA for free. Removing .rt spans yields clean lyrics text.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from ..models import LyricLine, Lyrics
from ..utils import get_session
from .base import BaseFetcher

SEARCH_URL = "https://utaten.com/lyric/search"
DETAIL_URL = "https://utaten.com{path}"


class UtatenFetcher(BaseFetcher):
    name = "utaten"

    def fetch(self, title: str, artist: str = "") -> Lyrics:
        with get_session() as s:
            r = s.get(SEARCH_URL, params={"title": title}, timeout=15)
            if r.status_code != 200:
                return Lyrics(source=self.name, title=title, artist=artist)
            # find lyric detail page URLs on the search results
            links = sorted(set(re.findall(r'href="(/lyric/[a-z]{2}\d{6,}/)"', r.text)))
            if not links:
                return Lyrics(source=self.name, title=title, artist=artist)

            detail = DETAIL_URL.format(path=links[0])
            r2 = s.get(detail, timeout=15)
            if r2.status_code != 200:
                return Lyrics(source=self.name, title=title, artist=artist)

        soup = BeautifulSoup(r2.text, "html.parser")
        body = soup.select_one(".lyricBody")
        if not body:
            return Lyrics(source=self.name, title=title, artist=artist)

        # ruby pairs: kanji -> reading, for furigana companion output
        ruby_all: dict[str, str] = {}
        for span in body.select("span.ruby"):
            rb = span.select_one(".rb")
            rt = span.select_one(".rt")
            if rb and rt:
                ruby_all[rb.get_text(strip=True)] = rt.get_text(strip=True)

        # clean lyrics text (readings removed)
        for rt in body.select("span.rt"):
            rt.decompose()
        raw_lines = [l.strip() for l in body.get_text().splitlines() if l.strip()]

        return Lyrics(
            source=self.name,
            source_url=detail,
            title=title,
            artist=artist,
            lines=[LyricLine(text=line, ruby=ruby_all) for line in raw_lines],
            ruby_all=ruby_all,
        )
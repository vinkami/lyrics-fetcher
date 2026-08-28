"""Genius fetcher — search API + scrape embedded lyrics state.

WORKS (2026-08-28): Genius search API returns FULL urls (do NOT re-prepend domain).
Romanization/translation pages are skipped when looking for originals.
Lyrics are scraped from the window.__PRELOADED_STATE__ JSON.

DISAMBIGUATION (2026-08-28): search returns multiple pages for a title; we score
each candidate by its title + primary artist against the requested song and pick
the best match, rather than always taking the first hit.
"""
from __future__ import annotations

import json
import re

from ..models import LyricLine, Lyrics
from ..utils import best_match_index, get_json
from .base import BaseFetcher

SEARCH_API = "https://genius.com/api/search/multi"
PRELOADED = "window.__PRELOADED_STATE__ = JSON.parse('"


class GeniusFetcher(BaseFetcher):
    name = "genius"

    def _search(self, title: str, artist: str) -> list[dict]:
        q = f"{artist} {title}".strip()
        data = get_json(SEARCH_API, params={"q": q, "per_page": 5})
        songs = []
        for sec in data.get("response", {}).get("sections", []):
            for hit in sec.get("hits", []):
                res = hit.get("result") or {}
                url = res.get("url", "")
                if not url.endswith("-lyrics"):
                    continue
                if "romanizations" in url or "english-translation" in url:
                    continue
                songs.append(res)
        return songs

    @staticmethod
    def _js_single_quoted_body(html: str, i: int) -> str:
        SIMPLE = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "v": "\v"}
        out = []
        while i < len(html):
            c = html[i]
            if c == "'":
                break
            if c == "\\":
                nxt = html[i + 1]
                if nxt in SIMPLE:
                    out.append(SIMPLE[nxt])
                elif nxt == "u":
                    hexpart = html[i + 2:i + 6]
                    if hexpart.startswith("{"):
                        end = html.index("}", i + 2)
                        out.append(chr(int(html[i + 3:end], 16)))
                        i = end + 1
                        continue
                    out.append(chr(int(hexpart, 16)))
                    i += 6
                    continue
                else:
                    out.append(nxt)
                i += 2
                continue
            out.append(c)
            i += 1
        return "".join(out)

    def _scrape(self, url: str) -> list[str]:
        from ..utils import get_session

        with get_session() as s:
            r = s.get(url, timeout=15)
            if r.status_code != 200:
                return []
            i = r.text.find(PRELOADED)
            if i < 0:
                return []
            try:
                state = json.loads(self._js_single_quoted_body(r.text, i + len(PRELOADED)))
                body_html = state["songPage"]["lyricsData"]["body"]["html"]
            except Exception:
                return []
        text = re.sub(r"<br[^>]*>", "\n", body_html)
        text = re.sub(r"<[^>]+>", "", text)
        lines = [l.strip() for l in text.split("\n")]
        return [l for l in lines if l and not re.match(r"^\[.*\]$", l)]

    @staticmethod
    def _song_artist(hit: dict) -> str:
        pa = hit.get("primary_artist") or {}
        return pa.get("name") or ""

    def fetch(self, title: str, artist: str = "") -> Lyrics:
        hits = self._search(title, artist)
        found = Lyrics(source=self.name, title=title, artist=artist)
        if not hits:
            return found

        # score candidates by title + primary artist
        idx = best_match_index(
            [(h.get("title") or h.get("full_title") or "", self._song_artist(h)) for h in hits],
            title, artist,
        )
        if idx is None:
            return found
        hits = [hits[idx]]

        for h in hits:
            lines = self._scrape(h.get("url", ""))
            if lines:
                found.source_url = h.get("url", "")
                found.title = h.get("full_title") or title
                found.artist = self._song_artist(h) or artist
                found.lines = [LyricLine(text=l) for l in lines]
                break
        return found
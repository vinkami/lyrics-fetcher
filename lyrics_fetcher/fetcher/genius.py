"""Genius fetcher — search API + scrape embedded lyrics state.

WORKS (2026-08-28): Genius search API returns FULL urls (do NOT re-prepend domain).
Romanization/translation pages are skipped when looking for originals.
Lyrics are scraped from the window.__PRELOADED_STATE__ JSON.
"""
from __future__ import annotations

import json
import re

from ..models import LyricLine, Lyrics
from ..utils import get_session
from .base import BaseFetcher

SEARCH_API = "https://genius.com/api/search/multi"
PRELOADED = "window.__PRELOADED_STATE__ = JSON.parse('"


class GeniusFetcher(BaseFetcher):
    name = "genius"

    def _search(self, title: str, artist: str) -> list[dict]:
        from ..utils import get_json

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

    def fetch(self, title: str, artist: str = "") -> Lyrics:
        hits = self._search(title, artist)
        found = Lyrics(source=self.name, title=title, artist=artist)
        for h in hits:
            lines = self._scrape(h.get("url", ""))
            if lines:
                found.source_url = h.get("url", "")
                found.title = h.get("full_title") or title
                found.lines = [LyricLine(text=l) for l in lines]
                break
        return found
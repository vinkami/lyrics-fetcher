"""SilentBlue.RemyWiki fetcher — maimai/chunithm/ongeki rhythm-game songs.

WORKING (2026-08-28):
- api.php and ?action=raw are Cloudflare-blocked (403 challenge).
- BUT plain page loads work via a desktop browser UA.
- BEST METHOD: `index.php?search=<song title>` returns the ARTICLE DIRECTLY
  when the title matches exactly. Lyrics under the "Lyrics" section, in <pre>.
- Some songs are instrumental -> "None." (returns empty lyrics).
- JP songs may be stored under an English page title (e.g. Fake Face Failsafe),
  so a failed exact match falls back to following the top search result.

Method: index.php?search=TITLE -> walk siblings of the Lyrics .mw-heading div
-> collect text lines until the next .mw-heading.
"""
from __future__ import annotations

from bs4 import BeautifulSoup

from ..models import LyricLine, Lyrics
from ..utils import get_html, get_session, is_html_cloudflare
from .base import BaseFetcher

BASE = "https://silentblue.remywiki.com"
SEARCH_URL = BASE + "/index.php"


class SilentBlueFetcher(BaseFetcher):
    name = "silentblue"

    def _page_soup(self, url: str, params: dict | None = None, timeout: int = 15) -> BeautifulSoup | None:
        with get_session() as s:
            r = s.get(url, params=params, timeout=timeout)
            if r.status_code != 200 or is_html_cloudflare(r.text):
                return None
            return BeautifulSoup(r.text, "html.parser")

    @staticmethod
    def _section_lines(content, heading_id: str) -> list[str]:
        """Return lyric lines under the h2 with given id, walking siblings.

        Stops at the next TOP-LEVEL section (div.mw-heading2). Ignores subsection
        headings (div.mw-heading3) like "Japanese"/"English" but keeps their text.
        """
        h2 = content.find("h2", id=heading_id)
        if not h2:
            return []
        node = h2.parent.find_next_sibling()
        lines = []
        while node is not None:
            cls = " ".join(node.get("class") or [])
            if "mw-heading2" in cls:
                break
            if node.name in ("p", "div", "ul", "ol", "li", "table", "pre"):
                text = node.get_text("\n", strip=True)
                if text and "mw-heading" not in cls:
                    for sub in text.split("\n"):
                        sub = sub.strip()
                        if sub and sub != "None.":
                            lines.append(sub)
            node = node.find_next_sibling()
        return lines

    def fetch(self, title: str, artist: str = "") -> Lyrics:
        found = Lyrics(source=self.name, title=title, artist=artist)
        soup = self._page_soup(SEARCH_URL, params={"search": title})
        if soup is None:
            return found  # Cloudflare-blocked or HTTP error

        page_title = soup.select_one(".firstHeading, h1")
        if not page_title:
            return found
        pt = page_title.get_text(strip=True)

        # No exact-title match -> follow the first real search result
        if pt == "Search results":
            candidate = None
            for a in soup.select(".mw-search-result-heading a"):
                href = a.get("href", "")
                t = a.get_text(strip=True)
                if href and t and href != "/Main_Page":
                    candidate = (t, href)
                    break
            if not candidate:
                return found
            soup = self._page_soup(BASE + candidate[1])
            if soup is None:
                return found
            pt2 = soup.select_one(".firstHeading, h1")
            pt = pt2.get_text(strip=True) if pt2 else candidate[0]

        content = soup.select_one("#mw-content-text")
        if not content:
            return found

        # verify the resolved page title actually matches the requested song
        from ..utils import title_match

        if not title_match(title, pt):
            return found

        lines = self._section_lines(content, "Lyrics")
        if lines:
            found.title = pt
            # prefer any Japanese sub-block if present
            h3 = content.find("h3", string=lambda s: s and "japanese" in s.lower())
            if h3:
                node = h3.parent.find_next_sibling()
                jp = []
                while node is not None:
                    cls = " ".join(node.get("class") or [])
                    if "mw-heading2" in cls:
                        break
                    if node.name in ("p", "div", "ul", "ol", "li", "table", "pre") and "mw-heading" not in cls:
                        t = node.get_text("\n", strip=True)
                        if t:
                            jp.extend(x for x in t.split("\n") if x.strip())
                    node = node.find_next_sibling()
                if jp:
                    lines = jp
            found.lines = [LyricLine(text=l) for l in lines]
        return found
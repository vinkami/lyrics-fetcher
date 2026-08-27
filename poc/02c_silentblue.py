"""PoC: SilentBlue.RemyWiki fetcher (maimai/chunithm/ongeki songs).

WORKING (2026-08-28):
- api.php and ?action=raw are Cloudflare-blocked (403 challenge).
- BUT plain page loads work via curl with a browser UA.
- BEST METHOD: `index.php?search=<song title>` returns the ARTICLE DIRECTLY
  when the title matches exactly (no 403). Lyrics are under the "Lyrics" section.
- Lyrics subheadings: "Japanese" (maybe "English"). Some songs = "None." (instrumental).
- 日本語 (Japanese) songs store kanji/kana text. Some lyrics are romaji transcription.

So the fetcher: index.php?search=TITLE → walk siblings of the Lyrics .mw-heading div
→ collect text lines until the next .mw-heading.
"""
from pathlib import Path

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
HEADERS = {"User-Agent": UA}
SEARCH_URL = "https://silentblue.remywiki.com/index.php"


def _section_lines(content, heading_id: str) -> list[str]:
    """Return lyric lines under the h2 with given id, walking siblings.

    Stops at the next TOP-LEVEL section (div.mw-heading2). Ignores subsection
    headings (div.mw-heading3) like "Japanese"/"English" but keeps their text.
    """
    h2 = content.find("h2", id=heading_id)
    if not h2:
        return []
    node = h2.parent.find_next_sibling()  # sibling after the .mw-heading wrapper
    lines = []
    while node is not None:
        node_cls = " ".join(node.get("class") or [])
        if "mw-heading2" in node_cls:  # next top-level section
            break
        if node.name in ("p", "div", "ul", "ol", "li", "table", "pre"):
            node_text = node.get_text("\n", strip=True)
            if node_text and "mw-heading" not in node_cls:
                for sub in node_text.split("\n"):
                    sub = sub.strip()
                    if sub and sub != "None.":
                        lines.append(sub)
        node = node.find_next_sibling()
    return lines


def sb_fetch(title: str) -> dict | None:
    """Fetch lyrics for a song from SilentBlue.

    Returns {title, lyrics, jp_lyrics} or None if page not resolvable.
    """
    r = requests.get(SEARCH_URL, params={"search": title}, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    soup = BeautifulSoup(r.text, "html.parser")
    page_title = soup.select_one(".firstHeading, h1")
    if not page_title:
        return None
    pt = page_title.get_text(strip=True)

    # No exact-title match: page shows "Search results". Pick the best result page.
    is_results = (pt == "Search results")
    if is_results:
        candidates = []
        for a in soup.select(".mw-search-result-heading a"):
            href = a.get("href", "")
            t = a.get_text(strip=True)
            if href and t and href != "/Main_Page":
                candidates.append((t, href))
        if not candidates:
            return {"title": pt, "error": "no results"}
        # follow the top candidate page (redirects to localized title if needed)
        fb = requests.get(
            f"https://silentblue.remywiki.com{candidates[0][1]}",
            headers=HEADERS, timeout=15,
        )
        soup = BeautifulSoup(fb.text, "html.parser")
        page_title = soup.select_one(".firstHeading, h1")
        pt = page_title.get_text(strip=True) if page_title else candidates[0][0]

    content = soup.select_one("#mw-content-text")
    if not content:
        return {"title": pt, "error": "no content"}

    lines = _section_lines(content, "Lyrics")

    # Isolate Japanese lines if a Japanese subsection exists
    jp_lines = []
    h3 = content.find("h3", string=lambda s: s and "japanese" in s.lower())
    if h3:
        node = h3.parent.find_next_sibling()
        while node is not None:
            node_cls = " ".join(node.get("class") or [])
            if "mw-heading2" in node_cls:
                break
            if node.name in ("p", "div", "ul", "ol", "li", "table", "pre") and "mw-heading" not in node_cls:
                t = node.get_text("\n", strip=True)
                if t:
                    jp_lines.extend(l for l in t.split("\n") if l.strip())
            node = node.find_next_sibling()

    return {
        "title": pt,
        "lyrics": "\n".join(lines) or None,
        "jp_lyrics": "\n".join(jp_lines) or None,
    }


def main():
    print("=== PoC: SilentBlue.RemyWiki (maimai) ===\n")
    # maimai songs from the user's library
    songs = [
        "NOIZY BOUNCE",
        "Cryptarithm",
        "Ether Second",
        "The Great Banquet",
        "フェイクフェイス・フェイルセイフ",
        "AFTER PANDORA",
    ]
    for s in songs:
        print(f"--- {s} ---")
        try:
            res = sb_fetch(s)
        except Exception as e:
            print(f"  error: {e}")
            continue
        if not res:
            print("  not found")
        elif "error" in res:
            print(f"  {res['error']}")
        elif res["lyrics"]:
            print(f"  page: {res['title']} | {len(res['lyrics'].splitlines())} lines")
            for l in res["lyrics"].splitlines()[:4]:
                print(f"    {l}")
        else:
            print(f"  page: {res['title']} | (instrumental: None.)")
        print()


if __name__ == "__main__":
    main()
"""PoC: Genius lyrics extraction from song page HTML."""
import json
import re
import sys

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"

SEARCH_API = "https://genius.com/api/search/multi"
# NOTE: search API returns FULL urls (https://genius.com/...) — do not prepend the domain.


def genius_search(title: str, artist: str = "") -> list[dict]:
    import requests

    q = f"{artist} {title}".strip()
    r = requests.get(
        SEARCH_API, params={"q": q, "per_page": 5}, headers={"User-Agent": UA}, timeout=15
    )
    r.raise_for_status()
    data = r.json()
    songs = []
    for sec in data["response"].get("sections", []):
        for hit in sec.get("hits", []):
            res = hit.get("result") or {}
            url = res.get("url", "")
            if not url.endswith("-lyrics"):
                continue
            # skip romanization/translation pages when looking for originals
            if "romanizations" in url or "english-translation" in url:
                continue
            songs.append(res)
    return songs


def _js_single_quoted_body(html: str, i: int) -> str:
    """Evaluate a JS single-quoted string starting AFTER the opening quote."""
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
                hexpart = html[i + 2 : i + 6]
                if hexpart.startswith("{"):
                    end = html.index("}", i + 2)
                    out.append(chr(int(html[i + 3 : end], 16)))
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


def genius_scrape(url: str) -> str | None:
    """Scrape lyrics from a genius song page via its embedded __PRELOADED_STATE__."""
    import requests

    r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
    if r.status_code != 200:
        return None
    marker = "window.__PRELOADED_STATE__ = JSON.parse('"
    i = r.text.find(marker)
    if i < 0:
        print("  [genius] no preloaded state")
        return None
    try:
        state = json.loads(_js_single_quoted_body(r.text, i + len(marker)))
        body_html = state["songPage"]["lyricsData"]["body"]["html"]
    except Exception as e:
        print(f"  [genius] parse failed: {e}")
        return None

    text = re.sub(r"<br[^>]*>", "\n", body_html)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [l.strip() for l in text.split("\n")]
    # drop section headers like [Verse 1] and the title line
    lines = [l for l in lines if l and not re.match(r"^\[.*\]$", l)]
    return "\n".join(lines) if lines else None


def main():
    print("=== PoC: Genius ===\n")
    tests = [
        ("パンダヒーロー", "ハチ"),
        ("パズル", "ゆ依"),
    ]
    for title, artist in tests:
        print(f"--- {title} / {artist} ---")
        hits = genius_search(title, artist)
        print(f"  {len(hits)} candidate pages")
        for h in hits[:4]:
            print(f"    {h.get('full_title')}  {h.get('url')}")
        if hits:
            lyrics = genius_scrape(hits[0]["url"])
            if lyrics:
                lines = lyrics.splitlines()
                print(f"  SCRAPED {len(lines)} lines:")
                for l in lines[:6]:
                    print(f"    {l}")
        print()


if __name__ == "__main__":
    sys.exit(main())

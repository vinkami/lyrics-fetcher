"""PoC: Fetch lyrics from online sources (utaten + silentblue probe).

TESTED RESULTS (2026-08-28):
- utaten: WORKS at utaten.com (NO www — DNS fails for www.utaten.com).
  Search: GET /lyric/search?title=<query>  → result links /lyric/<id>/
  Lyrics live in div.lyricBody. Contains span.ruby > span.rb (kanji) + span.rt (reading)
  => utaten provides FURIGANA for free. Remove .rt spans to get clean lyrics text.
- silentblue.remywiki.com: BLOCKED by Cloudflare challenge page ("Just a moment...").
  Needs cloudscraper or playwright — TODO.
- genius: see 02b_genius.py (works).
"""

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

MUSIC_DIR = Path("/mnt/fnos/storage/Music")
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
HEADERS = {"User-Agent": UA}


def fetch_utaten(title: str, artist: str = "") -> dict | None:
    """Search utaten.com, fetch first lyrics page.

    Returns {url, text, ruby: [(kanji, reading), ...]} or None.
    """
    r = requests.get(
        "https://utaten.com/lyric/search",
        params={"title": title},
        headers=HEADERS,
        timeout=15,
    )
    if r.status_code != 200:
        return None
    links = sorted(set(re.findall(r'href="(/lyric/[a-z]{2}\d{6,}/)"', r.text)))
    if not links:
        return None
    detail = f"https://utaten.com{links[0]}"
    r2 = requests.get(detail, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r2.text, "html.parser")
    body = soup.select_one(".lyricBody")
    if not body:
        return None

    # ruby pairs: kanji -> furigana reading
    ruby_pairs = []
    for span in body.select("span.ruby"):
        rb, rt = span.select_one(".rb"), span.select_one(".rt")
        if rb and rt:
            ruby_pairs.append((rb.get_text(strip=True), rt.get_text(strip=True)))

    # clean text = body with readings removed
    for rt in body.select("span.rt"):
        rt.decompose()
    lines = [l.strip() for l in body.get_text().splitlines() if l.strip()]
    return {"url": detail, "text": "\n".join(lines), "ruby": ruby_pairs}


def probe_silentblue(title: str) -> str:
    """Probe silentblue wiki API; report blocked/ok/not-found."""
    url = "https://silentblue.remywiki.com/api.php"
    params = {"action": "query", "list": "search", "srsearch": title, "format": "json"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if "Just a moment" in r.text or "cloudflare" in r.text.lower()[:500]:
            return "BLOCKED by Cloudflare challenge"
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        results = r.json().get("query", {}).get("search", [])
        return results[0]["title"] if results else "no results"
    except Exception as e:
        return f"error: {e}"


def main():
    print("=== PoC: Lyrics Fetching ===\n")

    tests = [
        ("パンダヒーロー", "ハチ feat. GUMI"),
        ("天ノ弱", "164 feat. 初音ミク"),
        ("8番出口", "EO"),
        ("NOIZY BOUNCE", "八王子P"),
    ]

    for title, artist in tests:
        print(f"--- {title} / {artist} ---")
        try:
            res = fetch_utaten(title, artist)
        except Exception as e:
            res = None
            print(f"  [utaten] error: {e}")
        if res:
            lines = res["text"].splitlines()
            print(f"  [utaten] FOUND {len(lines)} lines, {len(res['ruby'])} ruby pairs  {res['url']}")
            for l in lines[:4]:
                print(f"    {l}")
            if res["ruby"][:3]:
                print(f"    ruby sample: {res['ruby'][:3]}")
        else:
            print("  [utaten] not found")
        print(f"  [silentblue] {probe_silentblue(title)}")
        print()


if __name__ == "__main__":
    main()

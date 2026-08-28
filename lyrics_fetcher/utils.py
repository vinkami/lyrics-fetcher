"""Common helpers: HTTP client, metadata → search queries, path utilities."""
from __future__ import annotations

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
    import re

    return re.sub(r"^\d+\s+", "", title).strip()


def slugify(title: str) -> str:
    """A filesystem-safe slug from a song title."""
    safe = "".join(c if (c.isalnum() or c in "._- ") else "-" for c in title)
    safe = "_".join(safe.split())
    return safe or "untitled"


def quote_via(title: str) -> str:
    return quote(title, safe="")


MUSIC_DIR = Path("/mnt/fnos/storage/Music")
AI_DIR = Path.home() / "AI"
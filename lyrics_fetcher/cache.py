"""SQLite cache for fetched lyrics.

Keys on (source, title, artist) so repeated fetches reuse stored results
instead of hitting the network / OCR again. Also caches OCR page text keyed by
absolute image path (stable across batches).

Storage: one SQLite db at ~/.cache/lyrics-fetcher/cache.db (overridable).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class LyricsCache:
    def __init__(self, db_path: Path | None = None):
        default = Path.home() / ".cache" / "lyrics-fetcher" / "cache.db"
        self.db_path = db_path or default
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path))
        self._con.execute(
            """CREATE TABLE IF NOT EXISTS lyrics_cache (
                source TEXT NOT NULL,
                title   TEXT NOT NULL,
                artist  TEXT NOT NULL,
                payload TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (source, title, artist)
            )"""
        )
        self._con.execute(
            """CREATE TABLE IF NOT EXISTS ocr_cache (
                image_path TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"""
        )
        self._con.commit()

    # ---- lyrics cache ----
    def get_lyrics(self, source: str, title: str, artist: str) -> dict | None:
        row = self._con.execute(
            "SELECT payload FROM lyrics_cache WHERE source=? AND title=? AND artist=?",
            (source, title, artist),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_lyrics(self, source: str, title: str, artist: str, payload: dict) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO lyrics_cache (source, title, artist, payload) VALUES (?,?,?,?)",
            (source, title, artist, json.dumps(payload, ensure_ascii=False)),
        )
        self._con.commit()

    # ---- OCR cache (by absolute image path) ----
    def get_ocr(self, image_path: Path) -> str | None:
        row = self._con.execute(
            "SELECT text FROM ocr_cache WHERE image_path=?", (str(image_path),)
        ).fetchone()
        return row[0] if row else None

    def put_ocr(self, image_path: Path, text: str) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO ocr_cache (image_path, text) VALUES (?,?)",
            (str(image_path), text),
        )
        self._con.commit()

    def close(self) -> None:
        self._con.close()
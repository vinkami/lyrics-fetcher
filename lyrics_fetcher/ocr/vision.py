"""Vision LLM OCR via a local llama-server (Qwen3.5-9B on RX 9060 XT).

PRODUCTION (2026-08-28): Qwen3.5-9B (Q4_K_M) served on 127.0.0.1:8081 (my own
start-vision; user's ~/AI/start untouched). Uses ~7.6GB VRAM, leaving ~9GB so
whisper-medium can run simultaneously. Gemma-4-12B failed (mojibake); 27B was
worse and used 16.4GB.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import requests
from PIL import Image

from ..config import settings
from ..models import Lyrics, LyricLine
from ..utils import _norm_ja
from .base import BaseOCR

# default Qwen vision server (override via constructor or config)
DEFAULT_API = settings.vision_api
DEFAULT_MODEL = settings.vision_model

VLM_PROMPT = (
    "This is a photo of an album lyrics booklet page (Japanese). "
    "Extract ALL the text exactly as printed, preserving line breaks. "
    "Only output the transcribed lyrics text, one line per lyric line. "
    "Do not add commentary, do not translate, do not include furigana readings separately. "
    "If the page contains song title/artist headers, output them as [title] ... style lines."
)

# Post-OCR cleanup prompt: fixes minor VLM slips (dropped particles/morphemes,
# wrong kanji) WITHOUT re-reading the image. Deliberately asks to change as
# little as possible — we trust the transcription, only correct obvious slips.
CLEANUP_PROMPT = (
    "Below is an OCR transcription of Japanese song lyrics from a booklet. "
    "Fix ONLY clear transcription errors: a dropped particle/morpheme (e.g. "
    "missing の/一), a wrong kanji that's obviously a misread, or an awkward "
    "morph that breaks grammar. Keep every line's meaning and line breaks; do "
    "not reformat, do not add/remove lines, do not translate, do not editorialize. "
    "If a line looks fine, output it unchanged. Output only the corrected lines, "
    "one per line, nothing else.\n\n"
    "LYRICS:\n{text}"
)

# Structured per-song extraction. A booklet page often contains MORE THAN ONE
# song (real albums, e.g. PRiSM: プリズム + RondeauX on one photo). We ask the
# VLM to split by printed title header into a JSON {song_title: lyrics}. The
# caller then matches each block to a real on-disc track and reassembles split
# songs across pages. ``{known_titles}`` is a hint list of the expected tracks.
VLM_PROMPT_SONGS = (
    "This is a photo from a Japanese album lyrics booklet. The page may contain "
    "lyrics for MORE THAN ONE song. {known_titles}\n"
    "Identify EVERY song present on this page by its title, and extract each "
    "song's lyrics SEPARATELY. Return ONLY a JSON object whose keys are the "
    "exact song titles as printed and whose values are that song's lyric lines, "
    "one lyric line per line (use \\n between lines). Example:\n"
    '{{"song-title-A": "line1\\nline2", "song-title-B": "line1"}}.\n'
    'If a song has no lyrics (instrumental) do NOT include it. '
    "Output nothing but the JSON."
)

MAX_SIDE = 1568


class VLMOcr(BaseOCR):
    name = "ocr-vlm"

    def __init__(self, api: str | None = None, model: str | None = None,
                 timeout: int = 600, cache=None, clean: bool = True):
        self.api = api or settings.vision_api
        self.model = model or settings.vision_model
        self.timeout = timeout
        self.cache = cache
        self.clean = clean
        self._songs_cache: dict[Path, dict[str, str]] = {}

    def _chat(self, prompt: str, image: Path | None = None, max_tokens: int = 2048) -> str:
        """Call the llama-server vision/text endpoint; returns assistant text."""
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": self._encode(image)}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        r = requests.post(self.api, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _encode(image: Path) -> str:
        im = Image.open(image).convert("RGB")
        w, h = im.size
        scale = min(1.0, MAX_SIDE / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def cleanup(self, text: str) -> str:
        """Post-OCR cleanup pass: fix minor slips (dropped particles/kanji)."""
        return self._chat(CLEANUP_PROMPT.format(text=text), max_tokens=2048).strip()

    def ocr(self, image: Path) -> str:
        # cache hit by absolute image path
        if self.cache:
            hit = self.cache.get_ocr(image)
            if hit is not None:
                return hit
        text = self._chat(VLM_PROMPT, image=image)
        if self.clean:
            try:
                text = self.cleanup(text)
            except Exception:
                pass  # cleanup is best-effort; keep raw transcription
        if self.cache:
            self.cache.put_ocr(image, text)
        return text

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Robustly parse the model's JSON reply.

        Handles markdown code fences and trailing text — finds the first balanced
        JSON object. Returns {} on failure (caller falls back gracefully).
        """
        t = text.strip()
        fence = re.match(r"^```(?:json\s*)?\s*(.*?)\s*```\s*$", t, re.DOTALL)
        if fence:
            t = fence.group(1)
        start = t.find("{")
        if start == -1:
            return {}
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        return {}
                if depth < 0:
                    return {}
        return {}

    @staticmethod
    def _normalize_song_labels(songs: dict[str, str],
                               known_titles: list[str]) -> dict[str, str]:
        """Map each VLM song-label to the best-matching known track title.

        VLM titles vary (e.g. "プリズム△▽リズム（Long ver.）" vs the disc's
        "プリズム△▽リズム (Long ver.)", or phantom section headers like "Sanctus"
        / "レクイエム"). We fuzzy-match each label against the real track titles
        and, on a solid match, rename it to the canonical title. Blocks matching
        nothing (phantoms) keep their raw label so the caller can decide.
        """
        from thefuzz import fuzz

        if not known_titles:
            return songs
        out: dict[str, str] = {}
        for label, lyrics in songs.items():
            lr = _norm_ja(label)
            best, best_score = None, 0.0
            for t in known_titles:
                tr = _norm_ja(t)
                if not tr:
                    continue
                s = fuzz.ratio(lr, tr)
                if s > best_score:
                    best, best_score = t, s
            out[best if best_score >= 55 else label] = lyrics
        return out

    def extract_songs(self, image: Path,
                      known_titles: list[str] | None = None) -> dict[str, str]:
        """Split a booklet page into ``{song_title: lyrics}``.

        ``known_titles`` (optional) hints the expected on-disc tracks and is used
        to normalize the returned keys to canonical track titles. Results are
        cached per image within this instance's lifetime.
        """
        if image in self._songs_cache:
            return self._songs_cache[image]
        known = list(known_titles or [])
        hint = ("The album's known song titles are:\n"
                + "\n".join(f"  - {t}" for t in known)) if known else (
                    "There may be one, several, or no songs on this page."
                )
        prompt = VLM_PROMPT_SONGS.format(known_titles=hint)
        raw = self._chat(prompt, image=image, max_tokens=6000)
        data = self._parse_json_response(raw)
        songs = data.get("songs", data) if isinstance(data, dict) else {}
        if not isinstance(songs, dict):
            songs = {}
        clean = {
            str(k).strip(): v.strip()
            for k, v in songs.items()
            if isinstance(v, str) and v.strip()
        }
        clean = self._normalize_song_labels(clean, known)
        self._songs_cache[image] = clean
        return clean

    def fetch(self, image: Path, title: str = "", artist: str = "") -> Lyrics:
        """OCR an image, returning only the block matching ``title``.

        A multi-song page is split per-song; we pick the block whose title
        matches. If none matches (e.g. the page is a different song, or the
        track is instrumental so it has no block), return empty Lyrics instead
        of leaking another song's lyrics.
        """
        songs = self.extract_songs(image, known_titles=[title] if title else None)
        block = songs.get(title) if title else None
        if block is not None:
            return Lyrics(
                source=self.name, title=title, artist=artist,
                lines=[LyricLine(l) for l in block.splitlines() if l.strip()],
            )
        # no matching block for this track
        return Lyrics(source=self.name, title=title, artist=artist)
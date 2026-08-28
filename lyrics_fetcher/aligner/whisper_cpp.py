"""whisper.cpp aligner — line-level timestamps via GPU transcription + DTW.

Whisper runs on Vulkan (RX 9060 XT); the medium multilingual model is the most
accurate for synthetic vocals (better than large-v3-turbo/small, verified).

Known lyrics are AUTHORITATIVE for the text; whisper only supplies TIMESTAMPS.
Monotonic DP fits the lyric lines to whisper segments such that line n maps to a
segment >= line n-1 — this keeps repeated choruses in correct temporal order
(the greedy best-match version collapsed them).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from ..models import Lyrics
from .base import BaseAligner, TimedLine

DEFAULT_BIN = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
DEFAULT_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-medium.bin"


class WhisperCppAligner(BaseAligner):
    name = "whisper-cpp"

    def __init__(self, binary: Path = DEFAULT_BIN, model: Path = DEFAULT_MODEL,
                 lang: str = "ja", max_len: int = 40, device: int = 0):
        self.binary = binary
        self.model = model
        self.lang = lang
        self.max_len = max_len
        self.device = device

    # ---- transcription ----
    def _segments(self, audio: Path) -> list[dict]:
        """Run whisper-cli, return segments: [{from_ms, to_ms, text}]."""
        out = Path(
            "/tmp"  # whisper writes <out>.json; use a unique temp prefix
        ) / f"_lf_whisper_{self.model.stem}"
        cmd = [
            str(self.binary), "-m", str(self.model), "-l", self.lang,
            "-f", str(audio), "-ml", str(self.max_len),
            "-oj", "-of", str(out), "--no-prints",
            "-dev", str(self.device),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        data = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
        segs = []
        for s in data.get("transcription", []):
            off = s["offsets"]
            segs.append({
                "from": off["from"],
                "to": off["to"],
                "text": s["text"].strip(),
            })
        return segs

    # ---- normalization ----
    @staticmethod
    def _clean(s: str) -> str:
        return re.sub(r"[\s「」『』（）()〈〉【】、。,.!！?？\-]", "", s)

    # ---- monotonic DP alignment ----
    @staticmethod
    def _align(known: list[str], segs: list[dict]) -> list[int]:
        """Return, for each known line, the assigned whisper segment index.

        Uses dynamic programming over a similarity matrix with a NON-DECREASING
        constraint: line i maps to a segment index >= line i-1's. This maximizes
        cumulative fuzzy-match similarity while preserving lyric order and keeping
        repeated-chorus lines in correct temporal arrangement.
        """
        from thefuzz import fuzz

        n_lines, n_segs = len(known), len(segs)
        # similarity[i][j] = fuzz ratio of known[i] vs clean(seg[j].text)
        sim = [[0.0] * n_segs for _ in range(n_lines)]
        clean = WhisperCppAligner._clean
        for i, line in enumerate(known):
            cl = clean(line)
            for j, seg in enumerate(segs):
                sim[i][j] = fuzz.ratio(cl, clean(seg["text"]))

        NEG = float("-inf")
        dp = [[NEG] * n_segs for _ in range(n_lines)]
        back = [[-1] * n_segs for _ in range(n_lines)]
        for j in range(n_segs):
            dp[0][j] = sim[0][j]
        for i in range(1, n_lines):
            running = NEG
            argk = None
            for j in range(n_segs):
                if dp[i - 1][j] > running:  # prefix max of prev row up to j
                    running = dp[i - 1][j]
                    argk = j
                if argk is not None:
                    dp[i][j] = running + sim[i][j]
                    back[i][j] = argk

        best_j = max(range(n_segs), key=lambda j: dp[n_lines - 1][j])
        assign = [0] * n_lines
        j = best_j
        for i in range(n_lines - 1, 0, -1):
            assign[i] = j
            j = back[i][j]
        assign[0] = j
        return assign

    # ---- public API ----
    def align(self, audio: Path, lyrics: Lyrics) -> list[TimedLine]:
        segs = self._segments(audio)
        known = [l.text for l in lyrics.lines]
        if not known or not segs:
            return [TimedLine(text=l.text, start=0.0) for l in lyrics.lines]

        assign = self._align(known, segs)
        timed = []
        for i, line in enumerate(lyrics.lines):
            seg = segs[assign[i]]
            timed.append(TimedLine(text=line.text, start=seg["from"] / 1000.0, end=seg["to"] / 1000.0))
        return timed
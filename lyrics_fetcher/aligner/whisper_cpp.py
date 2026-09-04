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

from ..config import settings
from ..models import Lyrics
from .base import BaseAligner, TimedLine

class WhisperCppAligner(BaseAligner):
    name = "whisper-cpp"

    def __init__(self, binary: Path | None = None, model: Path | None = None,
                 lang: str | None = None, max_len: int | None = None,
                 device: int | None = None,
                 extra_models: tuple[Path, ...] | None = None):
        # resolve against live config at construction time (so --config works)
        self.binary = binary or settings.whisper_bin
        self.model = model or settings.whisper_model
        self.lang = lang or settings.whisper_lang
        self.max_len = max_len if max_len is not None else settings.whisper_max_len
        self.device = device if device is not None else settings.whisper_device
        # additional whisper models to ALSO transcribe with, useful when the
        # primary model hallucinates on a hard song (e.g. 告げよ). Their segments
        # join the same anchor pool so the best match per line wins.
        if extra_models is None:
            self.extra_models = list(settings.whisper_extra_models)
        else:
            self.extra_models = list(extra_models)

    # ---- transcription ----
    def _segments(self, audio: Path, model: Path | None = None) -> list[dict]:
        """Run whisper-cli with a model, return segments [{from_ms,to_ms,text}]."""
        model = model or self.model
        out = Path("/tmp") / f"_lf_whisper_{model.stem}"
        cmd = [
            str(self.binary), "-m", str(model), "-l", self.lang,
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

    def _all_segments(self, audio: Path) -> list[dict]:
        """Transcribe with the primary + any extra models, return merged segments."""
        all_segs = self._segments(audio, self.model)
        for m in self.extra_models:
            try:
                all_segs += self._segments(audio, m)
            except Exception:
                pass  # a failing extra model shouldn't abort alignment
        return all_segs

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
        repeated-chorus lines in correct temporal arrangement. Returns the
        similarity matrix too so the caller can gauge confidence.
        """
        from thefuzz import fuzz

        n_lines, n_segs = len(known), len(segs)
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
        return assign, sim

    @staticmethod
    def _anchor_align(known: list[str], segs: list[dict],
                      min_score: float = 58.0) -> list[int]:
        """Anchor-based alignment.

        Same monotonic DP as ``_align``, but only lines whose best similarity
        exceeds ``min_score`` are treated as reliable TIME ANCHORS. This fixes
        the failure mode where whisper hallucinates (e.g. 告げよ looped garbage):
        there, no line scores high, so we produce no anchors and fall back to
        an even spread in ``align``.

        Returns a list of ``line_idx -> time_anchor``; entries with None are
        unanchored and will be interpolated by the caller.
        """
        assign, sim = WhisperCppAligner._align(known, segs)
        anchors: list[int | None] = [None] * len(known)
        for i, si in enumerate(assign):
            score = sim[i][si]
            if score >= min_score:
                anchors[i] = si
        return anchors

    # ---- public API ----
    def align(self, audio: Path, lyrics: Lyrics) -> list[TimedLine]:
        segs = self._all_segments(audio)
        # merged segments from multiple models aren't time-sorted; sort so the
        # DP's monotonic constraint is meaningful (segments in chronological order)
        segs.sort(key=lambda s: s["from"])
        known = [l.text for l in lyrics.lines]
        if not known or not segs:
            return [TimedLine(text=l.text, start=0.0) for l in lyrics.lines]

        anchors = self._anchor_align(known, segs)
        n_lines = len(known)
        # song duration = last segment end
        duration = segs[-1]["to"] / 1000.0

        # Build the time for each line:
        #   - anchored lines get their segment's start time.
        #   - unanchored lines are interpolated between the nearest anchored
        #     neighbors (or the song edges) proportionally by line index.
        times: list[float] = [0.0] * n_lines
        # gather anchor times by line index
        anchor_times = {i: segs[si]["from"] / 1000.0 for i, si in enumerate(anchors) if si is not None}

        # fill ranges between consecutive anchor lines
        keys = sorted(anchor_times)
        # before the first anchor
        if keys:
            first, fv = keys[0], anchor_times[keys[0]]
            if first > 0:
                # spread lines [0, first) leading up to first anchor, starting near 0
                for i in range(first):
                    times[i] = fv * (i + 1) / (first + 1)
            for i in range(first, keys[-1] + 1):
                if i in anchor_times:
                    times[i] = anchor_times[i]
                else:
                    # find prev/next anchor
                    prev = None
                    for k in keys:
                        if k < i:
                            prev = k
                        else:
                            break
                    nxt = next((k for k in keys if k > i), None)
                    if prev is not None and nxt is not None:
                        pv, nv = anchor_times[prev], anchor_times[nxt]
                        times[i] = pv + (nv - pv) * (i - prev) / (nxt - prev)
                    elif prev is not None:
                        times[i] = anchor_times[prev]  # clamp
            # after the last anchor
            last, lv = keys[-1], anchor_times[keys[-1]]
            if last < n_lines - 1:
                remaining = (duration - lv) / (n_lines - last)
                for i in range(last + 1, n_lines):
                    times[i] = lv + remaining * (i - last)
        else:
            # no anchors at all (whisper hallucinated): even spread across song
            for i in range(n_lines):
                times[i] = duration * i / max(n_lines - 1, 1)

        # build timed lines (single-word lines get ~1s end for display)
        timed = [TimedLine(text=known[i], start=times[i]) for i in range(n_lines)]
        # hard safety: monotonic non-decreasing
        for i in range(1, len(timed)):
            if timed[i].start < timed[i - 1].start:
                timed[i] = TimedLine(text=timed[i].text, start=timed[i - 1].start)
        return timed
"""faster-whisper aligner — native Python alignment (alternative to whisper.cpp).

faster-whisper uses CTranslate2. On this machine whisper.cpp (Vulkan) is the
preferred backend here (Vulkan builds for AMD/NVIDIA); this class is a CPU/other-backend
fallback and for first-line segments. It runs faster-whisper directly and maps
its segment timestamps monotonically to the known lyrics.
"""
from __future__ import annotations

from pathlib import Path

from ..models import Lyrics
from .base import BaseAligner, TimedLine
from .whisper_cpp import WhisperCppAligner  # reuse monotonic DP


class FasterWhisperAligner(BaseAligner):
    name = "faster-whisper"

    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8", language: str = "ja"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device,
                                       compute_type=self.compute_type)

    def _segments(self, audio: Path) -> list[dict]:
        self._load()
        segs, _ = self._model.transcribe(str(audio), language=self.language,
                                          vad_filter=True, beam_size=5)
        return [
            {"from": int(s.start * 1000), "to": int(s.end * 1000), "text": s.text.strip()}
            for s in segs
        ]

    def align(self, audio: Path, lyrics: Lyrics) -> list[TimedLine]:
        segs = self._segments(audio)
        known = [l.text for l in lyrics.lines]
        if not known or not segs:
            return [TimedLine(text=l.text, start=0.0) for l in lyrics.lines]
        assign, _ = WhisperCppAligner._align(known, segs)
        return [
            TimedLine(text=lyrics.lines[i].text, start=segs[assign[i]]["from"] / 1000.0,
                      end=segs[assign[i]]["to"] / 1000.0)
            for i in range(len(known))
        ]
"""Qwen3-ForcedAligner — LLM-based forced alignment (multilingual, incl. Japanese).

Qwen3-ForcedAligner-0.6B is a non-autoregressive LLM timestamp predictor that
aligns known (audio, text) pairs. Unlike whisper it uses your reference lyrics
as ground truth (no recognition/hallucination step) — the right tool for songs
whisper can't transcribe (e.g. 告げよ) and as an independent timing cross-check.

CRITICAL SETUP (2026-08-28):
- Load via transformers-native `AutoModelForTokenClassification` + `AutoProcessor`,
  NOT the `qwen-asr` pip package (that wrapper instantiates the wrong 12.7B
  `Qwen3ASRForConditionalGeneration` and OOMs; the checkpoint is really 918M).
- This requires transformers built from git main (the `qwen3_asr` architecture
  is only in dev/main, not the 4.57.6 PyPI pin).
- torch 2.11.0+rocm7.2 from the PyTorch ROCm wheel index (not PyPI).

Model weights on the NAS:
  /mnt/fnos/storage/ai-models/qwen3-forcedaligner/model
"""
from __future__ import annotations

from pathlib import Path

from ..config import settings
from ..models import Lyrics
from .base import BaseAligner, TimedLine

LOCAL_MODEL = settings.qwen3_aligner_model


class Qwen3ForcedAligner(BaseAligner):
    name = "qwen3-forcedaligner"

    def __init__(self, model_dir: Path | None = None,
                 device: str | None = None, language: str | None = None):
        # resolve against live config at construction time (so --config works)
        self.model_dir = str(model_dir or settings.qwen3_aligner_model)
        self.device = device
        self.language = language or settings.qwen3_aligner_language
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_dir)
        self._model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        # move the whole model to the device (from_pretrained without device_map
        # leaves weights on CPU / meta in transformers 5.x)
        self._model.to(self.device)
        self._model.eval()

    def align(self, audio: Path, lyrics: Lyrics) -> list[TimedLine]:
        """Align known lyrics lines to the audio.

        Returns word-level timestamps from the aligner; we stitch words back
        into lyric lines (a line = the words until its text is matched).
        """
        import torch
        from thefuzz import fuzz

        self._load()
        transcript = "\n".join(l.text for l in lyrics.lines)

        # prepare forced-aligner inputs (audio + transcript + language)
        inputs, word_lists = self._processor.prepare_forced_aligner_inputs(
            audio=str(audio), transcript=transcript, language=self.language
        )
        # move tensors to the model device, casting float inputs to the model dtype
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if v.dtype.is_floating_point:
                    inputs[k] = v.to(self.device, self._model.dtype)
                else:
                    inputs[k] = v.to(self.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # decode word-level timestamps
        decoded = self._processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self._model.config.timestamp_token_id,
        )[0]

        # Reconstruct line boundaries by greedily consuming decoded words until
        # their concatenated text fuzzy-matches each known lyric line. The decoded
        # word stream covers the whole transcript; per-line boundaries aren't
        # provided, so we match by accumulating text (fuzzy — handles the kanji/
        # kana variance the aligner's transcript words and the OCR text can have).
        memo = [_norm(w["text"]) for w in decoded]
        timed_lines: list[TimedLine] = []
        li = 0
        n = len(decoded)
        for line in lyrics.lines:
            exp = _norm(line.text)
            best_score, best_j = 0, li
            acc = ""
            for j in range(li, min(n, li + len(exp) + 12)):
                acc += memo[j]
                s = fuzz.ratio(acc, exp)
                if s > best_score or (s == best_score and exp.startswith(acc)):
                    best_score = s
                    best_j = j + 1
                    if best_score == 100:
                        break
            if best_j > n:
                best_j = n
            start = float(decoded[li]["start_time"]) if li < n else 0.0
            end = float(decoded[best_j - 1]["end_time"]) if best_j > 0 else start
            timed_lines.append(TimedLine(line.text, start, end))
            li = best_j
            if li >= n:
                # pad remaining lines with the last end time
                for _r in range(len(timed_lines), len(lyrics.lines)):
                    timed_lines.append(TimedLine(lyrics.lines[_r].text, end, end))
                break

        return timed_lines


def _norm(s: str) -> str:
    import re

    return re.sub(r"[\s、。、」『』（）()]", "", s)
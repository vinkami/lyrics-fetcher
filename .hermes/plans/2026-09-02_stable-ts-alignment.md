# stable-ts Alignment Plan — fix the "breaks"/desync in whisper alignment

> **For Hermes:** Use subagent-driven-development skill to implement once the go/no-go gate (Phase 0) passes.

**Goal:** Replace/augment the whisper alignment so the recurring "break/intro desync" failures stop — the intro lines don't collapse to even-spread, off-by-one cascades correct themselves faster, and post-correction lines don't stay "~6 kanas fast".

**Context (established root cause, 2026-09-02):** The break failures come from three interaction in the current `WhisperCppAligner` (`lyrics_fetcher/aligner/whisper_cpp.py`):
1. **Hallucinated/dense intro** → no confident anchors → fallback to **even-spread across the whole song** (first lines wrong by design).
2. **Off-by-one cascade** when whisper merges two sung lines into one segment — the monotonic DP locks in the shift until a later confident anchor re-syncs.
3. **Repeated chorus lines** fuzzy-matched to the wrong occurrence (e.g. first ずっと綺麗 ~44s matched to a later repeat at 152s).
4. **Coarse 5s segments + even-spread interpolation** between sparse anchors → the "~6 kanas fast after self-correction."

**Proposed approach (from the other-AI review, ranked earlier):** **stable-ts** — explicitly engineered to fix drift (cross-attention dynamic programming forces a monotonic time axis), hallucination, and repetition. It runs on openai-whisper (torch/ROCm), so it fits our existing torch stack. Risk: it swaps away from whisper.cpp/Vulkan, and torch-whisper on ROCm must be validated for stability (the reason we chose Vulkan in the first place).

---

## ⛔ Go / No-Go Gate (end of Phase 0)

Proceed to Phase 1 ONLY if a small PoC shows stable-ts **meaningfully fixes the recorded break cases on real data.** Success measure on the ASTEROID vocal stems:
- **告げよ:** first lyric lines land at ~0:27/0:30/0:33 (not even-spread), and no line stays "~6 kanas fast."
- **アンデッド:** the intro no longer collapses to even-spread; the first ~5 lines (real ~5–24s) get anchors.
- **命を振り回せ / 黒い目 / サテライト:** no off-by-one cascade after a break.
- **Repeated-chorus songs:** the first chorus occurrence maps to the earlier timestamp (not the later repeat).

If stable-ts is too unstable on ROCm (crashes, OOM, wrong) or doesn't fix the breaks, **STOP and report** — do not adopt a fragile dependency; pivot back to the native token-level anchoring (see §6.2).

---

## Phase 0 — PoC (throwaway, no pipeline change)

### Task 0.1: Install stable-ts + openai-whisper
```bash
cd ~/Code/lyrics-fetcher
uv add --index-strategy unsafe-best-match --dev stable-ts openai-whisper
```
- Verify it imports + runs on our torch/ROCm: `uv run python -c "from stable_whisper import load_model; m=load_model('medium'); print('ok')"`.
- NOTE: `load_model('medium')` downloads openai's medium (~1.5GB) to `~/.cache/whisper` — verify network. The model differs from our whisper.cpp `.bin` but same arch; fine for a PoC.

### Task 0.2: Write `poc/stablets_align.py` (throwaway)
Reuse the ASTEROID vocal stems + the corrected re-OCR lyrics (from `poc/re_ocr_asteroid.py` outputs in `_lrc_re-ocr/`). For each song:
1. Load stable-ts `load_model('medium')`.
2. `result = model.align('vocals.wav', 'known-lyrics.txt')` — the forced-alignment API (takes known text, returns word/line timestamps).
3. Extract per-line start times.
4. Compare against the current `.lrc` times on the same stem.

```python
# poc/stablets_align.py
import stable_whisper, torch
from pathlib import Path
m = stable_whisper.load_model('medium')
for stem, lyr in songs:
    result = m.align(str(stem), lyr_text)
    for line in result['segments']:
        print(result.to_dict... )  # line timing
```

### Task 0.3: Compare on 告げよ + アンデッド
- Align the corrected 告げよ / アンデッド lyrics on their vocal stems with stable-ts.
- **Success signal:** first 告げよ lines at ~0:27/0:30/0:33 (stable-ts, NOT even-spread), and アンデッド intro anchored (~5–24s).
- **Failure signal:** ROCm crash/OOM, or still even-spread / wrong.

### Task 0.4: Regression compare on an easy song
- Check ordering/time plausibility, no monotonic violation, on a song that already aligned well.
- Confirm stable-ts doesn't make an easy song worse.

### Task 0.5: **GO/NO-GO** decision
- Per the gate in the header. If GO → Phase 1. If NO-GO → update HANDOFF with the finding and fall back to native token-level anchoring (§6.2).

---

## Phase 1 — Integration (ONLY if PoC passes)

### Task 1.1: Feature branch
```bash
git checkout -b feat/stable-ts-alignment
```

### Task 1.2: New aligner `StableTSAligner`
- Create: `lyrics_fetcher/aligner/stable_ts.py` implementing the `BaseAligner` interface (same contract as `WhisperCppAligner`: `align(audio, lyrics) -> list[TimedLine]`).
- Lazy-load the whisper model; reuse corrected lyrics from OCR/fetch.
- Graceful fallback: if stable-ts errors, fall back to the existing whisper.cpp aligner (never break a run).
- Keep whisper.cpp as the **default**; stable-ts becomes `--aligner stable-ts` (opt-in), matching the existing `--aligner whisper|qwen3` pattern.

### Task 1.3: Wire CLI flag
- Add `--aligner stable-ts` to `compile`/`full`/`album`/`cross-check` in `lyrics_fetcher/cli.py` + `pipeline.py`.
- TDD: extend `tests/test_aligner.py` asserting the stable-ts path is selectable and falls back on error.

### Task 1.4: Dep handling
- Add `stable-ts` as a **core-dep only when it becomes the chosen engine**, OR keep it import-optional so CI/tests pass without the model download. Decide in Phase 1; likely keep it import-optional (lazy).

### Task 1.5: Docs
- `README.md`: document `--aligner stable-ts`, its ROCm note, and when to prefer it over whisper.cpp.
- `HANDOFF.md`: log phase-0 result, model size, ROCm stability, GO/NO-GO.

### Task 1.6: CI + PR
- Ensure `uv run pytest -q` stays green (dep import-optional). Commit, push, PR, squash-merge.

---

## Files likely to change (Phase 1)
- Create: `lyrics_fetcher/aligner/stable_ts.py`, `tests/test_stablets.py`
- Modify: `lyrics_fetcher/cli.py`, `lyrics_fetcher/pipeline.py`, `lyrics_fetcher/aligner/__init__.py`, `pyproject.toml`, `README.md`, `HANDOFF.md`

## Tests / validation
- Unit: stable-ts aligner returns `TimedLine`s; fallback-on-error; CLI flag routing.
- Integration (manual, real stems): 告げよ intro at 0:27/0:30/0:33; アンデッド intro anchored; no monotonic violations; easy song non-regression.
- CI: `pytest -q` green.

## Risks, tradeoffs, open questions
- **ROCm stability of openai-whisper:** the reason we chose whisper.cpp/Vulkan. Must validate stable-ts doesn't crash on the 9060 XT. If it does, this is a NO-GO → native token anchoring instead.
- **Model duplication:** `load_model('medium')` downloads openai's medium, separate from our whisper.cpp `.bin`; ~1.5GB disk, ~few GB VRAM. Measured in Phase 0.
- **Speed:** torch-whisper is slower than whisper.cpp(Vulkan) per segment; measure in Phase 0.

## Open questions for the user (before/at execution)
- Prefer stable-ts as an **opt-in `--aligner`** over making it the new default? (default recommendation: opt-in for now.)
- Keep whisper.cpp unchanged as default until stable-ts proven on a real album?
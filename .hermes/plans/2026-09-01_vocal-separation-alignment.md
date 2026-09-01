# Vocal-Separation Preprocessing for Alignment — Spike → Integration Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task once the spike (Phase 0) passes its go/no-go gate.

**Goal:** Preprocess the raw FLAC through a vocal-separation model to produce a clean dry-vocal stem, then feed that to the *existing* `--aligner whisper` path, so whisper stops hallucinating/drifting on accompaniment-heavy tracks (告げよ, Cryptarithm). First verify it actually helps (spike), then wire it into the pipeline only if it does.

**Architecture:** A new optional preprocessing step `audio → vocal stem (.wav)` inserted in front of alignment. The aligner itself is **untouched** — it already accepts any audio path. CLI gets a new flag (`--separation` / `--separate`, and a `separation_model` config key).

**Tech Stack (GPU-forward — test what actually works best, don't pre-reject on footprint):**
- **`demucs` (facebookresearch, `htdemucs`)** — torch-based, runs on **ROCm/GPU** (our existing torch), generally SOTA-separator quality. *Primary GPU candidate.*
- **`audio-separator` (ONNX, BS‑RoFormer / MDX‑NET)** — default CPU via `onnxruntime` (**installed, 1.29.0**), but the same onnxruntime has a **ROCm backend** so it can run on the GPU too. *Good second candidate.*
- `librosa` (**installed**), `ffmpeg` (**present**) for resample to 16k mono.
- ⚠️ VRAM contention with the vision server / whisper is a *solvable* constraint (upgrade the GPU) — per the user, it must NOT be a reason to pick a weaker engine. In the spike we run each separator on its own (vision server off / sequential), so VRAM pressure doesn't bias the quality/speed comparison.

---

## Context (verified, read-only recon)

- **Hard tracks present on NAS** with existing `.lrc` to compare against:
  - `"/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID/03 告げよ.flac"` — *both whisper medium & large‑v3‑turbo hallucinate; Qwen3‑ForcedAligner gives 0.0 (speech-trained).* This is the primary test target.
  - `"/mnt/fnos/storage/Music/maimai でらっくす グッズキャンペーンDISC -PRiSM-/05 Cryptarithm (Full).flac"` — artificial-language song, second test target.
- **whisper.cpp aligner signature:** `WhisperCppAligner.align(audio: Path, lyrics: Lyrics)` — takes any path whisper-cli can read (FLAC/MP3/WAV), so a `.wav` vocal stem drops in with zero aligner changes. Anchor logic = `_anchor_align` (monotonic DP, confidence-gated at `anchor_min_score=58`, interpolates unanchored lines). **This is the success metric:** separated stems should produce *more confident anchors* (more lines ≥ 58) and *non-even-spread* timings vs. the hallucinated even-spread fallback.
- **`onnxruntime 1.29.0`, `librosa 1.0.0`, `ffmpeg`** all present. **`audio-separator` NOT installed** — that's the one new dep the spike needs.
- **Network to huggingface.co: OK (200).** Models download on first use.
- **Config:** `config.example.toml` → add `[separation] separation_model`; resolved by precedence CLI > config > defaults in `config.py`.

---

## ⛔ Go / No‑Go Gate (end of Phase 0)

**Proceed to Phase 1 ONLY if BOTH hold:**
1. On **告げよ**, the separated-vocal alignment produces **meaningfully more confirmed anchors** than raw-FLAC alignment (e.g. new anchors span the first ~2 min that currently yields zero; or anchor count roughly doubles), AND
2. It does **not** regress an easy song (spot-check `01 アンデッド` — anchors stay comparable).

If vocal separation doesn't crack 告げよ's acoustic intro, **STOP and report** — do not wire a ~200 MB model + new dep into the pipeline for no gain. Record the negative result in HANDOFF.md as a decided trade-off.

---

## Phase 0 — Spike (throwaway, no code changes to the pipeline)

> Species a quick validation script under `poc/`. Purposely ragged; not test-covered; discarded or kept raw afterward.

### Task 0.1: Install separation tooling into the env (BOTH engines)
```bash
cd ~/Code/lyrics-fetcher
uv add --index-strategy unsafe-best-match --dev demucs audio-separator
```
- `demucs` runs on ROCm/GPU via existing torch. `audio-separator` runs ONNX (CPU by default; ROCm onnxruntime optional add).
- Verify: `uv run python -c "import demucs; from audio_separator.separator import Separator; print('ok')"`.
- **Do NOT commit pyproject yet** — spike tooling (add to `[dev]` or revert; decide at Phase 1).

### Task 0.2: Write `poc/sep_align.py` (throwaway) — engine-agnostic
A script that:
1. Takes an input audio path + an optional lyrics `.txt` + a separator choice (`--engine demucs|audio-separator`).
2. Emits a **vocals.wav**:
   - `demucs`: `demucs -n htdemucs --two-stems vocals <audio>` (uses the GPU if `torch.cuda`/ROCm visible).
   - `audio-separator`: `Separator(...)` + `--model_filename UVR-MDX-NET-Inst_HQ_3` (or `bs-roformer`), first run downloads the ONNX model from HF.
3. Reuses the existing `WhisperCppAligner` on **both** the original FLAC and the vocal stem.
4. **Reports the comparative anchor stats**: per line, similarity score; count of anchors ≥ `anchor_min_score=58`; number of unanchored/interpolated lines; whether the result fell back to even-spread (the hallucination signature). **Also reports wall-clock separation time + device used** so we can judge GPU-vs-CPU trade-offs with an eye to a possible upgrade.
5. Optionally writes a `.lrc` diff summary (where the two alignments assign each line).

```python
# poc/sep_align.py
from pathlib import Path
from lyrics_fetcher.aligner.whisper_cpp import WhisperCppAligner
from lyrics_fetcher.models import Lyrics, LyricLine

def split_vocals(audio: Path, out_dir: Path, engine: str) -> Path:
    if engine == "demucs":
        name = audio.name  # run: demucs -n htdemucs --two-stems vocals <audio>
        return out_dir / "htdemucs" / Path(audio.stem) / "vocals.wav"
    from audio_separator.separator import Separator
    sep = Separator(output_dir=str(out_dir))
    # sep.load_model('UVR-MDX-NET-Inst_HQ_3'); separate single stem 'vocals'
    return out_dir / "vocals.wav"

def anchor_stats(aligner, audio, lines):
    segs = aligner._all_segments(audio)
    anchors = aligner._anchor_align([l.text for l in lines], segs)
    anchored = [i for i, a in enumerate(anchors) if a is not None]
    return dict(total=len(lines), anchored=len(anchored), anchors=anchored,
                even_spread=(len(anchored) == 0))
```
> Note: whisper-cli may want 16 kHz mono WAV; if the separator's `.wav` isn't accepted, `ffmpeg -i vocals.wav -ar 16000 -ac 1 vocals_16k.wav` first.

### Task 0.2b: Run BOTH engines head-to-head on one track
Before the 3-song battery, run `--engine demucs` **and** `--engine audio-separator` on 告げよ to pick a primary separator. **Decision rule:** whichever yields the stronger/more-anchored vocal stem (or if both equal, prefer demucs for quality / audio-separator for CPU flexibility). Record device + time for each to inform the possible GPU upgrade.

### Task 0.3: Run the spike on 告げよ
```bash
uv run python poc/sep_align.py \
  "/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID/03 告げよ.flac" \
  --lyrics /tmp/tsugeyo.txt
```
- Capture the before/after anchor table.
- **Success signal:** the vocal stem pulls anchors out of the first ~2 min, or anchors elsewhere that were absent on raw audio.
- **Failure signal:** anchors remain ~0 → hallucination is acoustic, not accompaniment → STOP (record in HANDOFF, no Phase 1).

### Task 0.4: Run the spike on Cryptarithm
```bash
uv run python poc/sep_align.py \
  "/mnt/fnos/storage/Music/maimai でらっくす グッズキャンペーンDISC -PRiSM-/05 Cryptarithm (Full).flac" \
  --lyrics /tmp/cryptarithm.txt
```
- Same comparison. Cryptarithm is artificial-language — separation won't fix the *phoneme* problem, but it should at least remove accompaniment interference. If anchors improve on raw FLAC, good; if it's still 0 (LLM never had phonemes), that's expected and backs the future **phoneme-CTC** idea (separate plan), not this one.

### Task 0.5: Regression spot-check on an easy song
```bash
uv run python poc/sep_align.py \
  "/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID/01 アンデッド.flac" --lyrics /tmp/andead.txt
```
- Confirm separation doesn't *break* what already works (anchors roughly unchanged vs. raw).

### Task 0.6: **GO/NO-GO decision**
- Compile the three runs into a short report (anchor counts + a couple of sampled timestamps).
- Decide per the gate in the header. If GO → Phase 1. If NO-GO → update HANDOFF.md §7 with the finding; stop.

---

## Phase 1 — Integration (ONLY if spike passes)

> Now the code-quality pass: real files, TDD, a feature branch, a config key, proper docs. Mirrors the repo's established workflow (main trunk + short-lived `feat/...` PR, CI gated).

### Task 1.1: Feature branch
```bash
git checkout -b feat/vocal-separation
```

### Task 1.2: Add `separation.py` module
- Create: `lyrics_fetcher/audio/separation.py` (or under a new `audio/` package; if overkill, `lyrics_fetcher/separation.py`).
- Class `VocalSeparator` with:
  - `separate(audio: Path, out_dir: Path | None = None) -> Path` — runs the ONNX separation, returns the vocal `.wav` (16 kHz mono, resampled via `librosa`/ffmpeg so whisper-cli reads it).
  - Lazy-loaded model; cached so repeated alignments in an `album` batch don't re-download.
  - Graceful failure: if the model can't download or separation errors, **fall back to the original audio** (separation must never break alignment).
- **TDD:** write a failing test first (`tests/test_separation.py`):
  - `test_returns_wav_path` — separation of a tiny synthetic tone/wav returns a `.wav` path.
  - `test_falls_back_on_error` — with a bogus model filename, `separate()` raises → caller path uses original (assert no hard crash).
  - `test_16k_mono` — output sample rate == 16000, channels == 1.

### Task 1.3: Config plumbing
- Add `[separation] separation_model = "..."` to `config.example.toml` (default e.g. `UVR-MDX-NET-Inst_HQ_3`).
- Add the field to `config.py` defaults (constructed at load, matching the PR #3 pattern).
- **TDD:** `tests/test_config.py` — `separation_model` parses from TOML and has a sane default.

### Task 1.4: Wire into aligner/pipeline with a `--separation` CLI flag
- Add flag to `compile` / `full` / `album` in `lyrics_fetcher/cli.py`: `--separation` (bool) → before aligning, if set, run `VocalSeparator.separate(audio)` and pass the stem to the aligner.
- Add the same to `batch.py` for `album` mode (opt-in, not default — separation is slow and adds a model download).
- Wire the fallback: if separation raises, warn and pass the original audio.
- **TDD:** extend `tests/test_batch.py` / a pipeline test asserting that (a) `--separation` causes the aligner to receive the output of `separate()` (mock `VocalSeparator`), and (b) a raising separator → original path used.

### Task 1.5: Cross-check compatibility
- Confirm `--aligner qwen3` + `--separation` can compose: separate vocal stem can also go to Qwen3-ForcedAligner. If not trivial, scope separation to whisper-only in the first pass and note it.

### Task 1.6: Docs
- `README.md`: add the `--separation` flag + a `audio-separator`/model setup note under a "vocal separation (optional)" section.
- `HANDOFF.md`: log the spike results, the model path/VRAM facts, the go/no-go outcome, and that separation runs on CPU.

### Task 1.7: CI + final PR
- Keep the new dep **optional** (only import inside `separation.py` when used) so CI/tests don't require the model download. Confirm `uv run pytest -q` (currently 72 tests) stays green.
- Commit on the branch, push early, open PR, squash-merge + delete branch per the established flow.

---

## Files likely to change (Phase 1)
- Create: `lyrics_fetcher/separation.py`, `tests/test_separation.py`
- Modify: `lyrics_fetcher/cli.py`, `lyrics_fetcher/batch.py`, `lyrics_fetcher/pipeline.py`, `lyrics_fetcher/config.py`, `config.example.toml`, `pyproject.toml` (runtime dep `audio-separator`), `README.md`, `HANDOFF.md`, `tests/test_config.py`, `tests/test_batch.py`

## Tests / validation
- Unit: `audio-separator` separation → 16k mono wav; fallback-on-error; config key; `--separation` routing (mocked separator).
- Integration (manual, real models): 告げよ + Cryptarithm anchor improvement vs. raw; `01 アンデッド` non-regression; whisper + qwen3 compose.
- CI: `uv run pytest -q` green; dep made import-optional so CI needs no model download.

## Risks, tradeoffs, open questions
- **Model size/download:** stemmer ~200–400 MB (audio-separator) or ~80 MB weights (demucs htdemucs), downloaded once and cached. First run implies a download — acceptable, mirrors existing model setup.
- **Heavy dep / engine choice:** demucs (torch, already a transitive dep of the Qwen aligner) is the quality/GPU candidate; audio-separator is the lighter CPU/ONNX candidate. **Head-to-head (Task 0.2b) decides the primary engine; not pre-rejected on VRAM.**
- **GPU/VRAM (upgradeable — do not gate on it):** demucs on ROCm wants a few GB beyond what whisper leaves free on the 16 GB card. The user has signaled a GPU upgrade is on the table, so the spike runs separators standalone/sequential and judges them on **quality + speed**, then we decide footprint-vs-upgrade separately. Record each engine's device + wall-clock for that decision.
- **Speed:** demucs on GPU ~fast; audio-separator CPU-bound ONNX is a few min per 4-min track. Acceptable for an opt-in flag; report both so we know the ceiling.
- **告げよ caveat:** separation can't fix a *fundamentally acoustic* intro (no reliable phoneme anchors) — the honest expectation is it removes **accompaniment-driven** hallucination only. Real gate is the anchor-count improvement on 告げよ.
- **Open questions (resolve during spike):** which engine/model wins (demucs-htdemucs vs audio-separator + which MDX/RoFormer model); on-device ROCm vs CPU time split for a possible upgrade; whether whisper-cli needs the 16 kHz pre-conversion or accepts the raw wav.
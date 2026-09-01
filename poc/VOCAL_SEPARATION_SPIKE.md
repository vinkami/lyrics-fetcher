# Vocal-Separation Spike Results (2026-09-01)

**Conclusion: NO-GO for wiring into the pipeline.** Separation gives a modest
quality bump on accompaniment-heavy songs but (a) does NOT fix the target
failure (告げよ's acoustic intro), (b) does NOT help artificial-language tracks
at all, and (c) shifts timing on songs that already align well (regression risk).
Not worth a ~200 MB model + new dep + slower batch runs for a mixed, small gain.

## Engine tested
- **demucs `htdemucs`** on ROCm/GPU (torch backend). Separation ~15–19s per
  ~3–4 min track on this machine. Used as primary + only working candidate.
- **audio-separator (UVR-MDX-NET Inst_HQ_5)** was BLOCKED on this machine:
  `RuntimeError: operator torchvision::nms does not exist` — ROCm torch vs
  torchvision build mismatch. Did not chase; demucs is the GPU-primary engine.

## Method
`poc/sep_align.py` runs the existing `WhisperCppAligner` on the raw FLAC and on a
demucs-separated vocal stem, and compares anchor counts (whisper segments scoring
>= `anchor_min_score`=58 that become time anchors in the monotonic DP). More
anchors + tighter non-even-spread timings = better alignment. Raw results cached
in `/tmp/sep_raw_cache.json` (transient).

## Results

| Track | Raw | Separated | Read |
|---|---|---|---|
| **告げよ** (hard, ASTEROID 03) | 28/54 anchored, even-spread OFF | **32/54 anchored** | modest win |
| **Cryptarithm** (artificial lang, PRiSM 05) | 0/12 anchored, even-spread ON | 0/12, even-spread ON | no help |
| **アンデッド** (easy, ASTEROID 01) | 19/28 anchored | 19/28 anchored | same count, timings diverge |

### 告げよ (01:20–03:20 detail)
Separated alignment tightens the drift zone (raw scattered 128–217s → sep 171–203s),
and +4 lines cross the anchor threshold. **But** the first real lyric is still ~33s
on both — separation does **not** recover the acoustic BGM-dense intro. This is the
primary target and it was NOT meaningfully improved.

### Cryptarithm
Identical even-spread fallback (0 anchors) on raw and stem. Confirms our prior
model: artificial-language (Kajiurago/IPA) is a **phoneme** problem — no vocal
separator touches it. (Future fix: phoneme-level CTC, e.g. Wav2Vec2/HuBERT on
romaji with the LM prior disabled.)

### アンデッド (regression concern)
Same 19/28 anchor count but alignment shifts: first anchor 14s→18s, and L12–16
move ~+10s (raw 63–86s → sep 73–92s). Separation is **not neutral** on songs that
already work — it can silently change their timestamps. That's the main reason not
to enable it by default.

## What remains the actual fix path (unchanged)
1. **Cryptarithm / artificial language** → phoneme-CTC alignment (romaji/phoneme,
   LM off). Separator irrelevant.
2. **告げよ's intro** → genuinely acoustic (clean audio, clear phoneme anchors
   absent). Both whisper and Qwen3-ForcedAligner fail regardless; `manual` mode is
   the honest answer. Separation doesn't change this.
3. **silentblue flakiness** (separate thread) is still the higher-leverage,
   lower-risk thing to fix (#songs silently skipped).

## Files
- `poc/sep_align.py` — throwaway spike harness (kept for reference).
- pyproject/uv.lock dev-dep changes were reverted; demucs/audio-separator/audioread
  are NOT in the project deps. Re-add only if this is revisited.

## To revisit later (if GPU is upgraded / appetite increases)
- Try a larger demucs model variant and measure whether the small 告げよ gain
  grows; and whether an opt-in `--separation` flag (never default) is worth the
  regression risk on easy songs. Would need a per-song before/after check.
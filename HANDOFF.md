# PROJECT HANDOFF — lyrics-fetcher

> **Purpose of this file:** Comprehensive context so a future Hermes session can
> resume work without re-deriving everything. Written 2026-08-28 after PR #3;
> updated 2026-08-29 after the maimai album3 batch + PRs #4/#5/#6.
> This is a *working handoff*, not user-facing docs (see `README.md` for that).

---

## 1. What the project is

**lyrics-fetcher** — a Python CLI tool that fetches song lyrics (web / booklet
OCR / AI recognition) and aligns them to timestamps, producing:
- a standard **`.lrc`** file (Jellyfin / media-player compatible), and
- a **`.html`** companion with `<ruby>` furigana.

Built for a real, thorny problem: the user rips Japanese CDs, many of which are
**obscure and not in lyrics databases** (maimai game songs, less-known vocaloid,
and albums like ASTEROID by 光収容 that aren't on MusicBrainz at all). So the
tool must try many sources and fail gracefully.

**User:** "vk" (GitHub `vinkami`). Communicates via Telegram. Likes concise,
structured replies, no fluff. Prefers bullet points, facts, honest
"what's NOT covered" reporting.

---

## 2. Current state (main branch, synced with origin)

`main` merged PRs (all squash-merged, CI green, feature branch deleted):

| Commit | PR | What |
|--------|-----|------|
| `c0b9f7c` | #6 | **Reject title-only OCR blocks** — index/worldview pages of the booklet aren't lyrics |
| `33b6101` | #5 | **Multi-song booklet OCR** + skip no-lyrics by default |
| `ca8554b` | #4 | **`cross-check` mode** — compare whisper vs Qwen3 timings `--tolerance` |
| `d8c1d82` | #3 | Central **TOML config file** (`config.py`, `config.example.toml`) |
| `e27986d` | #2 | CI workflow (`.github/workflows/ci.yml`) + lockfile + tests |
| `dfd2352` | #1 | The full working project (fetch/align/output + README) |

Working tree is clean, on `main`, synced with `origin/main`. **22+ commits of
prior work were on a stale `feature/poc-scripts` branch that we merged & deleted
in PR #1.**

**Branching strategy now active:**
- `main` = stable trunk, only receives squash-merged PRs
- Short-lived `feat/...` / `fix/...` / `docs/...` branches off main
- Every PR gated on CI (which runs `uv sync` + `pytest`)
- On merge, delete the branch locally too (`git branch -D`) + `git remote prune origin`

---

## 3. Architecture & pipeline

Per song:
1. **Read metadata** from audio tags via `mutagen` (`models.SongMeta.from_path`) —
   title/artist/album/tracknumber/date, attempts MusicBrainz fields too.
2. **Fetch lyrics**, in priority order:
   - Web fetchers (each a class in `lyrics_fetcher/fetcher/`):
     - `utaten.py` — vocaloid + **furigana** (`span.ruby`), clean text by removing `.rt`
     - `genius.py` — search API + scrape `__PRELOADED_STATE__`
     - `silentblue.py` — maimai/chunithm/ongeki wiki pages (works via
       `index.php?search=` ; API + `?action=raw` are Cloudflare-blocked)
   - **OCR of booklet photo** (`ocr/vision.py`) — local Qwen3.5-9B vision LLM
   - **whisper transcription** as last resort (`fetcher/whisper.py`)
   - All fetchers do **title/artist disambiguation** (thefuzz) so wrong songs
     aren't accepted (this was a real bug: アンデッド fetched the wrong song).
3. **Align** known lyric lines to timestamps (`aligner/`):
   - `whisper_cpp.py` — whisper.cpp (Vulkan), **monotonic anchor-based DP growth**:
     only confident matches become time anchors; lines between anchors interpolate;
     if whisper hallucinates (no anchors) it spreads evenly across the song.
   - `qwen3_forced_aligner.py` — Qwen3-ForcedAligner (LLM, Japanese support), the
     `--aligner qwen3` option, independent timing source.
   - `manual_align.py` — interactive tap-a-key timer (ffplay) for songs that
     defeat automation.
4. **Write** `.lrc` + `.html` (`output/writers.py`). `--jellyfin` writes next to
   the audio file (basename match) so media players auto-pick-up.

CLI subcommands: `fetch`, `ocr`, `compile`, `full`, `album` (batch), `manual`.

---

## 4. Infrastructure / external model paths (important!)

These live outside the repo — the repo only references their **paths** (now via
the config file, `config.py` defaults / `config.example.toml`).

- **whisper.cpp** (primary aligner): `~/whisper.cpp/`
  - binary `~/whisper.cpp/build/bin/whisper-cli` (built `-DGGML_VULKAN=ON` for RDNA4)
  - models: `~/whisper.cpp/models/ggml-medium.bin` (primary),
    `ggml-large-v3-turbo.bin` (extra, merged as anchors in `album` mode)
- **Qwen vision model** (booklet OCR) on the **NAS**: `/mnt/fnos/storage/ai-models/qwen3.5-9b/`
  - `Qwen3.5-9B-Q4_K_M.gguf` (~5.7 GB) + `mmproj-BF16.gguf` (~0.9 GB, REQUIRED for vision)
  - served by **llama-server on port 8081** (user's `~/AI/start-vision` + `~/AI/vision-config.ini`).
    User's `~/AI/start` and `~/AI/config.ini` are for the 27B model — **do not modify** them; make your own.
- **Qwen3-ForcedAligner** (optional aligner) on NAS: `/mnt/fnos/storage/ai-models/qwen3-forcedaligner/model/`
  - (Qwen/Qwen3-ForcedAligner-0.6B, ~918M params, 1.8GB safetensors)
- **NAS music library**: `/mnt/fnos/storage/Music/` (FNOS upload)
  - Albums of interest: `光収容の倉庫 ASTEROID/`, `maimai でらっくす グッズキャンペーンDISC -PRiSM-/`,
    `『魔法少女ノ魔女裁判』コンプリートオリジナルサウンドトラック/` (= "まのさば"/manosaba),
    `VOCALOID 超BEST -memories-/`

**ROCm hardware:** RX 9060 XT (16GB, RDNA4, RADV GFX1200). Also has an iGPU
(Ryzen 5 9600X). An **RX 6600 XT eGPU attempt FAILED** — OCuLink→PCIe adapter on
X870M AORUS ELITE never trained its PCIe link (`DLActive-`, kernel lockdown
blocks `setpci`); not worth re-debugging unless the user asks.

---

## 5. GPU / VRAM juggling (critical know-how)

- Qwen vision model (llama-server:8081) resident uses **~7.6 GB** VRAM.
  whisper `medium` (~2.5 GB) coexists — verified both run at once.
- Qwen3-ForcedAligner needs its own VRAM; **don't load it while the vision server
  is resident on a 16 GB card** — OOMs. (Kill vision server with `pkill -9 -f "vision-config.ini"`
  first.)
- `whisper.cpp` runs on **Vulkan** (not ROCm) — that's the whole reason the
  eGPU idea surfaced; it was abandoned. Vulkan works cleanly on the 9060 XT.

---

## 6. Problems encountered & decisions made (the "why")

1. **whisper.cpp + Vulkan chosen over faster-whisper** — faster-whisper knows
   poor ROCm; whisper.cpp's standalone Vulkan binary runs cleanly on RDNA4 via
   the same Mesa path as the rest of the setup. `medium` was the accuracy sweet
   spot for synthetic vocals (better than small AND large-v3-turbo on most
   songs).
2. **Model comparison (2026-08-28):** Qwen3.5-9B (vision OCR) beat Gemma-4-12B
   (mojibake, unusable) and Qwen3.8-27B (decent but worse AND 16.4GB VRAM).
   Qwen3.5-9B = **7.6GB, more accurate**, the production OCR choice.
3. **OCR via vision LLM, not Tesseract** — real phone-photo booklets (uneven
   lighting, no scanner) defeat Tesseract; the VLM reads kanji + line breaks far
   better. Tesseract stays as CPU fallback.
4. **Known lyrics are authoritative; whisper only supplies timestamps** — even
   when whisper garbles an obscure song, keep the correct fetched text, fuzzy-match
   to segments for timing.
5. **Monotonic anchor alignment** (not greedy) — greedy mapped both identical
   repeated-chorus lines to the SAME first segment (out-of-order/duplicated).
   Non-decreasing DP keeps each chorus occurrence in correct temporal order.
6. **`album` batch + positional booklet→track fallback** — page OCR is
   nondeterministic and sometimes drops the `[title]` header; unmatched pages
   pair positionally with unmatched tracks (one-per-track, in order).
7. **Coordinates/config** — added a central TOML config (PR #3) so all model
   paths/tuning are configurable, resolved at construction time (fix a rebinding
   bug where modules captured the old singleton; fix bool-is-int coercion).
8. **Git/branching** — originally everything piled onto `feature/poc-scripts` and
   **wasn't pushed**. Established: main trunk + short-lived PR branches, squash-merge,
   `uv.lock` committed, CI gating. **Always `git push` features!**
9. **Multi-song booklet OCR (PR #5)** — pages often hold 2–3 songs plus worldview
   pages. `ocr/extract_songs(image, known_titles)` returns a dict of
   `title -> text`, split via the VLM; the album batch then matches each block to
   on-disc tracks. **No-lyrics songs (instrumentals/BGM) are skipped by default**;
   a best-effort option exists for e.g. ATLAS RUSH (sampled human "lyrics").
10. **Title-only blocks are NOT lyrics (PR #6)** — index/title pages and worldview/
    header pages come back from the VLM as single-line blocks whose content is just
    the printed title (e.g. an index page listing QUIQ/Edelweiss/Bloody Trail).
    `BookletMapper._is_title_only_block()` drops blocks (≤2 lines) whose cleaned
    text matches their own title/header → reported as unmatched, not invented lyrics.

---

## 7. What's still broken / honest limits

- ~~**告げよ (ASTEROID track 03): NO automatic aligner handles its first ~2 min.**~~
  ✅ **RESOLVED 2026-09-04** — `--aligner stable-ts --separation` anchors the
  intro (27.4/30.5/33.8s, PR #9 / §9d). Historical failure modes kept below.
  - whisper `medium` **hallucinates** ("メルエリアルリン" ×137); `large-v3-turbo`
    loops "作詞・作曲 初音ミク" then recognizes the back half (gave 24/53 anchors).
  - **Qwen3-ForcedAligner ALSO gives 0.0 for the first ~2min** (it's speech-trained;
    spec rate is "Speech", not singing/song). It's useful for the back half and as
    a cross-check, but not a fix for 告げよ's intro.
  - Audio file is **clean** (decodes fine, normal levels) — the failure is acoustic
    (dense BGM + vocals with unclear phoneme anchors), not bad data.
  - This is why `manual` (tap-a-key) mode exists.
- Minor OCR slips occasionally (dropped morphemes like 一/の). A post-OCR
  cleanup pass (Qwen re-reads) fixes grammar-detectable ones (思うは→思うのは) but
  NOT grammatically-valid single-character drops (欠片 vs 一欠片) — those need
  higher-res transcription.
- **告げよ title header** can leak as a fake lyric line in `full` mode (the
  `album` path strips it; `ocr/base.fetch` now also strips leading `[title]`).

---

## 8. Next steps (agreed priority)

- **A — `cross-check` mode:** ✅ **DONE (PR #4)**. Run **both** whisper and
  Qwen3-ForcedAligner on a song; report lines where their timestamps diverge, so
  the user can spot drifting lines (their original 黒い目 drift goal) and
  hand-fix only those. Command: `cross-check song.flac lyrics.txt --tolerance 2.5`.
- **B — process the maimai prism / manosaba albums:** ✅ **STARTED — maimai
  ベストアルバムちほー3 run in progress (first full run done; see §10).** The
  manosaba (魔法少女ノ魔女裁判) album is still pending.

Other ideas we brainstormed (lower priority): furigana for non-utaten sources
(janome/kytea), coverage reporting (flag AI-guessed vs curated lyrics), "smart
告げよ" (auto-time back half + flag intro for manual fill).

---

## 9. Session log — 2026-08-29 maimai ベストアルバムちほー3

Ran the real `album` batch on the photographed 30-page / 101-track booklet
(maimai ベストアルバムちほー3 on the NAS). The album has real **instrumentals/
BGM (skipped by design)**, **worldview pages sitting mid-booklet (need to not
leak)**, 1–3-song pages, and artificial-language songs (Cryptarithm, IF:U).

**Run 1 output (before fixes):** many vocal songs wrongly produced a single-line
`.lrc` whose only content was the printed **title header** (`[00:00.00]QUIQ`
etc.) for songs that have no lyrics. Root cause: index/title pages (e.g.
`002013` listing QUIQ/Edelweiss/Bloody Trail) and header pages (e.g. `001816`
`BUDDiES PLUS` block) came back from the VLM as single-line blocks the agg step
treated as real lyrics. → **PR #6 fix (`_is_title_only_block`).**

**Run 2 (after #6):** the 9 garbage single-line `.lrc` are gone; those songs
now **skip** correctly; index + worldview pages report as unmatched phantoms
(`CHIBI` pages, `53-6`/`518-4` page "phrases", くま pages; `002013` title-only).
**Final on-disk state: 49 `.lrc`, all with real lyrics, zero garbage.**

**Key finding — run-to-run instability (NOT the OCR logic):** the set of songs
that get lyrics wanders between runs because coverage leans on the **web
fetchers**, and **silentblue is intermittently Cloudflare-blocked** (already in
§6/§3 caveat). ~8 tracks flip to "skipped" when silentblue is down at that
moment (エスオーエス, のじゃロリック, RE-INCARNATED, RondeauX, Cryptarithm, 有明,
Ref-rain, Flashback — their good content came from web fetch in a prior run).
The VLM page-split is also nondeterministic, so OCR-attached tracks vary too.
**No data loss** (an existing `.lrc` is never clobbered by a failed fetch), but
"re-run and hope." Not yet fixed.

**Open options discussed (user hasn't picked yet):** (a) add an explicit
`--no-overwrite` guard so a run *guarantees* it never deletes/overwrites an
existing `.lrc` it couldn't fetch; (b) harden silentblue (use the
`index.php?search=` walkaround + retries/caching); (c) leave as-is.

**To resume:** the vision server is **stopped** (user reclaimed the GPU; VRAM
released to idle ~6%). Relaunch with `~/AI/start-vision qwen3.5-9b` (port 8081)
before any new OCR run.

---

## 9b. Session log — 2026-09-02 ASTEROID: vocal separation + re-OCR + alignment breaks

### What happened this session
1. **Vocal-separation spike (demucs) → NO-GO for pipeline, but a real quality win.**
   Tested `demucs htdemucs` (ROCm/GPU, ~15s/track) + `audio-separator` (BLOCKED on
   ROCm — `torchvision::nms does not exist` build mismatch). Results on ASTEROID:
   - **告げよ:** 28→32 anchors; separated stem pulled out the intro that raw even-spread
     couldn't — first lines landed at **0:27/0:30/0:33** (matches user's manual timing).
   - **Cryptarithm (artificial language):** 0/12 both raw & sep → confirms artificial
     language is a **phoneme** problem, NOT accompaniment; a separator can't fix it.
   - **アンデッド:** same anchor count but timings shifted ~10s (regression risk on
     easy songs).
   Decision: separation is a quality win (esp. dense-BGM intros) but shifts already-good
   songs — so it's a candidate for an **opt-in `--separation` flag, never default.**
   Artifacts: `poc/sep_lrc.py`, `poc/VOCAL_SEPARATION_SPIKE.md` (results), vocal stems in
   `~/Code/lyrics-fetcher/_sep_out/*_vocals.wav`. Deps NOT kept in pyproject (reverted).

2. **Re-OCR of a freshly-photographed booklet → BEST lyrics yet.**
   User re-photographed (5 photos, one per song, `20260902_*.jpg`). Re-OCR'd each with the
   vision server + re-aligned on the vocal stems (`poc/re_ocr_asteroid.py` → `_lrc_re-ocr/`).
   - **告げよ:** was skipping lines (OCR dropped then) → now continuous, intro at 0:27/0:30/0:33.
   - **命を振り回せ:** was 63 lines with **告げよ lyrics mixed in** → now clean 38 lines.
   - These are now deployed live (album folder); backups were removed per user.

3. **Root cause of the remaining "breaks"/desync (from `-ojf` token data):**
   - Intro hallucination → 0 confident anchors → even-spread (first lines wrong).
   - whisper merges 2 lines into 1 segment → monotonic DP goes off-by-one until a later
     anchor re-syncs ("all lines one late until a break").
   - Repeated chorus lines fuzzy-match to the wrong (later) occurrence.
   - Coarse 5s segments + even-spread between sparse anchors → "~6 kanas fast" drift after
     self-correction.

4. **Two-column OCR finding:** the VLM (Qwen3.5-9B) **fails to read 2-column booklets**
   correctly — 命を振り回せ's two-column layout was mis-read (skips to the 2nd column's
   tail after the 1st column). **告げよ** also has 2 columns and *succeeded* → success is
   non-deterministic. This is a known OCR limitation to address (multi-column reading).

### Priorities going forward
- **Alignment "breaks" (most annoying):** ✅ **DONE** — stable-ts PoC → GO →
  shipped as `--aligner stable-ts` (PR #9, §9d). Fixes intro even-spread,
  off-by-one cascades, repeated-chorus mis-matching, post-correction drift.
- **Two-column OCR** is a real gap (below stable-ts priority).
- ✅ **Vocal separation SHIPPED** (see §9c below) as an opt-in `--separation` flag.

---

## 9c. Session log — 2026-09-02 (cont.): vocal separation merged to main

User decided the vocal-separation win is worth keeping → merged into `main`.

**Feature (`lyrics_fetcher/separation.py` + `--separation` flag):**
- `VocalSeparator` wraps **demucs `htdemucs`** (ROCm/GPU), writing a dry-vocal
  `.wav` stem.
- `Pipeline._align` separates the audio to a stem before alignment when a
  separator is set; **falls back to raw audio on any separator error** (never
  breaks a run).
- `--separation` flag wired into `full` and `album` commands.
- **Opt-in, not default** — separation can shift already-good songs by ~seconds.

**Dependency handling (`pyproject.toml`):**
- demucs is in the **dev dependency group** (`demucs==4.1.0`). Added a Linux-only
  `[tool.uv] environments` restriction: demucs pins `numpy<2` on **macOS** only,
  which conflicts with this project's `numpy>=2.5.2` in uv's universal lock. Since
  this project only runs on Linux/ROCm (see Hardware), restricting uv resolution
  to linux resolves the lock cleanly (demucs 4.1.0 + numpy 2.5.2 coexist fine).
- `--separation` **degrades gracefully** if demucs isn't installed (warns + aligns
  raw audio), so CI/tests pass without the model download.

**Tests:** added `tests/test_separation.py` (3 tests: separator-absent→None,
FakeSep used by Pipeline, FakeSep error→fallback). Suite now **75 passed**.

---

## 9d. Session log — 2026-09-04: stable-ts Phase 0 gate = GO + Phase 1 shipped

**Phase 0 (validated, evidence: `poc/stablets_align.py`, `poc/stablets_vram.py`,
`poc/out/stablets_results.json`):** stable-ts 2.19.1 + openai-whisper run on our
torch 2.11.0+rocm7.2 (RX 9060 XT = dev `cuda:0`); medium model ~5s load, 6–36s
align per ASTEROID song, **3.23 GiB peak VRAM**, **0 monotonic violations** on
every song. Gate results per song:
- **告げよ:** intro anchored at **27.4 / 30.5 / 33.8s** (the even-spread failure
  mode fixed — starts come from audio, not interpolation).
- **アンデッド:** intro anchored at **14.3s**; first **ずっと綺麗…** line at
  **43.9s** (correct occurrence — whisper.cpp's fuzzy DP had matched the later
  repeat at 152s).
- 命を振り回せ / 黒い目 / サテライト: sane starts, 0 monotonic violations; the
  >2s-vs-current deltas are stable-ts *correcting* the old interpolated times.

**ROCm finding:** no patching needed — `stable_whisper.load_model(device="cuda")`
works out of the box on RDNA4. **eGPU note:** user wired the RX 6600 XT for the
vision server; stable-ts must stay on the 9060 XT — the eGPU is torch device
index 2 and **HANGS on first compute** under ROCm. Do NOT target it.

**Phase 1 (this session):** shipped as an opt-in aligner, same graceful-degradation
contract as `--separation`:
- `lyrics_fetcher/aligner/stable_ts.py` — `StableTSAligner` (name `stable-ts`):
  ported the validated PoC algorithm (`regroup="p"`, word-timestamp flatten +
  greedy per-line char-count assignment, whitespace-normalized) + monotonic
  clamp. `stable_whisper` imported LAZILY inside `_load` (dev dep, like demucs);
  any failure (missing lib / OOM / download) warns on stderr and falls back to
  `WhisperCppAligner` — built lazily only when needed.
- CLI: `--aligner stable-ts` on `compile`/`full`/`album` (lazy-import branch in
  `_make_aligner`). `cross-check` intentionally unchanged (still whisper+qwen3).
- Config: `[stable_ts]` section (`stable_ts_model`=medium, `stable_ts_lang`=ja,
  `stable_ts_device`=cuda) in `config.py` + `config.example.toml`. All three are
  plain strings and intentionally excluded from `_coerce()`'s Path-coercion branch
  (which triggers on Path-typed field defaults): they are a whisper size name, a
  language code, and a torch device string — never filesystem paths.
- Deps: `stable-ts` + `openai-whisper` **added to the dev group on this
  branch** (`uv sync --dev`; a plain `uv sync` already includes dev, so CI
  installs them too — the lazy import is for `--no-dev` / production installs).
  Phase-0 evidence `poc/out/stablets_results.json` is deliberately tracked
  (first tracked file in `poc/out`).
- **Tests:** `tests/test_stablets.py` — 24 tests (fake-result `_line_times`
  units: exact/whitespace/repeated-line/overshoot/exhausted, NFKC+punctuation
  parity, frozen-tail coverage guard, empty/punct-only line hold-previous;
  lazy-import contract; load & align failure → fallback; monotonic clamp; CLI
  routing + fallback flag passthrough; settings resolution). Suite now
  **99 passed**; the tests are pure logic, so CI passes on a runner with no
  GPU or model download (CI does install the dev group). README: Stable-TS
  section + engine-selection + config-key updates.
- **2-stage subagent review findings that shaped the final code:** the
  `regroup="p"` + word-stream-coverage guard (`_InsufficientCoverage` → same
  fallback path; ≥4 unanchored trailing lines is truncation, ≤3 is a legit
  silent outro), `_norm` = NFKC + `WhisperCppAligner._clean`'s class (a ±1-char
  count error per line cascades as drift), and empty/punctuation-only lyric
  lines must HOLD the previous start (a `None` escapes the clamp outside the
  try → uncaught TypeError; utaten emits blank lines via bare splitlines).
- **MERGED as PR #9 (`7d15816`), CI green.** Post-merge end-to-end trial via
  `lyrics-fetcher compile <stem> <lyrics.txt> --aligner stable-ts` on all 5
  ASTEROID songs (`out/stablets_trial/`): first-3 告げよ = 27.36/30.54/33.76
  (reproduces the gate exactly), 0 monotonic violations on all 5. 命を振り回せ
  ends its last 12 lines ~140–144s (deployed says 182–206s) — plausible but
  UNVERIFIED by ear; that song's deployed .lrc is the known-bad one, so
  listen before deploying either.

---

## 10. Commands cheat-sheet

```bash
# local dev
cd ~/Code/lyrics-fetcher
uv sync --index-strategy unsafe-best-match   # ROCm torch + git-main transformers
uv run pytest -q                              # 72 tests
uv run lyrics-fetcher --help

# typical use
uv run lyrics-fetcher full "/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID/01 アンデッド.flac" \
    --image ".../booklet/20260828_060250.jpg" --jellyfin --aligner whisper
uv run lyrics-fetcher album "/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID" --jellyfin
uv run lyrics-fetcher manual song.flac lyrics.txt -o out.lrc
uv run lyrics-fetcher cross-check song.flac lyrics.txt --tolerance 2.5

# branch + PR flow
git checkout -b feat/xxx && git add -A && git commit -m "feat: ..."
git push -u origin feat/xxx && gh pr create --base main --head feat/xxx --title "..."
gh pr merge <n> --squash --delete-branch
git checkout main && git pull origin main && git branch -D feat/xxx && git remote prune origin
```

**Deps worth noting:** `uv sync` needs `--index-strategy unsafe-best-match`
(torch/torchaudio from the ROCm wheel index at `download.pytorch.org/whl/rocm7.2`,
transformers from git main for `qwen3_asr`). The `qwen-asr` PyPI package builds
the WRONG 12.7B architecture and OOMs — avoid it; use `AutoModelForTokenClassification`.
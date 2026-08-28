# PROJECT HANDOFF — lyrics-fetcher

> **Purpose of this file:** Comprehensive context so a future Hermes session can
> resume work without re-deriving everything. Written 2026-08-28 after PR #3.
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

`main` has 3 merged PRs (all squash-merged, CI green):

| Commit | PR | What |
|--------|-----|------|
| `d8c1d82` | #3 | Central **TOML config file** (`config.py`, `config.example.toml`) |
| `e27986d` | #2 | CI workflow (`.github/workflows/ci.yml`) + lockfile + tests |
| `dfd2352` | #1 | The full working project (fetch/align/output + README) |

Working tree is clean, on `main`, synced with `origin/main`. **22+ commits of
prior work were on a stale `feature/poc-scripts` branch that we merged & deleted
in PR #1.**

**Branching strategy now active:**
- `main` = stable trunk, only receives squash-merged PRs
- Short-lived `feat/...` / `fix/...` / `docs/...` branches off main
- Every PR gated on CI (which runs `uv sync` + `pytest`, 33 tests)
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

---

## 7. What's still broken / honest limits

- **告げよ (ASTEROID track 03): NO automatic aligner handles its first ~2 min.**
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

## 8. Next steps (agreed priority, NOT yet started)

The user explicitly wanted these two, in this order:

- **A — `cross-check` mode:** run **both** whisper and Qwen3-ForcedAligner on a
  song; report lines where their timestamps diverge, so the user can spot
  drifting lines (their original 黒い目 drift goal) and hand-fix only those.
- **B — process the maimai prism / manosaba albums:** run the real `album` batch
  on the photographed albums (they're the hard cases: artificial-language songs
  like Cryptarithm, まのさば's JP translations). Surfaces real bugs/quality issues.

Other ideas we brainstormed (lower priority): furigana for non-utaten sources
(janome/kytea), coverage reporting (flag AI-guessed vs curated lyrics), "smart
告げよ" (auto-time back half + flag intro for manual fill).

---

## 9. Commands cheat-sheet

```bash
# local dev
cd ~/Code/lyrics-fetcher
uv sync --index-strategy unsafe-best-match   # ROCm torch + git-main transformers
uv run pytest -q                              # 33 tests
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
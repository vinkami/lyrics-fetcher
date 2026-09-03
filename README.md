# lyrics-fetcher

Fetch song lyrics (from the web, album booklets, or AI) and align them to
timestamps so you can host `.lrc` + a furigana `.html` companion next to your
music for Jellyfin / media players.

Built around real-world needs: many songs are obscure (maimai game tracks, less
popular vocaloid, albums like ASTEROID that aren't on MusicBrainz), so the
project tries multiple sources in order and falls back gracefully.

---

## What it does

For each song it:

1. **Reads metadata** from the audio file tags (title, artist, album).
2. **Fetches lyrics** from, in order:
   - **Web fetchers**: [utaten](https://utaten.com) (vocaloid + furigana),
     [Genius](https://genius.com), [SilentBlue.RemyWiki](https://silentblue.remywiki.com)
     (maimai/chunithm/ongeki rhythm-game songs). Each is a class under
     `lyrics_fetcher/fetcher/`, with song-match disambiguation (a requested
     title/artist is verified against each candidate before accepting).
   - **OCR of a booklet photo** (if you provide one): a local vision LLM
     (Qwen3.5-9B) reads the printed lyrics. Good for albums not on any database.
   - **AI recognition** (last resort): whisper transcribes the audio directly.
3. **Aligns** the known lyric lines to timestamps using:
   - **whisper.cpp** (Vulkan on the RX 9060 XT, medium model) with monotonic
     anchor-based DP — interpolates between confident matches.
   - **stable-ts** (opt-in `--aligner stable-ts`): forces the known lyrics
     through whisper with word-level timestamps — fixes the anchor-DP failure
     modes (intro even-spread, off-by-one cascades, ~6-kana drift); any
     failure falls back to whisper.cpp automatically.
   - **Qwen3-ForcedAligner** (optional LLM aligner, Japanese support) as a
     second, independent timing source.
   - **`manual`** — tap-a-key timing by ear for songs automation can't handle.
4. **Writes** a standard `.lrc` (Jellyfin-compatible) and a `.html` companion
   with `<ruby>` furigana.

---

## Project layout

```
lyrics-fetcher/
├── lyrics_fetcher/
│   ├── cli.py                  # CLI (fetch/ocr/compile/cross-check/full/album/manual)
│   ├── pipeline.py             # end-to-end: fetch/OCR -> align -> write
│   ├── batch.py                # album batch + booklet->track mapper
│   ├── crosscheck.py           # compare whisper vs Qwen3 timings (cross-check)
│   ├── cache.py                # SQLite lyrics + OCR cache
│   ├── models.py               # SongMeta, LyricLine, Lyrics
│   ├── utils.py                # HTTP session, title/artist matching, paths
│   ├── manual_align.py         # interactive tap-a-key timer
│   ├── fetcher/                # each source = one class
│   │   ├── base.py  utaten.py  genius.py  silentblue.py  whisper.py  orchestrator.py
│   ├── ocr/                    # booklet OCR as a lyrics source
│   │   ├── base.py  vision.py (Qwen3.5-9B VLM)  tesseract.py (CPU fallback)
│   ├── aligner/
│   │   ├── base.py  whisper_cpp.py  faster_whisper_aligner.py  qwen3_forced_aligner.py  stable_ts.py
│   └── output/writers.py       # LRC + HTML(furigana) writers
├── poc/                        # proof-of-concept scripts (single-song demos)
├── PLAN.md                     # design decisions & evolution notes
└── pyproject.toml              # uv project + dependencies (incl. ROCm torch)
```

---

## Setup

Requires: **Python 3.12**, **uv** (package manager), a system with the needed
GPU/CPU bits below. This README documents the canonical setup used by the author
(RX 9060 XT, ROCm).

### 1. Clone & install Python deps

```bash
git clone git@github.com:vinkami/lyrics-fetcher.git   # or your remote
cd lyrics-fetcher
uv sync
```

`uv sync` reads `pyproject.toml`. Note two special dependency sources that
`uv sync` handles:

### 1b. Configuration file (optional)

Settings are resolved by precedence: **CLI arguments > config file > defaults**.
You can skip this — built-in defaults work out of the box — but a config file
lets you point the tool at your models/library without repeating flags.

Copy the template and edit:
```bash
cp config.example.toml ~/.config/lyrics-fetcher/config.toml   # or ./config.toml
```

Key sections:
- `[paths]` — `music_dir`, `cache_db`, `out_dir`
- `[whisper]` — `whisper_bin`, `whisper_model`, `whisper_extra_models`,
  `whisper_lang`, `whisper_max_len`, `whisper_device`
- `[vision]` — `vision_api`, `vision_model` (the local llama-server for OCR)
- `[qwen3_aligner]` — `qwen3_aligner_model`, `qwen3_aligner_language`
- `[stable_ts]` — `stable_ts_model`, `stable_ts_lang`, `stable_ts_device`
- `[output]` — `lrc_by`, `jellyfin_default`, `write_html_default`
- `[tuning]` — `anchor_min_score`, `request_timeout`

It's auto-detected from `$LF_CONFIG`, `$XDG_CONFIG_HOME/lyrics-fetcher/`, or
`./config.toml`. To use an explicit file:
```bash
lyrics-fetcher full song.flac --config /path/to/config.toml
```

- **torch / torchaudio** come from the **PyTorch ROCm wheel index**, not PyPI:
  ```ini
  [[tool.uv.index]]
  url = "https://download.pytorch.org/whl/rocm7.2"
  ```
  (This is why torch isn't on PyPI — AMD ships ROCm wheels separately.) uv
  uses `--index-strategy unsafe-best-match` so non-torch deps still resolve
  from PyPI.
- **transformers** is pinned to the **git main** branch, needed for the
  `qwen3_asr` architecture used by Qwen3-ForcedAligner:
  ```ini
  [tool.uv.sources]
  transformers = { git = "https://github.com/huggingface/transformers.git" }
  ```

If you add/remove deps, re-run:
```bash
uv sync --index-strategy unsafe-best-match
```

### 2. `whisper.cpp` (the primary aligner)

Build with the Vulkan backend so it runs on the AMD GPU (it is a separate C++
project, not a pip package):

```bash
git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j
```

Download the models it uses (paths are defaults in
`lyrics_fetcher/aligner/whisper_cpp.py`):
```bash
cd ~/whisper.cpp
./models/download-ggml-model.sh medium          # primary: ggml-medium.bin
./models/download-ggml-model.sh large-v3-turbo  # extra for hallucinating songs
```

Required model files:
- `~/whisper.cpp/models/ggml-medium.bin`
- `~/whisper.cpp/models/ggml-large-v3-turbo.bin` (used by `album` mode to merge
  a second opinion)

Check it works:
```bash
~/whisper.cpp/build/bin/whisper-cli -m ~/whisper.cpp/models/ggml-medium.bin -l ja -f some_song.flac
```

### 3. Qwen3.5-9B vision model (booklet OCR)

Needed for the `ocr` / `full --image` paths. It runs via a **llama-server**
(ROCm) that you keep running on port **8081**.

1. Put the model on the NAS (or wherever you keep models):
   ```bash
   mkdir -p /mnt/fnos/storage/ai-models/qwen3.5-9b
   # download from huggingface.co/unsloth/Qwen3.5-9B-GGUF:
   #   Qwen3.5-9B-Q4_K_M.gguf   (~5.7 GB)
   #   mmproj-BF16.gguf         (~0.9 GB, required for vision)
   ```
   (Store models on a drive with room — the Qwen vision model is ~6.6 GB, the
   forced aligner ~1.8 GB.)

2. Create a start script (`~/AI/start-vision`) and config (`~/AI/vision-config.ini`).
   The scripts in `~/AI` are the author's own; the repo expects a vision server
   at `127.0.0.1:8081`. A minimal start:
   ```bash
   # ~/AI/start-vision
   /home/vinkami/AI/llama-b10647-rocm/llama-server \
     --models-preset /home/vinkami/AI/vision-config.ini \
     --alias qwen3.5-9b --host 127.0.0.1 --port 8081
   ```
   The `vision-config.ini` declares the model + `dev = ROCm0` + a modest
   context (`ctx-size = 16384`) so ~7.6 GB of VRAM is used, leaving room for
   whisper to share the 16 GB card.

3. Start it:
   ```bash
   ~/AI/start-vision qwen3.5-9b
   ```

> **Model-download note (SMB/NAS):** `huggingface-cli download` fails with
> `PermissionError ... .lock` on some SMB mounts (file-lock not supported).
> If that happens, use plain `wget -c` on the GGUF files directly.

### 4. Qwen3-ForcedAligner (optional second aligner)

Supports Japanese + ~10 languages, LLM-based forced alignment of *known* text.
It is the `--aligner qwen3` option and a useful independent timing source
(compare against whisper to spot drifting lines).

1. Download the model to the NAS:
   ```bash
   mkdir -p /mnt/fnos/storage/ai-models/qwen3-forcedaligner
   huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B-hf \
     --local-dir /mnt/fnos/storage/ai-models/qwen3-forcedaligner/model
   ```
   (If the lock issue above bites, wget the 8 files from
   `https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf/resolve/main/...`.)

2. Its deps (torch ROCm, transformers git-main, nagisa, librosa) are already in
   `pyproject.toml` — `uv sync` installs them.

3. The aligner class defaults to that model path
   (`lyrics_fetcher/aligner/qwen3_forced_aligner.py::LOCAL_MODEL`), so it's
   ready to use once `uv sync` + the model download are done.

> **Why git-main transformers & ROCm torch?** The `qwen3_asr` architecture the
> aligner needs is only in transformers' dev/main (the 4.57.6 PyPI pin predates
> it), and the torch wheels for AMD only exist on the ROCm index. Trying the
> `qwen-asr` pip package instead instantiates the wrong 12.7B model and OOMs;
> use `AutoModelForTokenClassification` (see the aligner).

### 5. `ffplay` (for the `manual` aligner)

Part of ffmpeg — `sudo apt install ffmpeg`. Needed only for the interactive
manual timing command.

---

## Usage

The CLI is installed as `lyrics-fetcher` (or `uv run lyrics-fetcher ...`).

```bash
# Fetch lyrics for a song (tries all sources, shows which matched)
lyrics-fetcher fetch "天ノ弱" --artist "164"

# Fetch from one source
lyrics-fetcher fetch "Song Title" --source utaten

# OCR a booklet photo (needs the Qwen vision server on :8081)
lyrics-fetcher ocr path/to/booklet.jpg --engine vlm

# Compile: align known lyrics text against an audio file -> .lrc
lyrics-fetcher compile song.flac lyrics.txt -o song.lrc \
    --aligner whisper          # or --aligner qwen3 / stable-ts

# Full pipeline for one song (fetch/OCR -> align -> .lrc + .html)
# --jellyfin writes the output next to the song file (media-player layout)
lyrics-fetcher full song.flac --image booklet.jpg --jellyfin

# Batch an entire album: auto-map booklet pages -> tracks, align all
lyrics-fetcher album "/music/Album Name" --jellyfin

# Optionally separate a dry-vocal stem (demucs) before aligning — improves
# intro timing on BGM-dense songs that make whisper hallucinate/even-spread.
# Needs the `separation` dev extra installed (see setup). Slower.
lyrics-fetcher full song.flac --image booklet.jpg --jellyfin --separation

# Manually time lyrics by tapping a key per line (for songs that defeat AI)
lyrics-fetcher manual song.flac lyrics.txt -o song.lrc

# Cross-check: run whisper AND Qwen3, flag lines where timings diverge
# (e.g. > 2.5s apart) so you can hand-fix only the drifting lines
lyrics-fetcher cross-check song.flac lyrics.txt --tolerance 2.5
lyrics-fetcher cross-check song.flac lyrics.txt -v   # show every line
```

### Alignment engine selection
`--aligner whisper` (default) uses whisper.cpp with anchor-based DP; `album`
mode additionally merges large-v3-turbo.
`--aligner qwen3` uses Qwen3-ForcedAligner (Japanese-capable, independent).
`--aligner stable-ts` uses stable-whisper forced alignment (see below).

### Vocal separation (`--separation`)
`full` and `album` accept `--separation` to preprocess each audio file through
**demucs** (`htdemucs`) and align against the clean dry-vocal stem instead of the
full mix. This markedly improves intro/timing on **BGM-dense** songs — e.g.
ASTEROID's 告げよ intro landed at 0:27/0:30/0:33 only on the separated stem,
where raw whisper collapsed to even-spread. It is **opt-in, not default**, because
separation can shift already-good songs by a few seconds.

Install the optional extra (it is kept out of the default dependency graph because
demucs pins numpy<2 on macOS, which conflicts with this project's numpy>=2.5.2 in
uv's universal lock — the project targets Linux only):
```bash
uv sync --dev   # demucs is a dev dependency; installs on the linux/ROCm env
```
If demucs isn't installed, `--separation` degrades gracefully (warns and aligns
the raw audio).

### Stable-TS alignment (opt-in, `--aligner stable-ts`)
**stable-ts** (stabilize-whisper) forces the *known* lyrics through whisper and
gets word-level timestamps back, so every line start comes from the audio —
no anchors, no interpolation. It fixes the three failure modes of the
whisper.cpp anchor DP: **intro even-spread** (whisper hallucinates → zero
confident anchors), **off-by-one cascades** (two lyric lines merged into one
segment), and **~6-kana drift** after a resync. Validated on all 5 ASTEROID
songs: 告げよ's intro anchored at 0:27/0:30/0:34, アンデッド's intro fixed, and
repeated chorus lines kept on their correct occurrences (0 monotonic
violations across the album).

```bash
lyrics-fetcher full song.flac --image booklet.jpg --jellyfin --aligner stable-ts
```

- stable-ts is a **dev extra** like demucs (kept out of default deps for the
  same Linux-only lock reason): install with `uv sync --dev`.
- Runs on the RX 9060 XT via torch ROCm (`stable_ts_device = "cuda"`);
  `medium` peaks at **~3.2 GiB VRAM**, so it coexists with whisper.cpp's
  models and the eGPU-hosted vision server. Alignment takes ~6–36 s/song.
  Do NOT point it at the eGPU (the RX 6600 XT hosting the vision server is
  torch device index 2 and hangs on first compute).
- **Any failure — missing lib, OOM, model-download error — warns and falls
  back to whisper.cpp**, so opting in can never break a run.

### Cross-check mode
Whisper (Vulkan) and Qwen3-ForcedAligner are **independent** timing sources.
`cross-check` runs both on a song and reports every line whose two start times
differ by more than `--tolerance` seconds (default 2.5), so you can spot lines
where automatic alignment drifted from the true timing and fix just those with
`manual`. Exits non-zero when any line drifted or is missing (handy for
scripts). Whisper runs first as a subprocess so its video memory is freed before
Qwen3 loads in-process — the two don't need to be resident at once.

### Manual alignment controls
After the "GO" countdown, press **RETURN** each time a line starts:
- `b` = go back and re-mark the previous line
- `s` = skip the current line (set 0.0)
- `q` = quit and write the `.lrc` so far

---

## Notable design decisions (why things are the way they are)

- **whisper.cpp + Vulkan, not faster-whisper**: whisper.cpp builds a standalone
  Vulkan binary that runs cleanly on the RDNA4 (RADV GFX1200) GPU with the same
  Mesa path as the rest of the setup. `medium` was the accuracy sweet spot for
  synthetic vocals (better than small and large-v3-turbo on most songs).
- **Booklet OCR via a vision LLM, not Tesseract**: on real phone-photo booklets
  (uneven lighting, no scanner), the Qwen3.5-9B vision model is far more
  accurate than Tesseract at reading Japanese kanji + line breaks. Tesseract
  stays as a CPU fallback.
- **Known lyrics are authoritative; whisper only supplies timestamps**: even when
  whisper garbles an obscure song, the correct fetched text is kept and fuzzy-
  matched to segments for timing.
- **Monotonic anchor alignment**: lyric lines are only treated as time anchors
  when whisper's match is confident; lines between anchors are interpolated. If
  whisper hallucinates (no anchors at all), lines spread evenly across the song
  duration instead of collapsing.
- **Some songs defeat automation entirely** (e.g. ASTEROID's 告げよ — dense BGM
  intro that both whisper and Qwen3-ForcedAligner fail to time). That is why the
  `manual` subcommand exists: tap line starts by ear, then hand-edit.

### Vocal separation & known alignment limits
- A **vocal-separation spike** (demucs `htdemucs`) showed separation **clearly
  improves intro timing** on dense-BGM songs (告げよ's intro landed at 0:27/0:30/0:33
  on the separated stem) and is a candidate **opt-in `--separation` flag** — but it
  also shifts already-good songs, so it's **not default**. See `poc/VOCAL_SEPARATION_SPIKE.md`.
- Remaining **"break"/desync** (intro even-spread, off-by-one cascades, repeated-
  chorus mis-matching, post-drift) are a known limitation of whisper-based anchor
  alignment. **stable-ts forced alignment ships as the opt-in
  `--aligner stable-ts`** (Phase 0 gate passed 2026-09-04; see the Stable-TS
  section above and `poc/stablets_align.py`).
- **Two-column booklets:** the VLM OCR (Qwen3.5-9B) sometimes **mis-reads 2-column
  layouts** (skips to the second column's tail). This is non-deterministic (告げよ
  succeeds, 命を振り回せ failed once) and a known OCR gap.

---

## Troubleshooting

- **`uv sync` torch resolution conflict** — run with
  `uv sync --index-strategy unsafe-best-match` so PyPI supplies non-torch deps.
- **Qwen vision model OOM alongside whisper** — vision model uses ~7.6 GB;
  whisper `medium` fits (~2.5 GB) on a 16 GB card. If you use a bigger vision
  model or a big LLM in LM Studio at the same time, stop one before the other.
- **`huggingface-cli` lock error on NAS** — use `wget -c` for the GGUF files.
- **Qwen3ForcedAligner OOM** — make sure you're not loading it while the vision
  server (or another big model) holds VRAM; they don't share a 16 GB card in the
  same run.
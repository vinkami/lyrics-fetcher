# lyrics-fetcher

Fetch song lyrics — from the web, from album-booklet photos, or by AI
transcription — and **align them to timestamps**, so you can keep
Jellyfin-compatible `.lrc` files (plus a furigana `.html` companion) right
next to your music.

Built for songs that aren't in any lyrics database: rhythm-game tracks,
obscure vocaloid, imports with booklet-only lyrics. It tries multiple
sources in order, aligns the *known text* to the audio (the text is always
authoritative — the aligner only supplies timestamps), and falls back
gracefully at every step.

```bash
lyrics-fetcher album "/music/Album Name" --jellyfin --aligner stable-ts
lyrics-fetcher full song.flac --image booklet-page.jpg --jellyfin
```

- [Setup](#setup) · [Quick start](#quick-start) · [Configuration](#configuration)
- [Full CLI reference → `docs/CLI.md`](docs/CLI.md)
- [Every config key & env var → `docs/CONFIG.md`](docs/CONFIG.md)

---

## How it works

For each song:

1. **Metadata** — reads title/artist/album from the audio file's tags.
2. **Lyrics text**, tried in order until one hits:
   - **Web sources** — [utaten.com](https://utaten.com) (Japanese, with
     furigana readings), [Genius](https://genius.com), SilentBlue/RemyWiki
     (rhythm-game song databases). Every candidate is title/artist
     disambiguated before being accepted, so a wrong same-titled song can't
     slip through.
   - **Booklet OCR** — photograph the lyric booklet page and any
     **OpenAI-compatible vision model** transcribes it (see
     [Configuration](#configuration)). Handles multi-song pages: the model
     returns structured `{title: lyrics}` JSON, split per track and matched
     against the actual disc contents.
   - **Whisper transcription** (last resort, `--whisper-fallback`) — gets
     *searchable text* from the audio itself; alignment never trusts it
     over real lyrics.
   - Results are cached (SQLite): re-runs don't re-fetch or re-OCR.
3. **Alignment** — known lyric lines are fit to the audio with one of:
   - **whisper.cpp** (default) — GPU transcription + monotonic anchor DP.
   - **stable-ts** (`--aligner stable-ts`) — word-level forced alignment of
     the known text; fixes even-spread intros, off-by-one cascades and
     drift; any failure automatically falls back to whisper.cpp.
   - **Qwen3-ForcedAligner** (`--aligner qwen3`) — an independent LLM
     timing source; also powers `cross-check`.
   - **`manual`** — tap line starts by ear for songs that defeat everything.
4. **Output** — `.lrc` (standard enhanced-timestamp format; Jellyfin/MPV/
   etc.) next to or beside the audio, and a `.html` companion with
   `<ruby>` furigana and synchronized highlighting.

---

## Requirements

- **Linux** (the dependency lock targets Linux; other platforms need a
  manual `pyproject.toml` edit — see [Setup](#setup)).
- **Python ≥ 3.12** and **[uv](https://docs.astral.sh/uv/)**.
- An **NVIDIA or AMD GPU** is strongly recommended (whisper.cpp runs on
  Vulkan which covers both; the optional stable-ts/demucs paths run on
  CPU too, just slowly).
- **ffmpeg** (for audio handling and the `manual` command's playback).
- A **vision-capable AI model** reachable via an OpenAI-compatible API —
  any cloud provider (OpenAI, OpenRouter, Groq, …) or a self-hosted
  server (llama.cpp, vLLM, Ollama) that exposes `/v1/chat/completions`.
  This is only needed for booklet OCR (`ocr`, `full --image`, `album`).

## Setup

```bash
git clone <this repo> && cd lyrics-fetcher
uv sync
```

`uv sync` installs everything, including the heavy ML stack. Two notes on
how the lock is configured:

- **torch/torchaudio** resolve from the **PyTorch ROCm wheel index** by
  default (the project's reference hardware is an AMD GPU):
  `[[tool.uv.index]] url = "https://download.pytorch.org/whl/rocm7.2"`.
  For NVIDIA, change it to `https://download.pytorch.org/whl/cu126` (and
  the pins accordingly); for CPU, plain PyPI wheels work. On non-Linux
  you must also drop the `environments = ["sys_platform == 'linux'"]`
  restriction at the bottom of `pyproject.toml`.
- **transformers** is pulled from git main — the Qwen3-ForcedAligner
  architecture is newer than the last PyPI release. Not needed unless you
  use `--aligner qwen3` / `cross-check`.
- **stable-ts, openai-whisper and demucs live in the *dev* group** (they
  constrain torch/numpy in ways the universal lock can't satisfy on all
  platforms). A plain `uv sync` installs them already; `uv sync --dev` is
  a no-op in that sense. Without them, `--aligner stable-ts` /
  `--separation` degrade gracefully.

Verify the install:

```bash
uv run lyrics-fetcher --help
uv run pytest -q          # 109 tests, no GPU or network needed
```

### whisper.cpp (the default aligner)

A separate C++ project — build once:

```bash
git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_VULKAN=ON       # Vulkan works on AMD + most NVIDIA
cmake --build build --config Release -j
./models/download-ggml-model.sh medium          # primary
./models/download-ggml-model.sh large-v3-turbo  # optional extra anchors
```

These exact paths (`~/whisper.cpp/build/bin/whisper-cli`,
`~/whisper.cpp/models/ggml-medium.bin`) are the defaults; point
`[whisper]` in your config at them if you installed elsewhere.

### Optional: Qwen3-ForcedAligner

Only needed for `--aligner qwen3` and `cross-check`:

```bash
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B-hf \
  --local-dir ~/.cache/lyrics-fetcher/models/qwen3-forcedaligner
```

That location is the default; otherwise set
`[qwen3_aligner] qwen3_aligner_model` to wherever you put it.

### Optional: vocal separation

`--separation` runs **demucs** (`htdemucs`) to align against a dry-vocal
stem — markedly better intro timing on BGM-dense songs. Installed by the
dev group above; if it's missing, the flag warns and aligns raw audio.
First use downloads ~1 GB of demucs model weights.

## Quick start

```bash
# 1) text only: fetch and inspect
lyrics-fetcher fetch "Song Title" --artist "Artist" -v

# 2) one song end-to-end, booklet photo as the lyrics source
lyrics-fetcher full song.flac --image booklet.jpg --jellyfin

# 3) a whole album: booklet/*.jpg auto-mapped to tracks, best aligner
lyrics-fetcher album "/music/Album Name" --jellyfin --aligner stable-ts

# 4) audit a suspicious .lrc, then hand-fix the flagged lines
lyrics-fetcher cross-check song.flac lyrics.txt --tolerance 2.5
lyrics-fetcher manual song.flac lyrics.txt -o song.lrc
```

Every example above has more knobs — see
[docs/CLI.md](docs/CLI.md) for the complete flag reference.

## Configuration

Settings resolve by precedence: **CLI flags > environment / `.env` >
config file > built-in defaults.** Everything has a working default, so
the only file most users need is one `[vision]` block:

```toml
# ./config.toml   (auto-detected; or ~/.config/lyrics-fetcher/config.toml,
#                  or --config /path/to.toml, or LF_CONFIG=/path)
[vision]
vision_api = "https://openrouter.ai/api/v1"   # your provider's BASE URL
vision_model = "some-vision-model"            # any vision-capable model
```

```bash
# ./.env  (gitignored; real environment variables override it)
VISION_API_KEY=***
```

The code appends the fixed `/chat/completions` path itself — you only ever
configure the base URL ending at `/v1`, which is exactly what provider
docs show. Self-hosted servers work identically
(`http://127.0.0.1:8081/v1` is the built-in default, key left empty).

**Never put API keys in the TOML** — the `.env` / environment path exists
so keys can't be committed by accident; the key field is also hidden from
reprs and sanitized out of error messages.

Full reference of every section, key, default, env-var name and matching
CLI flag → **[docs/CONFIG.md](docs/CONFIG.md)**. Template:
[`config.example.toml`](config.example.toml).

## Aligners, in brief

| | `whisper` (default) | `stable-ts` | `qwen3` |
|---|---|---|---|
| Method | whisper.cpp + monotonic anchor DP | word-level forced alignment | independent LLM aligner |
| Install | whisper.cpp binary | `uv sync` (dev group) | HF model download |
| Strengths | fast, robust baseline | intros, drift, repeated choruses; auto-falls back to whisper | second timing source for `cross-check` |
| Weaknesses | even-spread on hallucinated intros; ~6-kana drift | slower (~6–36 s/song), ~3 GiB VRAM | speech-trained; can misjudge singing |

`--separation` composes with any of them. Detailed behaviour notes:
[docs/CLI.md → Alignment engines](docs/CLI.md#alignment-engines).

## Output formats

- **`.lrc`** — standard `[mm:ss.xx]line` timestamps + metadata header
  (`[ti:]`, `[ar:]`, `[al:]`, `[by:]`). With `--jellyfin` it's written as
  `<audio basename>.lrc` beside the audio — Jellyfin's expected layout.
- **`.html`** — self-contained companion: every line wrapped with `<ruby>`
  furigana (readings from utaten, or machine-segmented otherwise), lines
  highlight in sync with a sibling `.lrc`'s timestamps.

## Troubleshooting

- **`uv sync` can't resolve torch** — you edited an index/pin but the lock
  didn't regenerate: `uv sync --index-strategy unsafe-best-match`.
- **OCR returns empty / fails** — check `[vision]` base URL + that
  `VISION_API_KEY` is set (`lyrics-fetcher ocr page.jpg` tests it alone).
  Cloud endpoints need image_url support; images are downscaled to ~1568 px
  before upload, so request-size limits rarely bite.
- **Two-column booklet pages OCR short** — VLMs sometimes skip the second
  column; crop the page or re-shoot one column per photo.
- **VRAM pressure** — whisper.cpp medium ≈ 2.5 GB, stable-ts ≈ 3.2 GB,
  Qwen3-ForcedAligner ≈ 2 GB; they run sequentially in `cross-check`, but
  keep big other models in mind on small cards.
- **`--separation` says demucs missing** — it's a dev-group dep;
  `uv sync` includes it. Falls back to raw audio meanwhile.

## Project layout

```
lyrics-fetcher/
├── lyrics_fetcher/
│   ├── cli.py                  # 7 subcommands
│   ├── config.py               # defaults + TOML + env/.env layers
│   ├── pipeline.py / batch.py  # per-song and per-album orchestration
│   ├── fetcher/                # utaten, genius, silentblue, whisper, orchestrator
│   ├── ocr/                    # vision API OCR (+ tesseract CPU fallback)
│   ├── aligner/                # whisper_cpp, stable_ts, qwen3_forced_aligner
│   ├── crosscheck.py           # dual-engine drift report
│   ├── manual_align.py         # tap-a-key timing
│   ├── cache.py                # SQLite lyrics + OCR cache
│   └── output/writers.py       # .lrc + furigana .html
├── docs/CLI.md · docs/CONFIG.md
├── config.example.toml
└── poc/                        # experiment scripts (engineering history)
```

`PLAN.md` and `HANDOFF.md` record design decisions and the development
log (including the album-specific case studies that drove features like
`--separation` and stable-ts) — they're the maintainer's notebook, not
user docs.

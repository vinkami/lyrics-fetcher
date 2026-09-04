# Configuration reference

Everything configurable, what it does, and how to set it. Values resolve in
this order (highest wins):

```
CLI flag  >  environment variable / .env  >  config.toml  >  built-in default
```

## Config file discovery

One file is auto-detected at startup, first match wins:

1. `$LF_CONFIG` (explicit path)
2. `./config.toml` (working directory)
3. `$XDG_CONFIG_HOME/lyrics-fetcher/config.toml` (or `~/.config/...`)
4. `./lyrics-fetcher.toml` / `~/.config/lyrics-fetcher/lyrics-fetcher.toml`

`--config PATH` on any command loads that file instead. `config.toml` is
**gitignored by this repo** — it's machine-local config; copy
[`config.example.toml`](../config.example.toml) and edit.

## Environment variables

Two forms, both checked after the TOML is applied (so env beats file):

- **Generic:** `LYRICS_FETCHER_<SECTION>_<KEY>` — every key below, e.g.
  `LYRICS_FETCHER_VISION_VISION_MODEL=gpt-4o`,
  `LYRICS_FETCHER_WHISPER_WHISPER_BIN=/opt/whisper/whisper-cli`.
- **Secret alias:** `VISION_API_KEY` → `[vision] vision_api_key`.

## Secrets: `.env`

At startup the CLI also loads a dotenv file if present (via
`python-dotenv`): `./.env` in the working directory, then
`~/.config/lyrics-fetcher/.env`. **Already-exported environment variables
win over `.env`** (`override=False`) — `.env` is a convenience default,
your shell profile is the sledgehammer.

Rules of thumb:

- API keys go in `.env` (gitignored) or the environment — **never in
  `config.toml`**, which people commit by accident. `vision_api_key` also
  has `repr=False` so a stray `print(settings)` can't leak it, and OCR
  HTTP errors are sanitized to the URL + status code only (never the
  response body or request).
- The repo ships no `.env.example` because the only secret is
  `VISION_API_KEY=<your provider key>` — one line, self-explanatory.

---

## `[paths]`

| key | default | used by | meaning |
|---|---|---|---|
| `music_dir` | `~/Music` | `album` | Base library dir; `album` walks the folder you pass explicitly, this is the convenience root for relative paths |
| `cache_db` | `~/.cache/lyrics-fetcher/cache.db` | fetchers + OCR | SQLite file holding (a) web fetch results keyed by source/title/artist, (b) OCR text keyed by image path. Delete it or use `--no-cache` to force fresh lookups |
| `out_dir` | `out` | `full`, `album` | Where outputs go without `--jellyfin` |

## `[whisper]` — the default aligner (whisper.cpp)

| key | default | CLI | meaning |
|---|---|---|---|
| `whisper_bin` | `~/whisper.cpp/build/bin/whisper-cli` | `--binary` | Path to the whisper.cpp binary (it's a separate C++ project; see [Setup](../README.md#whispercpp-the-default-aligner)). Vulkan build recommended |
| `whisper_model` | `~/whisper.cpp/models/ggml-medium.bin` | `--model-whisper` | Primary ggml model. `medium` is the accuracy sweet spot for this task (verified against small/large-v3-turbo on synthetic vocals) |
| `whisper_extra_models` | `~/whisper.cpp/models/ggml-large-v3-turbo.bin` | `--extra-model` (repeatable) | Second-opinion models whose transcription segments merge into the anchor pool. Only helps songs where the primary model hallucinates; each adds transcription time. Set to `[]` to disable |
| `whisper_lang` | `ja` | — | Language hint passed to whisper |
| `whisper_max_len` | `40` | — | Max segment length (tokens) for whisper — kept small so segments ≈ lyric lines, which the DP matches against |
| `whisper_device` | `0` | — | GPU index for whisper-cli's Vulkan backend (`-dev`). Multi-GPU boxes: pick the card you want, e.g. 0 |

## `[vision]` — booklet OCR model

| key | default | CLI | meaning |
|---|---|---|---|
| `vision_api` | `http://127.0.0.1:8081/v1` | `--api` | OpenAI-compatible **base URL**, ending at `/v1`. The client appends the spec-fixed `/chat/completions` itself. Works with any provider: `https://api.openai.com/v1`, `https://openrouter.ai/api/v1`, `https://api.groq.com/openai/v1`, a self-hosted llama.cpp/vLLM/Ollama server, … |
| `vision_model` | `qwen3.5-9b` | `--model` | Model name passed in the request — must match what your provider exposes **and have vision capability** (accept `image_url` content parts). Any strong multimodal model works; we validated Qwen-VL-class and GPT-4-class readers on printed JP booklets |
| `vision_api_key` | `""` | `--api-key` | Bearer token sent as `Authorization: Bearer …`. Set via `VISION_API_KEY` in `.env`/env, not here, unless your config file is private. Local servers usually need none — empty key = no auth header |

Notes: images are downscaled to ≤1568 px longest side, JPEG q88, before
upload (keeps cloud request sizes small; booklet text survives the
compression). A post-OCR cleanup pass (same endpoint) fixes minor
transcription slips; it's best-effort and skipped silently if it errors.
Results are cached in the SQLite OCR cache by image path.

## `[qwen3_aligner]` — `--aligner qwen3` / `cross-check`

| key | default | CLI | meaning |
|---|---|---|---|
| `qwen3_aligner_model` | `~/.cache/lyrics-fetcher/models/qwen3-forcedaligner` | `--qwen3-model` | Local dir with the Qwen3-ForcedAligner snapshot (`Qwen/Qwen3-ForcedAligner-0.6B-hf` from HF). Only loaded when you actually use the qwen3 engine |
| `qwen3_aligner_language` | `Japanese` | — | Language hint to the aligner (~30 supported) |

Requires transformers-from-git-main (the architecture predates nothing on
PyPI — see README dependency notes). If that import fails, everything
else in the project still works; only the qwen3 engine is unavailable.

## `[stable_ts]` — `--aligner stable-ts`

| key | default | CLI | meaning |
|---|---|---|---|
| `stable_ts_model` | `medium` | — | Whisper model *size name* (openai-whisper downloads it on first use to `~/.cache/whisper`). `medium` ≈ 3.2 GiB VRAM / 6–36 s per song. `large-v3-turbo` is more accurate if you have VRAM to spare |
| `stable_ts_lang` | `ja` | — | Language passed to the aligner |
| `stable_ts_device` | `cuda` | — | Torch device string: `"cuda"` = your primary CUDA/ROCm GPU, `"cpu"` works (slow), `"cuda:1"` for multi-GPU. NVIDIA and AMD(ROCm) both present as `cuda` |

The dev group must be installed (`uv sync` includes it). At runtime any
failure — lib missing, OOM, download blocked — prints a warning and the
line falls through to the whisper engine. One-way door: none.

## `[output]`

| key | default | CLI | meaning |
|---|---|---|---|
| `lrc_by` | `lyrics-fetcher` | — | Value of the `[by:]` credit tag written into every `.lrc` header — set it to your own handle |
| `jellyfin_default` | `false` | `--jellyfin` | When true, `full`/`album` write `<basename>.lrc`+`.html` next to each audio file (Jellyfin's layout) even without the flag |
| `write_html_default` | `true` | `--no-html` | Whether the furigana `.html` companion is written |

## `[tuning]`

| key | default | meaning |
|---|---|---|
| `anchor_min_score` | `58.0` | Minimum fuzzy score (0–100) for a whisper segment to count as a time anchor in the DP. Lower = more anchors (tighter follow of whisper's guess, more risk of bad anchors); higher = more interpolation between fewer, safer anchors. Rarely worth touching |
| `request_timeout` | `20` | HTTP timeout (s) for web fetchers. The vision client uses its own longer timeout (600 s) because cloud OCR of a full page legitimately takes ~30 s |

## Command-specific env

| variable | effect |
|---|---|
| `LF_CONFIG` | path to the config file (overrides auto-detection entirely) |
| `XDG_CONFIG_HOME` | relocates the `~/.config/lyrics-fetcher/` search dir |
| `VISION_API_KEY` | alias for `vision_api_key` |

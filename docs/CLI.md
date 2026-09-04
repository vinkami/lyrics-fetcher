# CLI reference

`lyrics-fetcher <command> [options]` — installed by `uv sync` (run it as
`lyrics-fetcher ...`, or without installing: `uv run lyrics-fetcher ...`).

Seven commands:

| command | one-liner |
|---|---|
| [`fetch`](#fetch) | get lyric text for a song title (no audio needed) |
| [`ocr`](#ocr) | transcribe a booklet photo → plain text |
| [`compile`](#compile) | align a known lyrics file against audio → `.lrc` |
| [`cross-check`](#cross-check) | run two aligners, report lines they disagree on |
| [`full`](#full) | everything for one song: metadata → text → align → outputs |
| [`album`](#album) | `full` for every track in a folder + booklet auto-mapping |
| [`manual`](#manual) | tap line starts by ear → `.lrc` |

Common options on **every** command: `--config PATH` (use this TOML config
file instead of auto-detection; see [CONFIG.md](CONFIG.md)).

---

## fetch

Look up lyrics text across the configured web sources; shows which source
matched (or a candidate-match report when nothing is accepted). No audio,
no alignment — it's the "do I even have the words?" step, and handy for
debugging why `full` picked a source.

```bash
lyrics-fetcher fetch "天ノ弱" --artist "164" -v
lyrics-fetcher fetch "Song" --source utaten
```

| option | default | meaning |
|---|---|---|
| `title` (positional, required) | — | song title to search |
| `--artist` | `""` | artist for disambiguation; strongly recommended (many titles collide across artists) |
| `--source {utaten, genius, silentblue, whisper}` | all, in order | restrict to one source. `whisper` transcribes audio — pointless here without… audio, it exists for `full`'s pipeline; here it just reports unavailable |
| `-v`, `--verbose` | off | print the full lyric text, not just the status line |
| `-o`, `--output` | — | write the accepted lyrics to a `.txt` file (`-` = stdout). Exactly what `compile` consumes — the fetch→compile scripting bridge. With `-o`, only a one-line `[saved]` summary prints; exit code 1 if nothing matched |

**Match rule:** a source's candidate is accepted only when the fuzzy title
*and* artist similarity pass a threshold; otherwise the next source tries.
This is why a bare title can return "not found" even when Genius has a
page — same title, different artist. Disc titles ending in a version
marker (`"Song (Long ver.)"`, `"Song Short ver"`) are also retried with
the marker stripped, since lyrics DBs index the bare song title —
instrumental/karaoke/language versions are deliberately exempt (their
audio doesn't match the original lyrics).

## ocr

Transcribe one booklet photo via the `[vision]` endpoint.

```bash
lyrics-fetcher ocr page.jpg
lyrics_fetcher ocr page.jpg --engine tesseract --language jpn
```

| option | default | meaning |
|---|---|---|
| `image` (positional, required) | — | photo/scan path |
| `--engine {vlm, tesseract}` | `vlm` | `vlm` = your configured OpenAI-compatible vision model (accurate on photos). `tesseract` = local CPU OCR, needs the `tesseract` binary; only good for clean flat scans |
| `--api URL` | `[vision] vision_api` | override the vision endpoint (base URL ending at `/v1`) |
| `--model NAME` | `[vision] vision_model` | override the model name |
| `--api-key KEY` | `VISION_API_KEY` env / `.env` | override the bearer token |
| `--language` | `jpn+eng` | tesseract language pack spec (ignored by `vlm`) |
| `-o`, `--output` | stdout | write the transcription to a `.txt` file instead of printing it — feed straight into `compile` |

Behaviour notes: images are downscaled (longest side 1568 px, JPEG q88)
before upload. A best-effort second pass asks the model to fix obvious
transcription slips without re-reading the image. Successful `vlm` reads
are cached in the SQLite OCR cache keyed by image path — same file = free
second run (`--no-cache`-style freshness: delete the row or the cache file).

## compile

Align a **known lyrics text file** to an audio file and write an `.lrc`.
This is the core operation the other aligning commands reuse.

```bash
lyrics-fetcher compile song.flac lyrics.txt -o song.lrc
lyrics-fetcher compile song.flac "$(cat lyrics.txt)" --aligner stable-ts
```

| option | default | meaning |
|---|---|---|
| `audio` (positional, required) | — | `.flac/.mp3/.wav/...` (decoded via ffmpeg) |
| `lyrics_file` (positional, optional) | — | path to `.txt` (one line per lyric line) or `.lrc` (timestamps stripped); a literal string is also accepted if it contains a newline. Omit → transcribes with whisper to *obtain* text first (last-resort quality) |
| `-o`, `--output` | `<audio>.lrc` | output path |
| `--title` / `--artist` / `--album` | from audio tags | override `.lrc` header fields |
| `--aligner {whisper, qwen3, stable-ts}` | `whisper` | see [Alignment engines](#alignment-engines) |
| `--binary PATH` | `[whisper] whisper_bin` | whisper-cli executable (whisper engine) |
| `--model-whisper PATH` | `[whisper] whisper_model` | ggml model file (whisper engine) |
| `--qwen3-model PATH` | `[qwen3_aligner] qwen3_aligner_model` | model dir (qwen3 engine) |

## cross-check

Run **any two or more aligners** on the same song and diff their per-line
start times. Reports lines where the engines disagree beyond a tolerance —
your triage tool for a suspicious `.lrc`.

```bash
lyrics-fetcher cross-check song.flac lyrics.txt                    # whisper vs qwen3 (default)
lyrics-fetcher cross-check song.flac lyrics.txt --engines whisper stable-ts qwen3
lyrics-fetcher cross-check song.flac lyrics.txt --tolerance 2.5 -v
```

| option | default | meaning |
|---|---|---|
| `audio`, `lyrics_file` | as `compile` | |
| `--engines ENGINE [ENGINE ...]` | `whisper qwen3` | which aligners to run and compare: any of `whisper`, `qwen3`, `stable-ts`. Two is the classic audit; three gives spread-across-engines (a line where all three agree but sit is still `ok`; one where one engine differs is `drift` with the outlier visible per column) |
| `--tolerance SECONDS` | `2.5` | a line drifts when its **spread** (max start − min start across engines) exceeds this |
| `-v`, `--verbose` | off | list every line, not just drifted/missing |
| `--binary`, `--model-whisper`, `--qwen3-model` | config | as `compile` |

Behaviour notes: whisper runs as a subprocess **first** so its GPU memory
frees before an in-process engine (qwen3 / stable-ts) loads. An engine
that errors (missing model, OOM) is reported and excluded — the remaining
engines still compare, so `--engines whisper qwen3 stable-ts` degrades to
a 2-way diff instead of dying. Whisper always runs its lean single model
here (no extra anchor models) so diffs measure the engines, not extra
config. **Exit code is non-zero if anything drifted or is missing** —
scriptable ("which of my album files need a listen?").

## full

One song, end to end: tags → text (web sources; `--image` OCR; optional
whisper fallback) → align → `.lrc` (+ `.html`).

```bash
lyrics-fetcher full song.flac --jellyfin
lyrics-fetcher full song.flac --image booklet.jpg --jellyfin --aligner stable-ts --separation
```

All of [`compile`](#compile)'s aligner options apply, plus:

| option | default | meaning |
|---|---|---|
| `--image PATH` | — | booklet photo used as an OCR lyrics source |
| `--web-first` | off | try web fetchers *before* OCR even when `--image` is given (default: OCR first — a booklet photo is authoritative for *this* pressing; web DBs may hold a different arrangement/censor) |
| `--whisper-fallback` | off | if no source has the song, best-effort whisper transcription (JP singing is unreliable; off by default so garbage text never silently aligns) |
| `--separation` | off | demucs dry-vocal stem before aligning (see [Alignment engines](#alignment-engines)) |
| `--extra-model PATH` | `[whisper] whisper_extra_models` | extra ggml model(s) whose segments merge as anchors for the whisper engine; repeatable. Helps when the primary model hallucinates a section |
| `--api` / `--model` / `--api-key` | `[vision]` | vision endpoint overrides (for `--image`) |
| `--binary` / `--model-whisper` | `[whisper]` | as `compile` |
| `-o`, `--out-dir DIR` | `out` | where to write outputs (ignored with `--jellyfin`) |
| `--jellyfin` | `[output] jellyfin_default` | write `<basename>.lrc`/`.html` **next to the audio file** instead of out-dir |
| `--no-html` | off | skip the furigana `.html` companion |
| `--no-cache` | off | bypass the SQLite lyrics/OCR cache for this run |

## album

Batch every track in a folder — same pipeline per track, plus the
booklet→track mapper: all images in the booklet dir are OCR'd
(multi-song pages split via structured JSON), fuzzy-matched to actual
disc titles (instrumentals and title-only blocks are skipped), and split
songs reassembled across pages.

```bash
lyrics-fetcher album "/music/Album Name" --jellyfin --aligner stable-ts
lyrics-fetcher album "/music/Album Name" --booklet ~/shots -q
```

Same options as [`full`](#full) (with `album_dir` positional instead of
`audio`), plus:

| option | default | meaning |
|---|---|---|
| `--booklet DIR` | `<album_dir>/booklet` | folder of booklet page images |
| `-q`, `--quiet` | off | only the per-track status table, no progress noise |

Per-track result lines are `OK` (source used), `SKIP` (no lyrics found /
instrumental), or `ERR` (message) — one track failing never aborts the
album. Exit code is non-zero if any track errored. Re-running after
fixing one photo is cheap: cached web/OCR results are reused.

## manual

Interactive fallback for songs that defeat every aligner. Plays the track
(ffplay) and you press **RETURN at each line's start**.

```bash
lyrics-fetcher manual song.flac lyrics.txt -o song.lrc
```

| key | effect |
|---|---|
| `RETURN` | stamp the current line's start time |
| `b` | back up: re-mark the previous line |
| `s` | skip current line (timestamp 0) |
| `q` | quit; writes the `.lrc` for lines marked so far |

Header options `--title/--artist/--album` as `compile`.

---

## Alignment engines

`--aligner` selects how known text is fit to audio:

**`whisper` (default)** — whisper.cpp transcribes (Vulkan GPU: AMD via
RADV, NVIDIA via its Vulkan driver), then a monotonic dynamic program fits
lyric lines to segments: only confident matches become time anchors, the
rest interpolate *between* them, order-preserving (repeated choruses stay
on the right occurrence). Known text is never replaced by the transcript.
Failure modes it has: even-spread when a dense intro produces zero anchors;
~6-kana drift after a resync; two lines merged into one segment.

**`stable-ts`** — stabilize-whisper forced alignment: the *provided text*
is walked through the acoustic model at word level, so every line start
comes from audio — no anchors, no interpolation. Fixes all three modes
above; on **any** failure (missing lib, OOM, download error, low word
coverage) it warns and falls back to the whisper engine automatically.
Needs the dev group (`uv sync`) and ~3.2 GiB VRAM with the `medium` model
(configurable to `large-v3-turbo` for more accuracy). Typical 6–36 s/song
on a mid-range GPU.

**`qwen3`** — Qwen3-ForcedAligner (0.6B): a non-autoregressive LLM that
predicts timestamps for known text, ~30 languages. Its value is
independence: it shares no failure modes with whisper, which is what
makes `cross-check` meaningful. Weak on singing (speech-trained) — rarely
your primary, good as an auditor.

**`--separation`** (orthogonal, composes with any engine) — demucs
(`htdemucs`) extracts a dry-vocal stem first and aligns against that.
Big win on BGM-dense songs where full mix makes whisper hallucinate the
intro; can *shift* songs that were already fine, hence opt-in. Adds ~30–60
s/track and needs the dev group; missing demucs = warn + align raw audio.

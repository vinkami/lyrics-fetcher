# lyrics-fetcher — Project Plan

## Overview

A tool to fetch song lyrics (text) and compile them into timestamped lyrics files. Handles obscure songs that don't appear in mainstream databases by using multiple fallback sources and AI-based recognition.

---

## Git Workflow

### Branching Strategy
- `main` — stable, working code only. Never push broken code here.
- `develop` — integration branch. Features are merged here before release.
- `feature/<name>` — individual features (e.g., `feature/utaten-fetcher`).
- `bugfix/<name>` — bug fixes.

### Commit Guidelines
- **Atomic commits**: One logical change per commit.
- **Descriptive messages**: Use conventional commits format:
  ```
  feat: add utaten.com scraper
  fix: handle encoding errors in musixmatch response
  docs: update README with installation instructions
  ```
- **Never force-push to main/develop**.

### Workflow
1. Create feature branch from `develop`: `git checkout -b feature/<name> develop`
2. Make small, focused commits with clear messages.
3. Test locally before pushing.
4. Push to remote: `git push origin feature/<name>`
5. Open PR to merge into `develop`.
6. After review, squash-merge to keep history clean.
7. Tag releases on `main`: `git tag -a v0.1.0 -m "First release" && git push --tags`

### Pre-commit Checks
- Run tests before committing: `pytest`
- Check formatting: `ruff check .`
- Type checking: `mypy lyrics_fetcher/` (if using type hints)

### Rollback Plan
If something breaks:
```bash
# Revert last commit (keeps changes in working directory)
git revert HEAD

# Or reset to previous state (discards changes)
git checkout develop
git reset --hard origin/develop
```

---

## Part 1: Lyrics Fetching

### Goal
Given a song title + artist, produce plain-text lyrics (one line per verse/chorus/stanza).

### Sources (in priority order)

#### 1a. Structured Lyrics Databases (APIs / Scraping)
- **Genius** — has API, good coverage for mainstream. Scrape as fallback.
- **Musixmatch** — large database, but scraping is tricky (JS-rendered).
- **AZLyrics / LyricFind** — traditional sites, often have obscure content.
- **UTAMAP / UTA-10** — Japanese vocaloid/community song databases.

#### 1b. Niche Community Sites
- **utaten.com** — Vocaloid lyrics (Japanese + romaji). Scraping needed.
- **silentblue.remywiki.com** — maimai/chunithm/ongeki game songs. **WORKING via index.php?search=<title>** (see `poc/02c_silentblue.py`). Covers instrumentals (returns "None."). JP songs may be stored under English page titles (Fake Face Failsafe) — need candidate fallback.
- **maimai-info / other rhythm game wikis** — for game-specific songs.

#### 1c. OCR from Physical Booklets
- User provides photos/scans of album lyric booklets.
- Pipeline: image → OCR → cleaned text.
- **TESTED (see `poc/ocr.py`):** Vision LLM (Qwen3.8-27B via llama-server on RX 9060 XT) is **far superior** to Tesseract for real phone photos (uneven lighting, no scanner). VLM transcribed ASTEROID アンデッド with accurate kanji + line breaks (~2 minor errors); Tesseract lost most lines and garbled kanji. VLM also handles maimai prism layouts and manosaba artificial-language lyrics.
- **Decision:** Vision LLM = primary OCR; Tesseract (jpn+eng) = CPU fallback for well-lit flat pages.
- Post-processing: LLM cleanup to fix minor errors, remove page numbers.

#### 1d. AI Lyrics Recognition from Audio
- Use a speech-to-text model fine-tuned for singing (e.g., Whisper with singing mode).
- Process audio file → extract lyrics line-by-line.
- Quality varies; useful as last resort or to fill gaps.

### Fetch Flow
```
Song title + artist → try sources in order → return plain text lyrics
```

Each source returns a "lyrics provider" object that can:
- Take (title, artist) → return text or None
- Have metadata: source name, confidence, language detected

---

## Part 2: Timestamp Compilation

### Goal
Take plain-text lyrics + audio file → produce a `.lrc` file with timestamps per line.

### Approach Options

#### Option A: Forced Alignment (Recommended)
- Use **Montreal Forced Aligner (MFA)** or **kaldi-based alignment** trained on singing.
- Input: audio + transcript → output: word/line-level timestamps.
- Pros: accurate, handles singing well.
- Cons: needs training data for singing voices; may need custom models.

#### Option B: Whisper + Line Segmentation
- Run Whisper on the full song to get word-level timestamps.
- Map words to lyrics lines (fuzzy match).
- Assign line timestamps from word boundaries.
- Pros: no extra training, works out of box.
- Cons: Whisper may miss syllables in fast songs; singing mode needed.

#### Option C: Manual / Semi-Automatic
- User provides approximate timing or edits manually.
- Tool shows alignment suggestions, user corrects.
- UI: terminal-based editor or simple web interface.

### Output Format
- `.lrc` format (standard for lyrics files):
  ```
  [00:12.34]Line one
  [00:15.67]Line two
  ```
- Optional: extended LRC with metadata (title, artist, album).

---

## Orchestrator Workflow

### How it works:
```python
class Orchestrator:
    def fetch(self, title, artist):
        results = []
        
        # Try all sources in parallel (or sequentially if rate-limited)
        for source in SOURCES:
            try:
                lyrics = source.fetch(title, artist)
                if lyrics:
                    snippet = extract_first_lines(lyrics, n=2)
                    confidence = source.estimate_confidence(lyrics)
                    results.append({
                        'source': source.name,
                        'lyrics': lyrics,
                        'snippet': snippet,
                        'confidence': confidence
                    })
            except Exception:
                continue
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If multiple results, present choices to user
        if len(results) > 1:
            self.present_choices(results)  # Show numbered list with snippets
            choice = input("Pick source number: ")
            return results[int(choice) - 1]['lyrics']
        
        return results[0]['lyrics'] if results else None
```

### Confidence Scoring Factors:
- Source reliability (wiki > random blog)
- Language match (Japanese source for Japanese song)
- Completeness (full lyrics vs partial)
- Formatting quality (clean text vs HTML tags)

### Snippet Extraction:
- First 1-2 lines of lyrics for disambiguation
- Helps user identify the correct song when titles are ambiguous

---

## Output Format Details

### LRC File (Primary)
```lrc
[ti:Song Title]
[ar:Artist Name]
[al:Album Name]
[by:lyrics-fetcher]

[00:12.34]First line of lyrics
[00:15.67]Second line
[00:18.90]Third line
```

### HTML Companion File (with Furigana)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Song Title - Artist</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .lyrics-line { margin: 10px 0; line-height: 1.6; }
        ruby { ruby-position: over; }
        rt { font-size: 0.7em; color: #666; }
    </style>
</head>
<body>
    <h1>Song Title</h1>
    <p>Artist: Artist Name | Album: Album Name</p>
    
    <div class="lyrics-line">
        <ruby>漢字<rp>(</rp><rt>かんじ</rt><rp>)</rp></ruby>の歌詞
    </div>
    
    <div class="lyrics-line">
        別の行のテキスト
    </div>
</body>
</html>
```

### Furigana Generation:
- Use a Japanese reading library (e.g., `kytea`, `janome`, or `mecab`) to convert kanji to hiragana
- Store furigana in HTML companion file
- LRC remains standard for Jellyfin compatibility

---

## Architecture

### Project Structure
```
lyrics-fetcher/
├── lyrics_fetcher/
│   ├── __init__.py
│   ├── cli.py              # Main CLI entry point
│   ├── cache/              # SQLite caching layer
│   │   ├── db.py           # Database connection & queries
│   │   └── models.py       # Cache table schemas
│   ├── fetcher/            # Part 1: Lyrics fetching
│   │   ├── base.py         # Abstract provider interface
│   │   ├── genius.py       # Genius API/scrape
│   │   ├── musixmatch.py   # Musixmatch
│   │   ├── azlyrics.py     # AZLyrics scraper
│   │   ├── utaten.py       # Vocaloid lyrics site
│   │   ├── silentblue.py   # maimai wiki
│   │   ├── utamap.py       # UTA-10 / other databases
│   │   └── orchestrator.py # Tries sources, presents choices
│   ├── ocr/                # Part 1b: OCR from booklets
│   │   ├── preprocess.py   # Image preprocessing
│   │   ├── recognize.py    # OCR engine (Tesseract wrapper)
│   │   └── clean.py        # Post-processing cleaned text
│   ├── aligner/            # Part 2: Timestamp alignment
│   │   ├── base.py         # Abstract aligner interface
│   │   ├── whisper_align.py# Whisper-based alignment (faster-whisper)
│   │   ├── whisper_cpp.py  # whisper.cpp with ROCm/Vulkan
│   │   ├── mfa_align.py    # Montreal Forced Aligner (CPU)
│   │   └── manual.py       # Semi-automatic editing
│   ├── output/             # LRC + HTML file generation
│   │   ├── lrc_writer.py   # Write .lrc with metadata
│   │   ├── html_writer.py  # Write companion HTML with furigana
│   │   └── furigana.py     # Kanji → hiragana converter
│   └── utils/              # Helpers
│       ├── audio.py        # Audio loading (pydub / librosa)
│       └── text.py         # Text normalization, fuzzy matching
├── tests/
├── setup.py / pyproject.toml
├── requirements.txt
└── README.md
```

### CLI Design
```bash
# Fetch lyrics for a song (tries all sources, shows choices if ambiguous)
lyrics-fetcher fetch "Song Title" --artist "Artist Name"

# Fetch with specific source
lyrics-fetcher fetch "Song Title" --artist "Artist Name" --source utaten

# OCR from images
lyrics-fetcher ocr path/to/booklet.jpg --language ja

# Compile: align lyrics with audio → LRC + HTML files
lyrics-fetcher compile song.mp3 lyrics.txt --output song.lrc

# Full pipeline: fetch + compile (generates .lrc + .html)
lyrics-fetcher full "Song Title" --artist "Artist" --audio song.mp3 --output song.lrc
```

### Output Files
For each song, generate two files:
- `song.lrc` — Standard LRC for Jellyfin/media players
- `song.html` — Companion HTML with furigana for local viewing

---

## Environment Setup

### Python Version
- **Python 3.12** — Latest stable with full library support.
- All dependencies tested against 3.12.

### Virtual Environment with uv
```bash
# Create project with uv
cd ~/Code/lyrics-fetcher
uv init --python 3.12

# Add dependencies
uv add requests httpx pytesseract pydub faster-whisper thefuzz typer sqlite3

# Add dev dependencies
uv add --dev pytest ruff mypy

# Run in environment
uv run python -m lyrics_fetcher.cli ...
```

### Dependencies (Tentative)

| Component | Library | Purpose |
|-----------|---------|---------|
| HTTP requests | `requests` / `httpx` | Web scraping |
| OCR | `pytesseract` + Tesseract | Text recognition from images |
| Audio loading | `pydub` / `librosa` | Read audio files |
| Whisper alignment | `faster-whisper` (CTranslate2) | Speech-to-text with timestamps, Vulkan/ROCm support |
| Whisper.cpp | `whisper.cpp` (compiled for ROCm) | Alternative whisper backend |
| Forced alignment | `montreal-forced-aligner` (MFA) | Word-level alignment (CPU) |
| Fuzzy matching | `thefuzz` (fuzzywuzzy) | Match lyrics lines to words |
| CLI | `typer` | Command-line interface |
| Audio processing | `soundfile` / `scipy` | Format conversion, resampling |
| LLM cleanup | `transformers` + small model (phi-2, T5) | Text normalization & error correction |
| Japanese reading | `kytea`, `janome`, or `mecab` | Kanji → hiragana for furigana |
| SQLite | `sqlite3` (built-in) | Cache fetch results |
| Testing | `pytest` | Test framework |
| Linting | `ruff` | Code formatting & linting |
| Type checking | `mypy` | Static type checking |

---

## Design Decisions

### 1. Language Support
- **Decision:** Prioritize English + Japanese. Chinese (traditional) as stretch goal.
- **Context:** utaten = Japanese vocaloid, maimai wiki = Japanese songs. OCR needs CJK models.

### 2. Whisper vs MFA for Alignment
- **Decision:** Try both. Use `faster-whisper` (CTranslate2 with Vulkan/ROCm support) and `whisper.cpp` compiled for ROCm as primary options. MFA as fallback (CPU-based, less GPU-friendly).
- **Context:** RX 9060 XT with ROCm/Vulkan. Whisper via faster-whisper should work well. MFA is CPU-bound but more accurate for singing.

### 3. OCR Quality & LLM Cleanup
- **Decision:** Aggressive preprocessing + small local LLM (phi-2, starcoder, or fine-tuned T5) for cleanup. Minimize user input while maintaining accuracy. Human-in-the-loop available for corrections.
- **Flow:** Raw OCR/scrape → LLM cleanup (fix typos, normalize format) → confidence score

### 4. Output Format
- **Decision:** Standard LRC as primary output. HTML companion file with furigana on kanji characters for local use. No romaji needed. Jellyfin-compatible.
- **LRC:** `[mm:ss.xx]Line text`
- **HTML companion:** `<ruby>漢字<rp>(</rp><rt>ふりがな</rt><rp>)</rp></ruby>` for proper furigana rendering

### 5. Caching Strategy
- **Decision:** SQLite cache for fetch results (title → source URL + raw text). File-based output for final LRC/HTML files. Track provenance without redundant scraping.

### 6. Error Handling & Fallbacks
- **Decision:** Confidence scoring per source. Present choices to user with snippets for disambiguation when multiple sources return results.

### 7. User Interaction Model
- **Decision:** Numbered list → user picks number. Include first 1-2 lines of lyrics as snippet for better disambiguation.

---

## Implementation Order (Suggested)

**Remember**: Each step should be committed to a feature branch before moving to the next. Test thoroughly before merging to `develop`.

1. **Foundation** — Project structure, CLI skeleton, utils (audio/text helpers), SQLite cache
   - `git checkout -b feature/foundation develop`
   - Commit: `feat: initial project structure with CLI and cache`
2. **Basic fetcher** — One working source (e.g., Genius API) to validate the pipeline
   - `git checkout -b feature/genius-fetcher develop`
   - Commit: `feat: add Genius API scraper with basic error handling`
3. **Orchestrator** — Multiple sources with fallback logic and confidence scoring
   - `git checkout -b feature/orchestrator develop`
   - Commit: `feat: implement orchestrator with confidence scoring and disambiguation`
4. **Niche sources** — utaten, silentblue.remywiki.com (scraping these specific sites)
   - `git checkout -b feature/niche-sources develop`
   - Commit: `feat: add utaten.com and silentblue scraper for vocaloid/maimai songs`
5. **OCR module** — Image preprocessing + Tesseract integration + LLM cleanup
   - `git checkout -b feature/ocr-module develop`
   - Commit: `feat: add OCR pipeline with Tesseract and LLM cleanup`
6. **Alignment** — Whisper-based alignment first (easier), MFA as advanced option
   - `git checkout -b feature/alignment develop`
   - Commit: `feat: implement Whisper-based timestamp alignment`
7. **LRC output** — Write proper .lrc files with metadata
   - `git checkout -b feature/lrc-output develop`
   - Commit: `feat: add LRC file writer with metadata support`
8. **HTML companion** — Generate HTML files with furigana
   - `git checkout -b feature/html-companion develop`
   - Commit: `feat: generate HTML companion files with furigana`
9. **Full pipeline** — CLI that chains fetch → OCR/align → LRC + HTML output
   - `git checkout -b feature/full-pipeline develop`
   - Commit: `feat: implement full pipeline CLI command`
10. **Manual editing mode** — Semi-automatic correction interface
    - `git checkout -b feature/manual-editing develop`
    - Commit: `feat: add semi-automatic lyrics editing interface`

---

## Risks & Considerations

- **Scraping legality:** Some sites (Genius, Musixmatch) have ToS against scraping. API keys may be required.
- **Whisper singing accuracy:** Whisper was trained on speech, not singing. Performance may be poor for fast vocaloid or high-pitched songs.
- **OCR for CJK:** Requires proper font/model training. Tesseract's CJK models may need tuning.
- **MFA setup complexity:** Montreal Forced Aligner has many dependencies (kaldi, python libraries). May be hard to install on all systems.
- **LLM cleanup overhead:** Adding a small LLM increases processing time and memory usage. Consider using a lightweight model or API call if local resources are limited.

# poc/ocr_mynagi_retry.py — OCR the 命を振り回せ booklet photo N times, report
# line-count variance. Known VLM gap: 2-column layouts drop the second column.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lyrics_fetcher.ocr.vision import VLMOcr

IMG = Path("/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID/booklet/20260902_004849.jpg")

ocr = VLMOcr()
(Path("out/ocr_try")).mkdir(parents=True, exist_ok=True)
for attempt in range(3):
    songs = ocr.extract_songs(IMG, known_titles=["命を振り回せ"])
    label, text = next(iter(songs.items()))
    lines = [l for l in text.splitlines() if l.strip()]
    print(f"attempt {attempt}: label={label!r} lines={len(lines)}")
    (Path("out/ocr_try") / f"try{attempt}.txt").write_text(text, encoding="utf-8")
Path("out/ocr_try/done.flag").touch()

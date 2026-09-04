# poc/ocr_probe_photos.py — which song does each booklet photo hold?
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lyrics_fetcher.ocr.vision import VLMOcr

BOOKLET = Path("/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID/booklet")
ocr = VLMOcr()
out = Path("out/ocr_probe"); out.mkdir(parents=True, exist_ok=True)
for img in sorted(BOOKLET.glob("*.jpg")):
    try:
        songs = ocr.extract_songs(img)
    except Exception as e:
        print(f"{img.name}: ERROR {e}")
        continue
    for label, text in songs.items():
        lines = [l for l in text.splitlines() if l.strip()]
        print(f"{img.name}: {label!r} ({len(lines)} lines)")
        (out / f"{img.stem}_{label[:8]}.txt").write_text(text, encoding="utf-8")
print("PROBE_DONE")

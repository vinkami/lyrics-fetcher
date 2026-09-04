# poc/mynagi_words.py — word timestamps around the 尊大/なぁなぁ lines (55-66s)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_whisper
import whisper

AUDIO = "/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID/02 命を振り回せ.flac"
TXT = Path("out/fix/02 命を振り回せ_full52.txt").read_text(encoding="utf-8").splitlines()

m = stable_whisper.load_model("medium", device="cuda")
result = m.align(AUDIO, "\n".join(TXT), language="ja", regroup="p", verbose=False)
for seg in result.segments:
    for w in seg.words:
        s, e = w.start, w.end
        if 50 <= s <= 68:
            print(f"{s:7.2f}-{e:6.2f} {w.word}")

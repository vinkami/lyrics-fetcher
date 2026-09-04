# poc/mynagi_region.py — where is 尊大ぶった閻魔帳 actually sung? (raw flac,
# slice 52-70s, forced-align the 4-line region, per-char word starts)
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_whisper
import whisper

AUDIO = "/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID/02 命を振り回せ.flac"
region = [
    "前頭葉はデッドなう 生存匂わせカット",
    "尊大ぶった閻魔帳",
    "なぁなぁのままです中途半端",
    "現実妙なファンタジー 環状線あっちで乗った",
]


def norm(t):
    return re.sub(r"[\s、。「」『』（）()・—ー!！?？,.]", "",
                  unicodedata.normalize("NFKC", t))


m = stable_whisper.load_model("medium", device="cuda")
audio = whisper.load_audio(AUDIO)
seg = audio[52 * 16000:70 * 16000]
res = m.align(seg, "\n".join(region), language="ja", regroup=False, verbose=False)
starts = []
for s in res.segments:
    for w in s.words:
        starts.extend([w.start] * len(norm(w.word)))
print("aligned chars:", len(starts), "expected:",
      sum(len(norm(r)) for r in region))
pos = 0
for r in region:
    n = len(norm(r))
    t = starts[pos] + 52 if pos < len(starts) else -1
    print(f"{t:7.2f}  {r!r}")
    pos += n

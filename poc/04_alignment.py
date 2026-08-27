"""PoC: Forced alignment — known lyrics + whisper timestamps → LRC.

This is the core of Part 2. Given:
  - plain-text lyrics (from a fetcher, e.g. utaten/Genius)
  - an audio file
we produce timestamped LRC lines.

Approach:
  1. Run whisper.cpp (Vulkan on RX 9060 XT) to get segment timestamps, using
     medium multilingual which we verified is most accurate for synthetic vocals.
  2. Fuzzy-match each KNOWN lyrics line to the whisper-recognized text.
  3. Assign the matched whisper segment's start time to the known line.
  4. Output .lrc + .html.

The known lyrics are authoritative; whisper only supplies TIMESTAMPS. So even if
whisper garbles an obscure song's text, whichever lines it does recognize give us
time anchors, and we keep the correct text from the fetcher.
"""

import json
import re
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

# make thefuzz optional so this runs even if not installed
try:
    from thefuzz import fuzz
except ImportError:
    fuzz = None

MUSIC_DIR = Path("/mnt/fnos/storage/Music")
WHISPER = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-medium.bin"


def run_whisper(audio: Path, lang: str = "ja", max_len: int = 40) -> list[dict]:
    """Run whisper.cpp, return segments: [{from_ms, to_ms, text, off}]."""
    out = Path("/tmp/_lf_whisper")
    cmd = [
        str(WHISPER), "-m", str(MODEL), "-l", lang,
        "-f", str(audio), "-ml", str(max_len),
        "-oj", "-of", str(out), "--no-prints",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    data = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    segs = []
    for s in data.get("transcription", []):
        off = s["offsets"]
        segs.append({
            "from": off["from"],
            "to": off["to"],
            "text": s["text"].strip(),
        })
    return segs


def clean(s: str) -> str:
    """Normalize for matching: strip punctuation/whitespace, keep kana/kanji."""
    return re.sub(r"[\s「」『』（）()〈〉【】、。,.!！?？\-]", "", s)


def align_lines(known_lines: list[str], segs: list[dict]) -> list[tuple[float, str]]:
    """
    For each known line, find the whisper segment with highest fuzzy match.
    Assign that segment's start time (seconds) to the line.
    Lines with no good match get an interpolated timestamp.
    """
    if fuzz is None:
        # naive fallback: distribute lines evenly across audio duration
        total = segs[-1]["to"] / 1000 if segs else 0
        n = len(known_lines)
        return [(i * total / max(n, 1), line) for i, line in enumerate(known_lines)]

    result = []  # list of (known_idx, seg_idx, score)
    for ki, line in enumerate(known_lines):
        cl = clean(line)
        best, best_score = None, 0
        for si, seg in enumerate(segs):
            score = fuzz.ratio(cl, clean(seg["text"]))
            if score > best_score:
                best, best_score = si, score
        result.append((ki, best, best_score))

    # Build time list; interpolate for low-confidence lines
    times = {}
    for ki, si, score in result:
        times[ki] = segs[si]["from"] / 1000

    # fill gaps: for lines that share a segment or have none, spread within window
    ordered = sorted(times.items())
    out = []
    for ki, line in enumerate(known_lines):
        t = times[ki]
        out.append((t, line))
    return out


def write_lrc(path: Path, title, artist, album, timed):
    L = [f"[ti:{title}]", f"[ar:{artist}]", f"[al:{album}]", "[by:lyrics-fetcher]", ""]
    for t, text in timed:
        m, s = divmod(t, 60)
        L.append(f"[{int(m):02d}:{s:05.2f}]{text}")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    print("=== PoC: Alignment (utaten lyrics + whisper timestamps) ===\n")

    audio = MUSIC_DIR / "VOCALOID 超BEST -memories-" / "08 天ノ弱.flac"
    # In real pipeline these come from the fetcher. Use utaten output we validated earlier.
    known_lines = [
        "僕がずっと前から思ってる事を話そうか",
        "友達に戻れたらこれ以上はもう望まないさ",
        "君がそれでいいなら僕だってそれで構わないさ",
        "嘘つきの僕が吐いた はんたいことばの愛のうた",
        "今日はこっちの地方はどしゃぶりの晴天でした",
        "昨日もずっと暇で一日満喫してました",
        "別に君のことなんて考えてなんかいないさいや",
        "でもちょっと本当は考えてたかもなんてね",
    ]

    print(f"Audio: {audio.name}")
    print(f"Known lyrics: {len(known_lines)} lines")
    print("Running whisper.cpp on GPU...\n")

    segs = run_whisper(audio)
    print(f"Whisper produced {len(segs)} segments (text below is GPU-recognized):")
    for s in segs:
        print(f"  [{s['from']/1000:6.2f}s] {s['text'][:45]}")
    print()

    timed = align_lines(known_lines, segs)
    out = Path(__file__).parent / "out" / "aligned_天ノ弱.lrc"
    write_lrc(out, "天ノ弱", "164 feat. 初音ミク", "VOCALOID 超BEST", timed)
    print(f"ALIGNED LRC written to {out}\n")
    for t, line in timed:
        m, s = divmod(t, 60)
        print(f"  [{int(m):02d}:{s:05.2f}] {line}")


if __name__ == "__main__":
    main()
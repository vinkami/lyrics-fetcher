"""Test: Monotonic alignment on real vocaloid song (天ノ弱).

Uses the known utaten lyrics + real whisper.cpp segments (medium, GPU).
Gets whisper segments fresh (audio is /mnt/fnos/storage). Asserts the aligned
timestamps are strictly non-decreasing — the property the old greedy matcher
violated. Prints the full aligned LRC for visual inspection of correctness.
"""
import importlib.util
import json
import subprocess
from pathlib import Path

_p = str(Path(__file__).parent / "08_pipeline.py")
_spec = importlib.util.spec_from_file_location("pipeline08", _p)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
align_lines = _m.align_lines
run_whisper = _m.run_whisper

WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-medium.bin"
AUDIO = Path("/mnt/fnos/storage/Music/VOCALOID 超BEST -memories-") / "08 天ノ弱.flac"

# Known utaten lyrics (lines in order)
KNOWN = [
    "僕がずっと前から思ってる事を話そうか",
    "友達に戻れたらこれ以上はもう望まないさ",
    "君がそれでいいなら僕だってそれで構わないさ",
    "嘘つきの僕が吐いた はんたいことばの愛のうた",
    "今日はこっちの地方はどしゃぶりの晴天でした",
    "昨日もずっと暇で一日満喫してました",
    "別に君のことなんて考えてなんかいないさいや",
    "でもちょっと本当は考えてたかもなんてね",
]


def run_whisper_cached(audio):
    """Run whisper; cache JSON so repeat runs are instant."""
    cache = Path("/tmp/_lf_ameno_segs.json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    segs = run_whisper(audio)
    cache.write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")
    return segs


def main():
    print("=== Real-song monotonic alignment test: 天ノ弱 ===\n")
    print("Running whisper.cpp (medium, GPU)...")
    segs = run_whisper_cached(AUDIO)
    print(f"Whisper segments: {len(segs)}")
    print("Segments as recognized:")
    for s in segs:
        print(f"  [{s['from']/1000:5.2f}s] {s['text']}")
    print()

    timed = align_lines(KNOWN, segs)
    print("=== Aligned LRC ===")
    for t, line in timed:
        m, s = divmod(t, 60)
        print(f"  [{int(m):02d}:{s:05.2f}] {line}")

    times = [t for t, _ in timed]
    ok = all(times[i] <= times[i+1] for i in range(len(times)-1))
    print()
    print(f"MONOTONIC={ok}")
    if not ok:
        # find where it breaks
        for i in range(len(times)-1):
            if times[i] > times[i+1]:
                print(f"  BREAK at line {i}: {times[i]} -> {times[i+1]}")
                break
    assert ok, "timestamps not non-decreasing!"


if __name__ == "__main__":
    main()
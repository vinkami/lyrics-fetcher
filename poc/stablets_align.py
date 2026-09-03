# /// script
# throws away after Phase 0 gate
# ///
"""
poc/stablets_align.py — Phase 0 PoC: stable-ts forced alignment on ASTEROID vocal stems.

For each song:
  1. Take corrected lyrics (_lrc_re-ocr/*.lrc, the deployed good text).
  2. m.align(vocal_stem, lyrics_text) + regroup -> per-line start times.
  3. Compare line-by-line against the current .lrc times.

Writes results to poc/out/stablets_results.json + prints a per-song report.
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REOCR = ROOT / "_lrc_re-ocr"
STEMS = ROOT / "_sep_out"
OUT = ROOT / "poc" / "out"
OUT.mkdir(exist_ok=True)

LRC_TS = re.compile(r"^\[(\d\d):(\d\d(?:\.\d+)?)\](.*)$")

SONGS = [
    "01 アンデッド",
    "02 命を振り回せ",
    "03 告げよ",
    "04 黒い目",
    "05 サテライト",
]


def parse_lrc(path: Path):
    """Return (header_lines, [(start_sec, text), ...])."""
    times, texts = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = LRC_TS.match(raw.strip())
        if m:
            times.append(int(m.group(1)) * 60 + float(m.group(2)))
            texts.append(m.group(3).strip())
    return times, texts


def main():
    import stable_whisper  # noqa: E402

    print("loading medium...", flush=True)
    t0 = time.time()
    model = stable_whisper.load_model("medium", device="cuda")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    results = {}
    for song in SONGS:
        lrc = REOCR / f"{song}.lrc"
        stem = STEMS / f"{song}_vocals.wav"
        cur_times, lines = parse_lrc(lrc)
        text = "\n".join(lines)
        print(f"\n=== {song}: {len(lines)} lines ===", flush=True)

        t1 = time.time()
        result = model.align(str(stem), text, language="ja", regroup="p", verbose=False)
        align_secs = time.time() - t1

        # Flatten word timestamps, then assign words to lyric lines greedily:
        # a line starts at the word where its first unmatched char matched.
        words = []  # (start, text)
        for seg in result.segments:
            for w in (seg.words or []):
                words.append((w.start, w.word.replace(" ", "")))

        st_times, wi = [], 0
        for line in lines:
            target = re.sub(r"\s+", "", line)
            line_start = None
            acc = ""
            while wi < len(words) and len(acc) < len(target):
                ws, wt = words[wi]
                if line_start is None and wt:
                    line_start = ws
                acc += wt
                wi += 1
                if acc == target or (len(acc) >= len(target)):
                    break
            # if overshoot/mismatch, best-effort: rewind not implemented; use line_start
            st_times.append(line_start if line_start is not None else (
                st_times[-1] if st_times else 0.0))

        segs = result.segments
        st_texts = [seg.text.strip() for seg in segs]

        # alignment sanity: line count + word coverage
        n = len(lines)
        rows = []
        for i in range(n):
            echo = st_texts[min(i, len(st_texts) - 1)] if st_texts else ""
            rows.append({
                "line": i,
                "text": lines[i],
                "cur": round(cur_times[i], 2) if i < len(cur_times) else None,
                "stable": round(st_times[i], 2),
                "delta": (round(st_times[i] - cur_times[i], 2)
                          if i < len(cur_times) else None),
            })

        mono_bad = sum(1 for a, b in zip(st_times, st_times[1:]) if b < a - 0.01)
        drift2 = sum(1 for r in rows if r["delta"] is not None and abs(r["delta"]) > 2.0)

        results[song] = {
            "align_secs": round(align_secs, 1),
            "n_lines": len(lines),
            "n_segments": len(segs),
            "monotonic_violations": mono_bad,
            "lines_over_2s_vs_current": drift2,
            "rows": rows,
        }

        print(f"  aligned {len(segs)} segs / {len(lines)} lines in {align_secs:.0f}s "
              f"| mono_violations={mono_bad} | >2s vs current={drift2}", flush=True)
        print("  first 8 lines (cur -> stable-ts):", flush=True)
        for r in rows[:8]:
            print(f"    [{r['line']:>2}] {r['cur']:>7} -> {r['stable']:>7}  {r['text'][:28]}", flush=True)

    with open(OUT / "stablets_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\nwrote", OUT / "stablets_results.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())

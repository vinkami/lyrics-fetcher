# poc/apply_review_fixes2.py — round-2 playback corrections (user review).
#
# 命を振り回せ: (a) booklet reads まーいっか (revert my whisper-ized まぁいいか);
#   (b) ヒーローだってさいなら sat at 182.92 same as 青春時代 (frozen dup after
#   諦観's 4.6s hold = "took too long to transition"); whisper-medium has
#   ヒーローだってさいなら starting 180.0 — deploy 03:00.00, 青春 stays 182.92.
# サテライト: over-corrected — ONLY the 俯いたまま line is シューゲイザー;
#   the first really prints シューメイカー per user's booklet.
# 黒い目: booklet reads 何だか虚しくて (was 何か虚しくて).
#
# Edits the deployed .lrc AND .html on the NAS (Jellyfin layout).
import re
import sys
from pathlib import Path

ALBUM = Path("/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID")

# (file, old_substring, new_substring, replace_all)
EDITS = [
    ("02 命を振り回せ.lrc", "まぁいいかとやり過ごして", "まーいっかとやり過ごして", True),
    ("02 命を振り回せ.lrc", "[03:02.92]ヒーローだってさいなら",
     "[03:00.00]ヒーローだってさいなら", False),
    ("02 命を振り回せ.html", 'data-start="151.68">まぁいいか',
     'data-start="151.68">まーいっか', True),
    ("02 命を振り回せ.html", 'data-start="182.92">ヒーローだってさいなら',
     'data-start="180.00">ヒーローだってさいなら', False),
    ("05 サテライト.lrc", "どうしたって離れていくのシューゲイザー",
     "どうしたって離れていくのシューメイカー", False),
    ("05 サテライト.html", "離れていくのシューゲイザー",
     "離れていくのシューメイカー", False),
    ("04 黒い目.lrc", "どうにもこうにも何か虚しくて",
     "どうにもこうにも何だか虚しくて", True),
    ("04 黒い目.html", "どうにもこうにも何か虚しくて",
     "どうにもこうにも何だか虚しくて", True),
]


def main():
    for fname, old, new, all_ in EDITS:
        p = ALBUM / fname
        src = p.read_text(encoding="utf-8")
        cnt = src.count(old)
        if cnt == 0:
            print(f"!! NOT FOUND in {fname}: {old[:40]!r}")
            return 1
        if not all_ and cnt > 1:
            print(f"!! AMBIGUOUS ({cnt}x) in {fname}: {old[:40]!r}")
            return 1
        p.write_text(src.replace(old, new), encoding="utf-8")
        print(f"ok {fname}: {old[:34]!r} -> {new[:34]!r} ({cnt}x)")
    print("ROUND2_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

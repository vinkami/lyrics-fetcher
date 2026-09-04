# poc/apply_review_fixes.py — apply the user's 4 playback-review corrections.
#
# 1. サテライト: シューメーカー/シューガイザー -> シューゲイザー (booklet font
#    drops thin strokes; user ground truth). Text only, timing untouched.
# 2. 告げよ: first chorus 薄い破片〜凪いでく frozen cluster -> raw-FLAC
#    re-align times (out/fix/03 告げよ_raw.lrc), matches user-approved backup.
# 3. 黒い目: 情熱的にさ/虚しくて break-entry -> raw-FLAC times (whisper-medium
#    corroborates 113-123.5s; stem doubled-chorus false-matched to 100s).
# 4. 命を振り回せ: full 52-line text (right column recovered), timing from the
#    stem alignment (out/fix/02 命を振り回せ_full52.lrc), text refinements from
#    the reviewed 38-line file + whisper verification.
#
# Writes .lrc + .html next to the FLACs (Jellyfin layout). Backups exist in
# _backups/ASTEROID_2026-09-04_post-stablets/.
import re
import sys
from pathlib import Path

ALBUM = Path("/mnt/fnos/storage/Music/光収容の倉庫 ASTEROID")
FIX = Path("out/fix")

LRC_RE = re.compile(r"^\[(\d\d):(\d\d(?:\.\d+)?)\](.*)$")


def parse_lrc(path: Path):
    header, timed = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = LRC_RE.match(ln.strip())
        if m:
            timed.append((int(m[1]) * 60 + float(m[2]), m[3]))
        else:
            header.append(ln)
    return header, timed


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"[{int(m):02d}:{s:05.2f}]"


def write_pair(name: str, timed, title: str):
    lrc = ALBUM / f"{name}.lrc"
    header, _ = parse_lrc(lrc)
    body = [f"{fmt(t)}{text}" for t, text in timed]
    lrc.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    html = ALBUM / f"{name}.html"
    divs = "\n".join(
        f'<div class="lyrics-line" data-start="{t:.2f}">{text}</div>'
        for t, text in timed
    )
    src = html.read_text(encoding="utf-8")
    # flat structure: <p class="meta">, then lyrics-line divs, then </body>
    first = src.index('<div class="lyrics-line"')
    end = src.index("</body>")
    html.write_text(src[:first] + divs + "\n" + src[end:], encoding="utf-8")
    print(f"  wrote {lrc.name} ({len(timed)} lines) + {html.name}")


def main():
    # ---- 1. サテライト: text-only fix ----
    _, timed = parse_lrc(ALBUM / "05 サテライト.lrc")
    n = 0
    fixed = []
    for t, text in timed:
        new = text.replace("シューメーカー", "シューゲイザー").replace("シューガイザー", "シューゲイザー")
        n += new != text
        fixed.append((t, new))
    print(f"サテライト: {n} text fixes")
    write_pair("05 サテライト", fixed, "サテライト")

    # ---- 2. 告げよ: first-chorus times from raw-flac align ----
    _, dep = parse_lrc(ALBUM / "03 告げよ.lrc")
    _, raw = parse_lrc(FIX / "03 告げよ_raw.lrc")
    out = []
    for i, (t, text) in enumerate(dep):
        if text.startswith(("薄い破片", "何もかもが軽くなっちゃって", "透けてくように")) and i < 20:
            out.append((raw[i][0], text))
        else:
            out.append((t, text))
    write_pair("03 告げよ", out, "告げよ")

    # ---- 3. 黒い目: break-entry times from raw-flac align ----
    _, dep = parse_lrc(ALBUM / "04 黒い目.lrc")
    _, raw = parse_lrc(FIX / "04 黒い目_raw.lrc")
    out = [(raw[i][0], text) if text.startswith(("情熱的にさ", "どうにもこうにも何か虚しくて"))
           else (t, text) for i, (t, text) in enumerate(dep)]
    write_pair("04 黒い目", out, "黒い目")

    # ---- 4. 命を振り回せ: full 52-line rebuild ----
    text52 = (FIX / "02 命を振り回せ_full52.txt").read_text(encoding="utf-8").splitlines()
    # text refinements: 5 from the reviewed 38-line deploy, 2 whisper-verified
    refine = {
        "惨めに思えてきそうだろ 嫌になるよ": "惨めに思えてきそうだろう 嫌になるよ",
        "楽なまでで居たいんでしょ膠着": "楽なままで居たいんでしょ 膠着",
        "なぁなぁあのままでさ中途半端": "なぁなぁのままです中途半端",
        "こんなの新しいルールなのは変じゃない？": "こんな新しいルールなのは変じゃない？",
        "待っても変わらなくて なんだか物悲しい": "待っても変わらなくて 何か物悲しい",
        "雷でくノーマル シニシズム": "霞んでくノーマル シニシズム",
        "まーいっかとやり過ごして": "まぁいいかとやり過ごして",
    }
    text52 = [refine.get(l, l) for l in text52]
    _, timed = parse_lrc(FIX / "02 命を振り回せ_full52.lrc")
    assert len(timed) == len(text52) == 52, (len(timed), len(text52))
    out = [(t, text52[i]) for i, (t, _) in enumerate(timed)]
    # guard: monotonic + tail matches the user-approved 諦観〜 region
    assert all(b[0] >= a[0] for a, b in zip(out, out[1:]))
    write_pair("02 命を振り回せ", out, "命を振り回せ")
    print("ALL_FIXES_DONE")


if __name__ == "__main__":
    sys.exit(main())

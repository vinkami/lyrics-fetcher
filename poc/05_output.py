"""PoC: Output generation — LRC file + HTML companion with furigana.

Tests:
- writing standard LRC with metadata headers (Jellyfin-compatible)
- writing HTML with <ruby> furigana from utaten's rb/rt pairs
- parsing an existing .lrc (the one shipped with the album) to check format compat

No AI needed — pure text work.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape

MUSIC_DIR = Path("/mnt/fnos/storage/Music")
OUT = Path(__file__).parent / "out"


def write_lrc(path: Path, title: str, artist: str, album: str, timed_lines: list[tuple[float, str]]):
    """timed_lines: [(seconds, text), ...] → standard LRC."""
    lines = [
        f"[ti:{title}]",
        f"[ar:{artist}]",
        f"[al:{album}]",
        "[by:lyrics-fetcher]",
        "",
    ]
    for t, text in timed_lines:
        m, s = divmod(t, 60)
        lines.append(f"[{int(m):02d}:{s:05.2f}]{text}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def html_template(title, artist, album, body_lines_html):
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>{escape(title)} - {escape(artist)}</title>
<style>
  body {{ font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
         max-width: 760px; margin: 0 auto; padding: 24px; background: #111; color: #eee; }}
  h1 {{ font-size: 1.4em; }} .meta {{ color: #999; font-size: .9em; margin-bottom: 24px; }}
  .lyrics-line {{ margin: 14px 0; line-height: 2.2; font-size: 1.05em; }}
  ruby {{ ruby-position: over; }}
  rt {{ font-size: .55em; color: #8ab; font-weight: normal; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<p class="meta">{escape(artist)} ・ {escape(album)}</p>
{body_lines_html}
</body>
</html>
"""


def render_furigana_line(line: str, ruby_map: dict[str, str]) -> str:
    """Wrap kanji runs with readings using utaten's ruby pairs.

    ruby_map: kanji_word -> reading (from utaten, already word-aligned).
    Simple greedy longest-match scan over the line.
    """
    out = []
    i = 0
    # sort keys by length desc for longest-match-first
    keys = sorted(ruby_map, key=len, reverse=True)
    while i < len(line):
        matched = False
        for k in keys:
            if line.startswith(k, i):
                out.append(f"<ruby>{escape(k)}<rt>{escape(ruby_map[k])}</rt></ruby>")
                i += len(k)
                matched = True
                break
        if not matched:
            out.append(escape(line[i]))
            i += 1
    return "".join(out)


def main():
    print("=== PoC: Output Generation ===\n")
    OUT.mkdir(exist_ok=True)

    # 1) Parse the existing LRC that shipped with the album
    existing = MUSIC_DIR / "VOCALOID 超BEST -memories-" / "Neru feat. 鏡音リン - ロストワンの号哭.lrc"
    if existing.exists():
        raw = existing.read_text(encoding="utf-8", errors="replace")
        timed = re.findall(r"\[(\d{2}):(\d{2})\.(\d{2})\](.*)", raw)
        print(f"1) Parsed existing LRC: {existing.name}")
        print(f"   {len(timed)} timed lines, first 3:")
        for m, s, cs, text in timed[:3]:
            print(f"     [{m}:{s}.{cs}]{text.strip()}")
        print()

    # 2) Fake timestamps (in real pipeline these come from alignment) + write LRC
    demo_lines = [
        "僕がずっと前から思ってる事を話そうか",
        "友達に戻れたらこれ以上はもう望まないさ",
        "君がそれでいいなら僕だってそれで構わないさ",
    ]
    timed_lines = [(i * 5.0, l) for i, l in enumerate(demo_lines)]  # placeholder timing
    lrc_path = write_lrc(OUT / "demo_天ノ弱.lrc", "天ノ弱", "164 feat. 初音ミク", "VOCALOID 超BEST", timed_lines)
    print(f"2) Wrote LRC: {lrc_path}")
    print(lrc_path.read_text(encoding="utf-8"))
    print()

    # 3) HTML with furigana from utaten ruby pairs (sample pairs from fetch)
    ruby_map = {"僕": "ぼく", "前": "まえ", "思": "おも", "話": "はな", "友達": "ともだち"}
    body = "\n".join(
        f'<div class="lyrics-line" data-start="{t:.2f}">{render_furigana_line(l, ruby_map)}</div>'
        for t, l in timed_lines
    )
    html_path = OUT / "demo_天ノ弱.html"
    html_path.write_text(html_template("天ノ弱", "164 feat. 初音ミク", "VOCALOID 超BEST", body), encoding="utf-8")
    print(f"3) Wrote HTML: {html_path}")
    print("   snippet:", body.splitlines()[0][:140])
    print("\nOpen out/demo_天ノ弱.html in a browser to check furigana rendering.")


if __name__ == "__main__":
    main()

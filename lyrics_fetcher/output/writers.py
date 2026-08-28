"""Output writers — .lrc and companion .html with furigana.

LRC is kept standard (Jellyfin/media-player compatible). Furigana <ruby> markup
goes only in the companion HTML file.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from ..aligner.base import TimedLine
from ..models import Lyrics

DEFAULT_BY = "lyrics-fetcher"


class LrcWriter:
    def write(self, path: Path, title: str, artist: str, album: str,
              timed: list[TimedLine], by: str = DEFAULT_BY) -> Path:
        lines = [f"[ti:{title}]", f"[ar:{artist}]", f"[al:{album}]",
                 f"[by:{by}]", ""]
        for tl in timed:
            m, s = divmod(tl.start, 60)
            lines.append(f"[{int(m):02d}:{s:05.2f}]{tl.text}")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


class HtmlWriter:
    def __init__(self, lyrics: Lyrics | None = None):
        # optional ruby map for furigana (from utaten fetch)
        self.ruby_map = lyrics.ruby_all if lyrics else {}

    @staticmethod
    def _furigana_line(line: str, ruby_map: dict[str, str]) -> str:
        """Wrap kanji runs with <ruby> readings using longest-match-first."""
        out = []
        i = 0
        keys = sorted(ruby_map, key=len, reverse=True)
        while i < len(line):
            matched = False
            for k in keys:
                if line.startswith(k, i):
                    out.append(f"<ruby>{escape(k)}<rp>(</rp><rt>{escape(ruby_map[k])}</rt><rp>)</rp></ruby>")
                    i += len(k)
                    matched = True
                    break
            if not matched:
                out.append(escape(line[i]))
                i += 1
        return "".join(out)

    def write(self, path: Path, title: str, artist: str, album: str,
              timed: list[TimedLine]) -> Path:
        body = "\n".join(
            f'<div class="lyrics-line" data-start="{tl.start:.2f}">'
            f'{self._furigana_line(tl.text, self.ruby_map)}</div>'
            for tl in timed
        )
        html = f"""<!DOCTYPE html>
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
{body}
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
        return path
"""Align a curated lyrics.txt onto an audio file and write Jellyfin .lrc/.html."""
import sys
from pathlib import Path

from lyrics_fetcher.models import LyricLine, Lyrics
from lyrics_fetcher.pipeline import Pipeline
from lyrics_fetcher.aligner.stable_ts import StableTSAligner

audio = Path(sys.argv[1])
txt = Path(sys.argv[2])
title, artist, source = sys.argv[3], sys.argv[4], sys.argv[5]

lines = [LyricLine(text=x) for x in txt.read_text(encoding="utf-8").splitlines() if x.strip()]
l = Lyrics(source=source, title=title, artist=artist, lines=lines)
pipe = Pipeline(aligner=StableTSAligner())
res = pipe.run(audio=audio, jellyfin=True, lyrics=l)
print(f"OK -> {res.lrc_path} | {res.lines} lines | source={res.lyrics_source}")

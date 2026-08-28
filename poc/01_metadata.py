"""PoC: Read metadata from audio files.

Tests that we can extract title/artist/album from FLAC files using mutagen.
This is the foundation for everything else — we need to know what song we're working with.
"""

from pathlib import Path
from mutagen.flac import FLAC

MUSIC_DIR = Path("/mnt/fnos/storage/Music")


def read_metadata(filepath: Path) -> dict:
    """Extract metadata tags from an audio file."""
    audio = FLAC(str(filepath))
    tags = {}
    for key in ["title", "artist", "album", "albumartist", "tracknumber", "date"]:
        # mutagen uses ID3-style keys; FLAC uses Vorbis comments
        if key == "albumartist":
            val = audio.get("ALBUMARTIST") or audio.get("album_artist")
        elif key == "tracknumber":
            val = audio.get("track") or audio.get("tracknumber")
        else:
            # Try both cases
            val = audio.get(key) or audio.get(key.upper())
        if val:
            tags[key] = str(val[0]) if isinstance(val, list) else str(val)
    return tags


def main():
    print("=== PoC: Metadata Reading ===\n")

    # Test with vocaloid track (well-tagged)
    vocaloid_dir = MUSIC_DIR / "VOCALOID 超BEST -memories-"
    maimai_dir = MUSIC_DIR / "maimai でらっくす ベストアルバムちほー3"
    eo_dir = MUSIC_DIR / "EO 8番出口"

    test_files = []
    for d in [vocaloid_dir, maimai_dir, eo_dir]:
        if d.exists():
            files = sorted(d.glob("*.flac"))[:2]
            test_files.extend(files)

    print(f"Testing {len(test_files)} files:\n")
    for f in test_files:
        meta = read_metadata(f)
        print(f"  {f.name}")
        print(f"    title:   {meta.get('title', 'N/A')}")
        print(f"    artist:  {meta.get('artist', 'N/A')}")
        print(f"    album:   {meta.get('album', 'N/A')}")
        print(f"    track:   {meta.get('tracknumber', 'N/A')}")
        print()

    # Also test the existing .lrc file that's already there
    lrc_files = list(vocaloid_dir.glob("*.lrc"))
    if lrc_files:
        print(f"\nExisting LRC found: {lrc_files[0].name}")
        with open(lrc_files[0], "r") as fh:
            content = fh.read()
        print(content[:500])


if __name__ == "__main__":
    main()

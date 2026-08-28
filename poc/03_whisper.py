"""PoC: Whisper-based speech-to-text on a song.

Tests whether faster-whisper can extract lyrics from a vocaloid track.
This is the "AI recognition" fallback for when no online source has the lyrics.

Uses the 'base' model first (fast, ~1GB RAM) to see if it's viable at all.
If quality is acceptable, we'd upgrade to 'small' or 'medium'.
"""

from pathlib import Path
from faster_whisper import WhisperModel

MUSIC_DIR = Path("/mnt/fnos/storage/Music")


def main():
    print("=== PoC: Whisper Lyrics Recognition ===\n")

    # Pick a vocaloid track (Japanese, should be recognizable)
    test_file = MUSIC_DIR / "VOCALOID 超BEST -memories-" / "08 天ノ弱.flac"
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    print(f"Testing: {test_file.name}")
    print("Loading model (base, int8 on CPU — swap to CUDA/ROCm for speed)...\n")

    # Load model
    # device="cuda" would use GPU; "cpu" for now since ROCm support in CTranslate2 is finicky
    model = WhisperModel("base", device="cpu", compute_type="int8")

    print("Transcribing (this may take a minute)...\n")

    # Transcribe with word timestamps
    segments, info = model.transcribe(
        str(test_file),
        language="ja",  # force Japanese
        vad_filter=True,  # skip silence
        beam_size=5,
    )

    print(f"Detected language: {info.language} (p={info.language_probability:.2f})")
    print(f"Duration: {info.duration:.1f}s\n")
    print("--- Transcription ---")

    lines = []
    for segment in segments:
        line = f"[{segment.start:07.2f} -> {segment.end:07.2f}] {segment.text.strip()}"
        lines.append(line)
        print(line)

    print(f"\n--- Total: {len(lines)} segments ---")
    print("\nAssessment:")
    print("- Are the Japanese characters correct?")
    print("- Are line breaks reasonable (one per sung phrase)?")
    print("- Any obvious errors (wrong words, missing syllables)?")


if __name__ == "__main__":
    main()

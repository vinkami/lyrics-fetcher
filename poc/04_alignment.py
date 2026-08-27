"""PoC: Forced alignment — map known lyrics to audio timestamps.

Tests whether we can take a known lyrics text and align it to the audio
to produce per-line timestamps. This is the core of Part 2 (timestamp compilation).

Approach: Use faster-whisper's word-level timestamps, then fuzzy-match
words from the known lyrics against whisper's output to build line timings.

Alternative tested here: simple energy-based segmentation + whisper alignment.
"""

from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel

MUSIC_DIR = Path("/mnt/fnos/storage/Music")


def get_energy_segments(filepath: str, hop_length: int = 512, sr: int = 16000) -> list[tuple[float, float]]:
    """Split audio into segments at low-energy points (likely phrase boundaries)."""
    import soundfile as sf
    data, sr_actual = sf.read(filepath, dtype="float32")
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Compute short-time energy
    n_frames = len(data) // hop_length
    energies = np.array([
        np.sum(data[i*hop_length:(i+1)*hop_length] ** 2)
        for i in range(n_frames)
    ])

    # Find low-energy points as potential segment boundaries
    threshold = np.percentile(energies, 20)
    segments = []
    start = 0
    for i in range(len(energies)):
        if energies[i] < threshold:
            end_time = (i + 1) * hop_length / sr_actual
            if end_time - start > 0.5:  # min 0.5s segment
                segments.append((start, end_time))
                start = end_time
    return segments


def align_lyrics_to_audio(lyrics_lines: list[str], filepath: str) -> list[tuple[float, str]]:
    """
    Align known lyrics lines to audio using whisper word timestamps.
    
    Strategy:
    1. Get word-level timestamps from whisper
    2. For each lyrics line, find the best-matching sequence of words
    3. Use the start time of the first matched word as the line timestamp
    """
    model = WhisperModel("base", device="cpu", compute_type="int8")

    # Get word-level timestamps
    segments, info = model.transcribe(
        filepath,
        language="ja",
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
    )

    # Collect all words with their timestamps
    all_words = []  # (start, end, text)
    for seg in segments:
        if seg.words:
            for w in seg.words:
                all_words.append((w.start, w.end, w.word.strip()))

    print(f"Whisper produced {len(all_words)} words")
    print(f"Lyrics has {len(lyrics_lines)} lines\n")

    # Simple alignment: for each line, find the first word that matches
    # (This is a rough PoC — real implementation would use DP/Hungarian matching)
    results = []
    word_idx = 0
    for i, line in enumerate(lyrics_lines):
        # Find the next occurrence of any significant word from this line
        # (crude: just look for the first word that shares a character)
        line_chars = set(line.replace(" ", ""))
        best_match = None
        for j in range(word_idx, min(word_idx + 20, len(all_words))):
            w_start, w_end, w_text = all_words[j]
            if any(c in line_chars for c in w_text if c.isalpha()):
                best_match = (w_start, w_end)
                word_idx = j + 1
                break

        if best_match:
            results.append((best_match[0], line))
        else:
            # No match found — estimate from position
            est_time = i * (info.duration / max(len(lyrics_lines), 1))
            results.append((est_time, line))

    return results


def main():
    print("=== PoC: Forced Alignment ===\n")

    test_file = MUSIC_DIR / "VOCALOID 超BEST -memories-" / "08 天ノ弱.flac"
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    # Known lyrics for 天ノ弱 (first few lines, from public knowledge)
    # In production these would come from the fetch step
    known_lyrics = [
        "このままじゃいられない",
        "この胸の鼓動が",
        "君を呼んでいる",
        "止まることのない",
        "時の流れの中で",
    ]

    print(f"Testing: {test_file.name}")
    print(f"Known lyrics lines: {len(known_lyrics)}")
    print("Running whisper with word timestamps...\n")

    results = align_lyrics_to_audio(known_lyrics, str(test_file))

    print("\n--- Alignment Results ---")
    for time, line in results:
        print(f"  [{time:07.2f}] {line}")

    print("\nAssessment:")
    print("- Do the timestamps roughly match when each line is sung?")
    print("- How accurate is the word matching?")
    print("- Would this be good enough for LRC generation?")


if __name__ == "__main__":
    main()

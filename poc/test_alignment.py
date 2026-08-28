"""Test: Monotonic (DTW) alignment fixes repeated-chorus collisions.

Verifies that align_lines keeps repeated lyric lines in correct temporal order,
unlike the old greedy best-match.

Uses a synthetic-but-realistic scenario: two identical chorus blocks separated
in time. The greedy version maps both to the SAME (first) segment; the monotonic
DP must map the second chorus to a LATER segment.
"""
import importlib.util
import sys
from pathlib import Path

# import align_lines from 08_pipeline.py (name starts with digit -> importlib)
_p = str(Path(__file__).parent / "08_pipeline.py")
_spec = importlib.util.spec_from_file_location("pipeline08", _p)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
align_lines = _m.align_lines


def make_seg(start_s, end_s, text):
    return {"from": int(start_s * 1000), "to": int(end_s * 1000), "text": text}


def test_repeated_chorus():
    # song ~ verse + CHORUS + verse + CHORUS
    segs = [
        make_seg(0, 4, "シャララ 静寂の夜に"),
        make_seg(4, 8, "揺れる心 まだ眠れない"),
        make_seg(8, 12, "ずっと綺麗なもの 変わらぬまま色褪せはしないから"),   # chorus 1a
        make_seg(12, 16, "只々生きているの くたばる日まで"),                 # chorus 1b
        make_seg(16, 20, "根拠なんてない説明もしない 遠くに残響"),           # chorus 1c
        make_seg(20, 24, "語り継ぐ 古の調べ"),
        make_seg(24, 28, "涙の跡 乾く前に"),
        make_seg(28, 32, "ずっと綺麗なもの 変わらぬまま色褪せはしないから"),   # chorus 2a
        make_seg(32, 36, "只々生きているの くたばる日まで"),                 # chorus 2b
        make_seg(36, 40, "根拠なんてない説明もしない 遠くに残響"),           # chorus 2c
    ]
    # NOTE: this synthetic case is for the ALGORITHM test; it won't be re-run with
    # real whisper (that's below). Verify monotonic ordering:
    known = [
        "シャララ 静寂の夜に",
        "揺れる心 まだ眠れない",
        "ずっと綺麗なもの 変わらぬまま色褪せはしないから",
        "只々生きているの くたばる日まで",
        "根拠なんてない説明もしない 遠くに残響",
        "語り継ぐ 古の調べ",
        "涙の跡 乾く前に",
        "ずっと綺麗なもの 変わらぬまま色褪せはしないから",
        "只々生きているの くたばる日まで",
        "根拠なんてない説明もしない 遠くに残響",
    ]
    timed = align_lines(known, segs)
    times = [t for t, _ in timed]
    # core assertion: times must be non-decreasing
    assert all(times[i] <= times[i+1] for i in range(len(times)-1)), times
    # the 7th line (2nd chorus 1a) must be strictly AFTER the 3rd line (1st chorus 1a)
    first_chorus_a = times[2]
    second_chorus_a = times[7]
    assert second_chorus_a > first_chorus_a, (
        f"repeated chorus not separated: 1st@{first_chorus_a}s, 2nd@{second_chorus_a}s"
    )
    print("PASS: repeated chorus kept in order")
    print("  chorus1a @", first_chorus_a, "s, chorus2a @", second_chorus_a, "s")


if __name__ == "__main__":
    test_repeated_chorus()
    print("\nAll monotonic-alignment assertions passed.")
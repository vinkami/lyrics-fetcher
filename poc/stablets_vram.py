# poc/stablets_vram.py — Phase 0 metrics: VRAM peak + eGPU compute sanity
import os
import re
import time

import torch
import stable_whisper
import whisper

m = stable_whisper.load_model("medium", device="cuda:0")
audio = whisper.load_audio("_sep_out/03 告げよ_vocals.wav")
raw = open("_lrc_re-ocr/03 告げよ.lrc", encoding="utf-8").read()
text = "\n".join(
    re.match(r"^\[\d\d:\d\d(?:\.\d+)?\](.*)$", ln.strip()).group(1).strip()
    for ln in raw.splitlines()
    if re.match(r"^\[\d\d:\d\d(?:\.\d+)?\]", ln.strip())
)
torch.cuda.reset_peak_memory_stats(0)
t0 = time.time()
r = m.align(audio, text, language="ja", regroup="p", verbose=False)
print(f"align {time.time()-t0:.1f}s, peak VRAM dev0 = {torch.cuda.max_memory_allocated(0)/2**30:.2f} GiB")

try:
    a = torch.randn(2048, 2048, device="cuda:2") @ torch.randn(2048, 2048, device="cuda:2")
    torch.cuda.synchronize(2)
    print("eGPU dev2 matmul OK", tuple(a.shape), torch.cuda.get_device_name(2))
except Exception as e:
    print("eGPU dev2 FAILED:", type(e).__name__, str(e)[:200])

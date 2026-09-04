"""Vocal separation — preprocess audio to a clean vocal stem before alignment.

Wraps demucs (facebookresearch, ``htdemucs``) running on the GPU (CUDA/ROCm)
or CPU. Separating
out the accompaniment before whisper alignment removes BGM-driven hallucination
and lets dense/intro timing lock onto real anchors.

Measured on dense-BGM albums (2026-09-02): long spoken/instrumental intros
anchor correctly only on the separated stem — raw alignment even-spreads
them. It is an OPT-IN
feature (``--separation``), not default, because it can shift already-good songs
(a few seconds mid-song while keeping the same anchor count).

demucs is imported lazily so the dependency is optional: tests / non-separating
runs never touch it, and the model only downloads on first use (~80 MB htdemucs,
into the huggingface cache — relocatable via [separation] separation_model_dir).
"""
from __future__ import annotations

from pathlib import Path

from .config import settings


class VocalSeparator:
    """Separate the vocal stem from an audio file using demucs htdemucs."""

    name = "demucs"

    def __init__(self, device: str | None = None, model: str | None = None,
                 model_dir: Path | str | None = None):
        # None => config layer decides (settings.*), "auto" => torch probe
        self._device = device
        self._model = model
        self._model_dir = Path(model_dir) if model_dir else None
        self._sep = None

    def _get(self):
        if self._sep is None:
            import torch  # ROCm exposes the card as cuda

            import demucs.api

            device = self._device or settings.separation_device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = self._model or settings.separation_model
            # Model weights download on first use (~80 MB htdemucs) via
            # huggingface_hub. Direct that cache to [separation].
            # separation_model_dir when configured (HF reads the env var at
            # import time inside its lazy downloader), else demucs' repo=
            # LocalRepo for a manually fetched folder.
            mdir = self._model_dir or settings.separation_model_dir
            kwargs = {}
            if mdir is not None:
                mdir = Path(mdir).expanduser()
                if (mdir / "bin").exists() or any(mdir.glob("*.th")):
                    # already a demucs LocalRepo layout -> use it directly
                    kwargs["repo"] = mdir
                else:
                    mdir.mkdir(parents=True, exist_ok=True)
                    import os
                    os.environ["HF_HUB_CACHE"] = str(mdir)
                    # if huggingface_hub was already imported, its constants
                    # captured the env at import time — rewrite them too
                    try:
                        import huggingface_hub.constants as _hfc
                        _hfc.HF_HUB_CACHE = str(mdir)
                    except ImportError:
                        pass
            self._sep = demucs.api.Separator(model=model, device=device, **kwargs)
        return self._sep

    def separate(self, audio: Path, out_dir: Path | None = None) -> Path:
        """Separate the vocal stem; returns the path to a vocal .wav.

        The stem is written as ``<out_dir>/<audio.stem>_vocals.wav`` (default
        ``out_dir`` = the audio's parent dir). Returns the stem path.
        """
        sep = self._get()
        _, stems = sep.separate_audio_file(str(audio))
        vocals = stems["vocals"]
        target = (out_dir or audio.parent) / f"{audio.stem}_vocals.wav"
        import soundfile as sf

        sf.write(str(target), vocals.cpu().numpy().T, sep.samplerate)
        return target


def make_separator(device: str | None = None, model: str | None = None,
                   model_dir: Path | str | None = None) -> VocalSeparator | None:
    """Construct a separator, or None if the optional demucs dep is missing.

    Used so ``--separation`` degrades gracefully when demucs isn't installed
    instead of crashing the whole command.
    """
    try:
        return VocalSeparator(device=device, model=model, model_dir=model_dir)
    except ImportError:
        return None
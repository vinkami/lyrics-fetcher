"""Configuration for lyrics-fetcher.

Settings come from (highest to lowest precedence):
  1. CLI arguments (parsed in cli.py, overlays these)
  2. A config file (TOML) given via --config / env LF_CONFIG
  3. Defaults defined here

Model/media paths are the only things that vary between machines; keeping them
in one place makes the tool portable. See config.example.toml for a template.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

#: default config file search locations
CONFIG_FILENAMES = ("config.toml", "lyrics-fetcher.toml")


@dataclass
class Settings:
    # --- media / data ---
    music_dir: Path = Path("/mnt/fnos/storage/Music")
    cache_db: Path = Path.home() / ".cache" / "lyrics-fetcher" / "cache.db"
    out_dir: Path = Path("out")

    # --- whisper.cpp ---
    whisper_bin: Path = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    whisper_model: Path = Path.home() / "whisper.cpp" / "models" / "ggml-medium.bin"
    whisper_extra_models: list[Path] = field(default_factory=lambda: [
        Path.home() / "whisper.cpp" / "models" / "ggml-large-v3-turbo.bin",
    ])
    whisper_lang: str = "ja"
    whisper_max_len: int = 40
    whisper_device: int = 0

    # --- Qwen vision OCR (llama-server) ---
    vision_api: str = "http://127.0.0.1:8081/v1/chat/completions"
    vision_model: str = "qwen3.5-9b"

    # --- Qwen3-ForcedAligner ---
    qwen3_aligner_model: Path = Path("/mnt/fnos/storage/ai-models/qwen3-forcedaligner/model")
    qwen3_aligner_language: str = "Japanese"

    # --- output ---
    lrc_by: str = "lyrics-fetcher"
    jellyfin_default: bool = False
    write_html_default: bool = True

    # --- tuning ---
    anchor_min_score: float = 58.0
    request_timeout: int = 20


#: the active settings (populated by load(); modules read this singleton)
#: NOTE: load() mutates this object in place rather than rebinding it, so
#: modules that did `from ..config import settings` at import time always see
#: the live values (important when load() runs after imports, e.g. in CLI main).
settings: Settings = Settings()


def _paths_equal(p: Path) -> bool:
    """True if the path is the default-looking sentinel (used for None-ish check)."""
    return str(p) in ("", "auto", "default")


def config_file() -> Path | None:
    """Locate a config file: $LF_CONFIG, or a CONFIG_FILENAMES in an
    XDG/standard location, or in the current dir."""
    override = os.environ.get("LF_CONFIG")
    if override:
        p = Path(override)
        if p.exists():
            return p
        return None
    # XDG config dir
    xdg = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    for cwd_candidate in (Path.cwd(), Path.home() / ".config" / "lyrics-fetcher"):
        for name in CONFIG_FILENAMES:
            cand = cwd_candidate / name
            if cand.exists():
                # prefer a project-local config over cwd
                return cand
    return None


def load(path: Path | None = None, overrides: dict | None = None) -> Settings:
    """Load settings from a TOML file (or defaults) and apply dict overrides.

    MUTATES the module-level ``settings`` singleton in place (so modules that
    imported it earlier see the change) and returns it.

    Args:
        path: explicit config file; if None, auto-detect via :func:`config_file`.
        overrides: flat dict of dotted keys, e.g. {"whisper_model": "/x"} or
            {"vision.api": "..."}. CLI args get merged as these.
    """
    global settings
    data: dict = {}
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    else:
        cf = config_file()
        if cf:
            with open(cf, "rb") as fh:
                data = tomllib.load(fh)

    fresh = Settings()
    fresh = _apply_toml(fresh, data)
    if overrides:
        fresh = _apply_overrides(fresh, overrides)
    # mutate the existing singleton in place so all importers see the change
    for fld in settings.__dataclass_fields__:
        setattr(settings, fld, getattr(fresh, fld))
    return settings


def _apply_toml(s: Settings, data: dict) -> Settings:
    # section -> attribute mapping
    section_map = {
        "paths": ["music_dir", "cache_db", "out_dir"],
        "whisper": ["whisper_bin", "whisper_model", "whisper_extra_models",
                    "whisper_lang", "whisper_max_len", "whisper_device"],
        "vision": ["vision_api", "vision_model"],
        "qwen3_aligner": ["qwen3_aligner_model", "qwen3_aligner_language"],
        "output": ["lrc_by", "jellyfin_default", "write_html_default"],
        "tuning": ["anchor_min_score", "request_timeout"],
    }
    for section, attrs in section_map.items():
        sd = data.get(section)
        if not isinstance(sd, dict):
            continue
        for a in attrs:
            if a in sd:
                v = sd[a]
                setattr(s, a, _coerce(s, a, v))
    return s


def _apply_overrides(s: Settings, overrides: dict) -> Settings:
    for key, val in overrides.items():
        if key in s.__dataclass_fields__:
            setattr(s, key, _coerce(s, key, val))
        else:
            # dotted section.key form
            section, _, attr = key.partition(".")
            sm = {
                "paths": ["music_dir", "cache_db", "out_dir"],
                "whisper": ["whisper_bin", "whisper_model", "whisper_extra_models",
                            "whisper_lang", "whisper_max_len", "whisper_device"],
                "vision": ["vision_api", "vision_model"],
                "qwen3_aligner": ["qwen3_aligner_model", "qwen3_aligner_language"],
                "output": ["lrc_by", "jellyfin_default", "write_html_default"],
                "tuning": ["anchor_min_score", "request_timeout"],
            }.get(section, [])
            if attr in sm:
                setattr(s, attr, _coerce(s, attr, val))
    return s


def _coerce(s: Settings, attr: str, v):
    """Convert a TOML/CLI value to the dataclass field type."""
    default = s.__dataclass_fields__[attr].default
    if isinstance(default, Path):
        if isinstance(v, list):
            return [Path(x.strip("'\"") + "") for x in v] if v else []
        return Path(str(v).strip("'\""))
    if isinstance(default, list) and all(isinstance(x, Path) for x in default):
        if isinstance(v, str):
            return [Path(x.strip()) for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [Path(x) for x in v]
        return []
    if isinstance(default, bool):
        # bool is an int subclass; handle BEFORE int so True stays bool
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    return v
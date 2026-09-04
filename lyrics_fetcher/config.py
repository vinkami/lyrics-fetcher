"""Configuration for lyrics-fetcher.

Settings come from (highest to lowest precedence):
  1. CLI arguments (parsed in cli.py, overlays these)
  2. Environment variables: LYRICS_FETCHER_<SECTION>_<ATTR>
     (plus the bare alias VISION_API_KEY for the OCR endpoint secret)
  3. A config file (TOML) given via --config / env LF_CONFIG
  4. Defaults defined here

Secrets (API keys) must NOT live in the TOML, which people commit by accident:
put them in a gitignored ``.env`` (loaded here via python-dotenv, real env
vars win) or export them. All keys are redacted from logs/reprs.

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

#: section -> settings attributes; single source for TOML, env and CLI overlays
SECTION_MAP = {
    "paths": ["music_dir", "cache_db", "out_dir"],
    "whisper": ["whisper_bin", "whisper_model", "whisper_extra_models",
                "whisper_lang", "whisper_max_len", "whisper_device"],
    "vision": ["vision_api", "vision_model", "vision_api_key"],
    "qwen3_aligner": ["qwen3_aligner_model", "qwen3_aligner_language"],
    "stable_ts": ["stable_ts_model", "stable_ts_lang", "stable_ts_device"],
    "output": ["lrc_by", "jellyfin_default", "write_html_default"],
    "tuning": ["anchor_min_score", "request_timeout"],
}


@dataclass
class Settings:
    # --- media / data ---
    music_dir: Path = Path.home() / "Music"
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

    # --- vision OCR (any OpenAI-compatible endpoint; base URL ends at /v1,
    #     the /chat/completions path is appended by the code) ---
    vision_api: str = "http://127.0.0.1:8081/v1"
    vision_model: str = "qwen3.5-9b"
    # Bearer token for cloud endpoints. Leave empty in TOML; set via
    # VISION_API_KEY in .env (gitignored) — never commit real keys.
    # repr=False so it can't leak via a stray `print(settings)`.
    vision_api_key: str = field(default="", repr=False)

    # --- Qwen3-ForcedAligner ---
    # local dir holding the model snapshot; an HF hub repo id also works
    # (transformers downloads it on first use)
    qwen3_aligner_model: Path = Path.home() / ".cache" / "lyrics-fetcher" / "models" / "qwen3-forcedaligner"
    qwen3_aligner_language: str = "Japanese"

    # --- stable-ts (opt-in forced alignment, --aligner stable-ts) ---
    stable_ts_model: str = "medium"
    stable_ts_lang: str = "ja"
    stable_ts_device: str = "cuda"  # primary GPU (CUDA or ROCm); "cpu" also works

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


def _apply_env(s: Settings) -> Settings:
    """Env-var layer: LYRICS_FETCHER_<SECTION>_<ATTR> (highest except CLI).

    Plus the bare convenience alias VISION_API_KEY for the OCR endpoint
    secret, matching what .env files conventionally carry.
    """
    for section, attrs in SECTION_MAP.items():
        for a in attrs:
            v = os.environ.get(f"LYRICS_FETCHER_{section.upper()}_{a.upper()}")
            if v is not None:
                setattr(s, a, _coerce(s, a, v))
    alias = os.environ.get("VISION_API_KEY")
    if alias is not None:
        s.vision_api_key = alias
    return s


def load_env_file(path: Path | None = None) -> None:
    """Load a .env into os.environ via python-dotenv, real env vars winning.

    Search order: explicit path, $PWD/.env, ~/.config/lyrics-fetcher/.env.
    No-ops when python-dotenv is absent (exported vars still work) or no
    file exists. The .env must be gitignored — keys never live in TOML.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if path is None:
        for cand in (Path.cwd() / ".env",
                     Path.home() / ".config" / "lyrics-fetcher" / ".env"):
            if cand.exists():
                path = cand
                break
    if path is not None and path.exists():
        load_dotenv(path, override=False)


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
    load_env_file()
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
    fresh = _apply_env(fresh)
    if overrides:
        fresh = _apply_overrides(fresh, overrides)
    # mutate the existing singleton in place so all importers see the change
    for fld in settings.__dataclass_fields__:
        setattr(settings, fld, getattr(fresh, fld))
    return settings


def _apply_toml(s: Settings, data: dict) -> Settings:
    for section, attrs in SECTION_MAP.items():
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
            if attr in SECTION_MAP.get(section, []):
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
"""Tests for the config module — loading, precedence, and type coercion."""
import os
import tempfile
from pathlib import Path

import pytest

from lyrics_fetcher.config import Settings, load, settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate config tests from the developer's real environment.

    load() now applies an env-var layer, and load_env_file() may pick up a
    cwd ~/.env — neither must leak into (or mask) these assertions.
    """
    for k in list(os.environ):
        if k.startswith("LYRICS_FETCHER_") or k == "VISION_API_KEY":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tempfile.mkdtemp())


def _write_config(text: str) -> Path:
    d = tempfile.mkdtemp()
    p = Path(d) / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


CONFIG = """[whisper]
whisper_model = "/custom/model.gguf"
whisper_bin = "/custom/whisper-cli"
whisper_lang = "en"

[vision]
vision_api = "http://localhost:9999/v1/chat/completions"
vision_model = "test-vision"

[qwen3_aligner]
qwen3_aligner_model = "/custom/qwen3/aligner"
qwen3_aligner_language = "Korean"

[output]
lrc_by = "myplugin"
jellyfin_default = true

[paths]
music_dir = "/custom/music"
"""


def test_load_applies_all_sections():
    load(_write_config(CONFIG))
    assert str(settings.whisper_model) == "/custom/model.gguf"
    assert str(settings.whisper_bin) == "/custom/whisper-cli"
    assert settings.whisper_lang == "en"
    assert settings.vision_api == "http://localhost:9999/v1/chat/completions"
    assert settings.vision_model == "test-vision"
    assert str(settings.qwen3_aligner_model) == "/custom/qwen3/aligner"
    assert settings.qwen3_aligner_language == "Korean"
    assert settings.lrc_by == "myplugin"
    assert settings.jellyfin_default is True
    assert str(settings.music_dir) == "/custom/music"


def test_bool_coerced_properly_not_as_int():
    # bool is an int subclass; must stay a real bool
    load(_write_config("[output]\njellyfin_default = true\n"))
    assert settings.jellyfin_default is True
    assert settings.jellyfin_default == 1  # but not identical-type surprise
    load(_write_config("[output]\njellyfin_default = false\n"))
    assert settings.jellyfin_default is False


def test_missing_sections_keep_defaults():
    load(_write_config("[output]\nlrc_by = 'x'\n"))
    # unrelated defaults untouched
    assert str(settings.whisper_model) == str(Settings().whisper_model)
    assert settings.whisper_lang == Settings().whisper_lang


def test_aligners_resolve_config_at_construction():
    load(_write_config(CONFIG))
    from lyrics_fetcher.aligner.whisper_cpp import WhisperCppAligner
    from lyrics_fetcher.ocr.vision import VLMOcr
    from lyrics_fetcher.aligner.qwen3_forced_aligner import Qwen3ForcedAligner

    assert str(WhisperCppAligner().model) == "/custom/model.gguf"
    assert str(WhisperCppAligner().binary) == "/custom/whisper-cli"
    assert WhisperCppAligner().lang == "en"
    assert VLMOcr().api == "http://localhost:9999/v1/chat/completions"
    assert VLMOcr().model == "test-vision"
    assert str(Qwen3ForcedAligner().model_dir).startswith("/custom/qwen3/aligner")
    assert Qwen3ForcedAligner().language == "Korean"


def test_missing_config_file_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load(Path("/nonexistent/config.toml"))

# ---- env-var layer + .env secrets ----
def test_env_overrides_toml(monkeypatch):
    monkeypatch.setenv("LYRICS_FETCHER_VISION_VISION_MODEL", "env-model")
    load(_write_config("[vision]\nvision_model = 'toml-model'\n"))
    assert settings.vision_model == "env-model"


def test_vision_api_key_alias(monkeypatch):
    monkeypatch.setenv("VISION_API_KEY", "from-env")
    load(_write_config("[vision]\nvision_api_key = 'from-toml'\n"))
    # env alias wins over TOML (keys belong in env/.env, not the file)
    assert settings.vision_api_key == "from-env"


def test_key_absent_from_repr():
    s = Settings()
    object.__setattr__(s, "vision_api_key", "super-secret-value")
    assert "super-secret-value" not in repr(s)


def test_load_env_file_reads_dotenv(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("VISION_API_KEY=from-dotenv-file\n")
    from lyrics_fetcher.config import load_env_file
    load_env_file()
    assert os.environ["VISION_API_KEY"] == "from-dotenv-file"
    load()
    assert settings.vision_api_key == "from-dotenv-file"


def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("VISION_API_KEY=from-dotenv-file\n")
    monkeypatch.setenv("VISION_API_KEY", "from-exported")
    from lyrics_fetcher.config import load_env_file
    load_env_file()
    assert os.environ["VISION_API_KEY"] == "from-exported"


def test_vlmocr_resolves_key_from_config(monkeypatch):
    load(_write_config("[vision]\nvision_api_key = 'from-config'\n"))
    from lyrics_fetcher.ocr.vision import VLMOcr
    o = VLMOcr()
    assert o.api_key == "from-config"
    # local default: no key -> no Authorization header at all
    load(_write_config(""))
    assert VLMOcr().api_key == ""

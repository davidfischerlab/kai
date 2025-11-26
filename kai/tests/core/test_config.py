from kai.config.settings import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.DEFAULT_LLM_PROVIDER == "ollama"
    assert settings.VERBOSE is True
    base_dir_str = str(settings.BASE_DIR)
    assert settings.BASE_DIR.exists() or base_dir_str.startswith("/home")


def test_settings_env(monkeypatch):
    monkeypatch.setenv("KAI_VERBOSE", "false")
    settings = Settings.from_env()
    assert settings.VERBOSE is False 
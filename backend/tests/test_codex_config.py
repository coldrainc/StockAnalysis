from pathlib import Path

from stock_agent.core.codex_config import load_codex_model_config


MODEL_ENV_KEYS = [
    "STOCK_MODEL_PROVIDER",
    "MODEL_PROVIDER",
    "STOCK_MODEL",
    "STOCK_MODEL_BASE_URL",
    "STOCK_MODEL_API_KEY_ENV",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "GATEWAY_API_KEY",
]


def clear_model_env(monkeypatch) -> None:
    for key in MODEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_codex_project_model_config(tmp_path: Path, monkeypatch) -> None:
    clear_model_env(monkeypatch)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    project = tmp_path / "project"
    codex_dir = project / ".codex"
    codex_dir.mkdir(parents=True)
    (codex_dir / "config.toml").write_text('model = "gpt-4.1-mini"\n', encoding="utf-8")

    config = load_codex_model_config(project)

    assert config.model == "gpt-4.1-mini"
    assert config.provider == "openai"
    assert config.env_key == "OPENAI_API_KEY"


def test_load_codex_user_base_url(tmp_path: Path, monkeypatch) -> None:
    clear_model_env(monkeypatch)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    (codex_home / "config.toml").write_text(
        'model = "gpt-4o-mini"\nopenai_base_url = "https://example.test/v1"\n',
        encoding="utf-8",
    )

    config = load_codex_model_config(tmp_path / "project")

    assert config.model == "gpt-4o-mini"
    assert config.base_url == "https://example.test/v1"


def test_load_codex_model_provider_config(tmp_path: Path, monkeypatch) -> None:
    clear_model_env(monkeypatch)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CUSTOM_API_KEY", "secret")
    (codex_home / "config.toml").write_text(
        """
model = "custom-model"
model_provider = "custom"

[model_providers.custom]
base_url = "https://custom.example/v1"
env_key = "CUSTOM_API_KEY"
wire_api = "responses"
""",
        encoding="utf-8",
    )

    config = load_codex_model_config(tmp_path / "project")

    assert config.model == "custom-model"
    assert config.provider == "custom"
    assert config.base_url == "https://custom.example/v1"
    assert config.env_key == "CUSTOM_API_KEY"
    assert config.wire_api == "responses"
    assert config.api_key == "secret"


def test_aigw_provider_defaults_to_gateway_api_key(tmp_path: Path, monkeypatch) -> None:
    clear_model_env(monkeypatch)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("GATEWAY_API_KEY", "gateway-secret")
    (codex_home / "config.toml").write_text(
        """
model = "gpt-5.5"
model_provider = "AIGW"

[model_providers.AIGW]
base_url = "https://aigateway.byteintl.net/v1"
""",
        encoding="utf-8",
    )

    config = load_codex_model_config(tmp_path / "project")

    assert config.provider == "AIGW"
    assert config.env_key == "GATEWAY_API_KEY"
    assert config.api_key == "gateway-secret"


def test_deepseek_provider_defaults_to_deepseek_openai_compatible_api(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clear_model_env(monkeypatch)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("STOCK_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    config = load_codex_model_config(tmp_path / "project")

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-pro"
    assert config.base_url == "https://api.deepseek.com"
    assert config.env_key == "DEEPSEEK_API_KEY"
    assert config.api_key == "deepseek-secret"


def test_deepseek_provider_allows_env_model_and_base_url_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clear_model_env(monkeypatch)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("STOCK_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://custom.deepseek.test")

    config = load_codex_model_config(tmp_path / "project")

    assert config.model == "deepseek-chat"
    assert config.base_url == "https://custom.deepseek.test"

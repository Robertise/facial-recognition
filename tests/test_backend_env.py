"""tests/test_backend_env.py – the repo-root .env loader used on Windows shells."""
from backend.env import load_env_file


def test_env_file_sets_missing_and_keeps_existing(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nINNO_TEST_A=1\nexport INNO_TEST_B="two"\nINNO_TEST_C=from-file\nnot a line\n', encoding="utf-8")
    monkeypatch.delenv("INNO_TEST_A", raising=False)
    monkeypatch.delenv("INNO_TEST_B", raising=False)
    monkeypatch.setenv("INNO_TEST_C", "from-shell")
    loaded = load_env_file(f)
    import os
    assert os.environ["INNO_TEST_A"] == "1" and os.environ["INNO_TEST_B"] == "two"
    assert os.environ["INNO_TEST_C"] == "from-shell"
    assert loaded == {"INNO_TEST_A": "1", "INNO_TEST_B": "two"}
    monkeypatch.delenv("INNO_TEST_A"); monkeypatch.delenv("INNO_TEST_B")


def test_missing_file_is_fine(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == {}


def test_extra_cors_origins(monkeypatch):
    import importlib
    monkeypatch.setenv("INNOVATION_ENABLED", "0")
    # 1. When EXTRA_CORS_ORIGINS is not set
    monkeypatch.delenv("EXTRA_CORS_ORIGINS", raising=False)
    import backend.main as main
    main = importlib.reload(main)
    assert main.cors_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]

    # 2. When EXTRA_CORS_ORIGINS has comma-separated origins with whitespace
    monkeypatch.setenv("EXTRA_CORS_ORIGINS", " http://192.168.1.100:3000, https://app.example.com , http://localhost:3000 ")
    main = importlib.reload(main)
    assert "http://192.168.1.100:3000" in main.cors_origins
    assert "https://app.example.com" in main.cors_origins
    assert main.cors_origins.count("http://localhost:3000") == 1

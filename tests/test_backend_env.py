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

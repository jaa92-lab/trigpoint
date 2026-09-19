import os

from forecaster.env import clean_secret_env


def test_stray_whitespace_is_stripped_from_keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "abc123\n")
    monkeypatch.setenv("ASKNEWS_API_KEY", "  xyz\r\n")
    monkeypatch.setenv("METACULUS_TOKEN", "clean")
    assert sorted(clean_secret_env()) == ["ASKNEWS_API_KEY", "OPENROUTER_API_KEY"]
    assert os.environ["OPENROUTER_API_KEY"] == "abc123"
    assert os.environ["ASKNEWS_API_KEY"] == "xyz"
    assert os.environ["METACULUS_TOKEN"] == "clean"

"""LLM access: key detection, readable errors, and graceful fallback when the provider fails."""
from filingsight import config, llm
from filingsight.graph import ask


class _Boom:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, *_a, **_k):
        raise self.exc


class AuthenticationError(Exception):
    pass


class APIConnectionError(Exception):
    pass


def test_placeholder_and_quoted_keys_are_cleaned():
    assert config._clean_key('  "abc123"  ') == "abc123"
    assert config._clean_key("your_key_here") == ""
    assert config._clean_key(None) == ""


def test_nebius_endpoints_include_legacy_fallback(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_BASE_URL", "")
    assert config.llm_base_urls() == [config.NEBIUS_BASE_URL, config.NEBIUS_LEGACY_BASE_URL]
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://example.test/v1/")
    assert config.llm_base_urls() == ["https://example.test/v1/"]


def test_missing_key_hint_spots_key_left_in_env_example(tmp_path, monkeypatch):
    (tmp_path / ".env.example").write_text("NEBIUS_API_KEY=v1.realkey\n")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert "only reads .env" in config.missing_key_hint()
    (tmp_path / ".env").write_text("NEBIUS_API_KEY=\n")
    assert "empty" in config.missing_key_hint() or "only reads" in config.missing_key_hint()


def test_explain_error_is_actionable():
    assert "401" in llm.explain_error(AuthenticationError("Error code: 401 invalid api key"))
    assert "reach" in llm.explain_error(APIConnectionError("Connection error."))


def test_provider_failure_falls_back_to_extractive(monkeypatch):
    monkeypatch.setattr(config, "llm_available", lambda: True)
    monkeypatch.setattr(llm, "get_chat_model", lambda *a, **k: _Boom(AuthenticationError("401")))
    s = ask("What was Uber's total revenue in 2021?", use_llm=True)
    assert s["generator"] == "extractive"
    assert "401" in s["llm_error"]
    assert s["route"] == "answered" and s["citations"]


def test_check_llm_without_key_reports_hint(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    status = llm.check_llm()
    assert not status.ok and ".env" in status.message

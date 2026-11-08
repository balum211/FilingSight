"""Local model access: readable status, graceful fallback, and the chat wrapper's message handling."""
from filingsight import config, llm
from filingsight.graph import ask, format_sources


class _Boom:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, *_a, **_k):
        raise self.exc


def test_no_api_key_settings_remain():
    for name in ("NEBIUS_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_DTYPE", "LLM_ONNX_FILE"):
        assert not hasattr(config, name)


def test_llm_switched_off_reports_reason(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", False)
    status = llm.check_llm()
    assert not status.ok and "FS_LLM=off" in status.message


def test_explain_error_is_actionable():
    assert "busy" in llm.explain_error(Exception("429 Too Many Requests"))
    assert "reach" in llm.explain_error(ConnectionError("Connection refused"))


def test_fullwidth_citations_are_normalised(monkeypatch):
    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Revenue was $17,455 million【1】."}}]}

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    out = llm.get_chat_model().invoke([("user", "q")])
    assert out.content == "Revenue was $17,455 million[1]."


def test_provider_failure_falls_back_to_extractive(monkeypatch):
    monkeypatch.setattr(config, "llm_available", lambda: True)
    monkeypatch.setattr(llm, "get_chat_model", lambda *a, **k: _Boom(TimeoutError('timed out')))
    s = ask("What was Uber's total revenue in 2021?", use_llm=True)
    assert s["generator"] == "extractive"
    assert s["llm_error"]
    assert s["route"] == "answered" and s["citations"]


def test_chat_message_mapping():
    chat = llm.FreeChatModel._to_chat([("system", "rules"), ("user", "q"), ("human", "h")])
    assert [m["role"] for m in chat] == ["system", "user", "user"]
    assert llm.FreeChatModel._to_chat("hi") == [{"role": "user", "content": "hi"}]


def test_sources_are_trimmed_for_the_prompt():
    src = [{"n": 1, "header": "Uber 10-K", "text": "x" * 2000}]
    assert len(format_sources(src, 100)) < 150
    assert len(format_sources(src)) > 2000


def test_ground_citations_links_supported_sentences_only():
    from filingsight.graph import ground_citations

    src = [{"n": 1, "text": "Lyft revenue was 3,208.3 million in 2021."},
           {"n": 2, "text": "Uber revenue was $17,455 million in 2021, up 57%."}]
    out = ground_citations("Uber's total revenue in 2021 was $17,455 million. Uber plans to fly to Mars.", src)
    assert "$17,455 million [2]." in out
    assert "Mars." in out and "Mars. [" not in out and "Mars [" not in out


def test_free_limit_is_explained_and_not_retried(monkeypatch):
    calls = []

    class R:
        status_code = 402
        text = '{"error":{"code":"FREE_LIMIT","message":"Free limit reached: 1 reply every 45 seconds per IP. Retry after 29s"}}'

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or R())
    try:
        llm.get_chat_model().invoke("q")
    except llm.FreeLimitError as exc:
        msg = llm.explain_error(exc)
    assert len(calls) == 1
    assert "45 seconds" in msg and "29 seconds" in msg


def test_repeated_citation_is_collapsed():
    assert llm.clean_answer("was $17,455 million [1]. [1]") == "was $17,455 million [1]."
    assert llm.clean_answer("x【2】【2】") == "x[2]"

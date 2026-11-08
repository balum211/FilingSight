"""Local model access: readable status, graceful fallback, and the chat wrapper's message handling."""
from filingsight import config, llm
from filingsight.graph import ask, format_sources


class _Boom:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, *_a, **_k):
        raise self.exc


def test_no_api_key_settings_remain():
    for name in ("NEBIUS_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_DTYPE"):
        assert not hasattr(config, name)


def test_llm_switched_off_reports_reason(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", False)
    status = llm.check_llm()
    assert not status.ok and "FS_LLM=off" in status.message


def test_explain_error_is_actionable():
    assert "memory" in llm.explain_error(MemoryError()).lower()
    assert "Hugging Face" in llm.explain_error(OSError("Connection error to huggingface.co"))


def test_provider_failure_falls_back_to_extractive(monkeypatch):
    monkeypatch.setattr(config, "llm_available", lambda: True)
    monkeypatch.setattr(llm, "get_chat_model", lambda *a, **k: _Boom(MemoryError()))
    s = ask("What was Uber's total revenue in 2021?", use_llm=True)
    assert s["generator"] == "extractive"
    assert "memory" in s["llm_error"].lower()
    assert s["route"] == "answered" and s["citations"]


def test_chat_message_mapping():
    chat = llm.LocalChatModel._to_chat([("system", "rules"), ("user", "q"), ("human", "h")])
    assert [m["role"] for m in chat] == ["system", "user", "user"]
    assert llm.LocalChatModel._to_chat("hi") == [{"role": "user", "content": "hi"}]


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


def test_chatml_prompt_shape():
    p = llm.LocalChatModel._chatml([{"role": "user", "content": "hi"}])
    assert p.startswith("<|im_start|>system\n") and p.endswith("<|im_start|>assistant\n")

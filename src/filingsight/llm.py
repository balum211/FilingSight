"""Answer model with no API key: GPT-OSS 20B (open weights, Apache 2.0) through the free
anonymous tier of Pollinations' OpenAI-compatible endpoint.

No account, no key, nothing to configure. The endpoint is a free community service with
rate limits, so every failure (rate limit, timeout, outage) falls back to extractive
mode (the top cited sentences) and the app says why.

`get_chat_model()` returns an object with LangChain's `.invoke(messages) -> .content`
shape, so the LangGraph flow does not change.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from types import SimpleNamespace

from . import config

_FULLWIDTH_CITE = re.compile(r"【\s*(\d+)[^】]*】")


@dataclass
class LLMStatus:
    ok: bool
    provider: str
    model: str
    message: str
    loaded: bool = False


class FreeChatModel:
    """Minimal OpenAI-compatible chat client (requests only, no key)."""

    def __init__(self, url: str, model: str, timeout: float):
        self.url, self.model, self.timeout = url, model, timeout

    @staticmethod
    def _to_chat(messages) -> list[dict]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        role_map = {"human": "user", "ai": "assistant"}
        return [{"role": role_map.get(r, r), "content": c} for r, c in messages]

    def invoke(self, messages, max_tokens: int | None = None):
        import requests

        body = {"model": self.model, "messages": self._to_chat(messages), "seed": 42, "private": True}
        if max_tokens:
            body["max_tokens"] = max_tokens
        last = None
        for attempt in range(2):  # one retry for rate limits and blips
            try:
                r = requests.post(self.url, json=body, timeout=self.timeout)
                if r.status_code == 429 and attempt == 0:
                    time.sleep(3)
                    continue
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                return SimpleNamespace(content=_FULLWIDTH_CITE.sub(r"[\1]", text).strip())
            except Exception as exc:  # noqa: BLE001 - retried once, then raised for the caller
                last = exc
                if attempt == 0:
                    time.sleep(1)
        raise last  # type: ignore[misc]


def get_chat_model(temperature: float = 0.0, model: str | None = None):
    """temperature is ignored: the free endpoint uses the model's default sampling with a fixed seed."""
    if not config.llm_available():
        raise RuntimeError(explain_unavailable())
    return FreeChatModel(config.LLM_URL, model or config.LLM_MODEL, config.LLM_TIMEOUT)


def explain_unavailable() -> str:
    return "Written answers are switched off (FS_LLM=off)."


def explain_error(exc: Exception) -> str:
    """Turn a request error into a reason a person can act on."""
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    if "429" in text or "rate" in low:
        return "The free model is busy (rate limited). Try again in a few seconds; this answer uses cited passages."
    if "timed out" in low or name in {"Timeout", "ReadTimeout", "ConnectTimeout"}:
        return "The free model took too long to reply. This answer uses cited passages."
    if "connection" in low or "resolve" in low or name == "ConnectionError":
        return "Could not reach the free model (network). This answer uses cited passages."
    if any(code in text for code in ("500", "502", "503", "504")):
        return "The free model service is having problems right now. This answer uses cited passages."
    return f"{name}: {text[:200]}"


def check_llm(load: bool = False) -> LLMStatus:
    """Report readiness. With load=True, send a tiny request to confirm the service answers."""
    if not config.llm_available():
        return LLMStatus(False, "free", config.LLM_MODEL_LABEL, explain_unavailable())
    if not load:
        return LLMStatus(True, "free", config.LLM_MODEL_LABEL, "Ready", loaded=True)
    try:
        t = time.perf_counter()
        get_chat_model().invoke("Reply with OK.")
        return LLMStatus(True, "free", config.LLM_MODEL_LABEL, f"Answered in {time.perf_counter() - t:.1f}s", loaded=True)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a readable status
        return LLMStatus(False, "free", config.LLM_MODEL_LABEL, explain_error(exc))


def is_loaded() -> bool:
    return True


def reset() -> None:
    """Nothing is cached; kept for API compatibility."""

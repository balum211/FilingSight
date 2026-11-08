"""Chat model factory. Nebius Token Factory (OpenAI-compatible, Llama 3.3 70B) by
default, any OpenAI-compatible endpoint via FS_LLM_BASE_URL, or OpenAI with OPENAI_API_KEY.

`check_llm()` makes one tiny call and turns provider errors into a plain-English
reason, so the app can say *why* the Llama model is not being used instead of
silently falling back.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import config

# Endpoint that answered the last health check (None = not checked yet).
_working_base_url: str | None = None


@dataclass
class LLMStatus:
    ok: bool
    provider: str
    model: str
    base_url: str
    message: str


@lru_cache(maxsize=8)
def _build(model: str, api_key: str, base_url: str, temperature: float, timeout: float | None = None, retries: int = 1):
    from langchain_openai import ChatOpenAI

    kwargs = dict(model=model, api_key=api_key, temperature=temperature,
                  timeout=timeout or config.LLM_TIMEOUT, max_retries=retries)
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def get_chat_model(temperature: float = 0.0, model: str | None = None):
    if not config.llm_available():
        raise RuntimeError(f"No LLM key set. {config.missing_key_hint()}")
    base_url = _working_base_url if _working_base_url is not None else config.llm_base_urls()[0]
    return _build(model or config.LLM_MODEL, config.llm_api_key(), base_url, temperature)


def explain_error(exc: Exception) -> str:
    """Map an OpenAI SDK error to a reason a person can act on."""
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    if name == "AuthenticationError" or "401" in text or "invalid api key" in low or "unauthorized" in low:
        return ("The provider rejected the API key (401). Create a new key in the Nebius console "
                "and put it in .env as NEBIUS_API_KEY.")
    if name == "PermissionDeniedError" or "403" in text:
        return "The key is valid but has no access to this model (403). Check billing or model access in the Nebius console."
    if name == "NotFoundError" or "404" in text or "does not exist" in low or "model_not_found" in low:
        return (f"Model '{config.LLM_MODEL}' was not found on this endpoint (404). Set FS_LLM_MODEL in .env "
                "to a model listed in your Nebius console, for example meta-llama/Llama-3.3-70B-Instruct.")
    if name == "RateLimitError" or "429" in text:
        return "Rate limited or out of credits (429). Wait a minute or top up credits in the Nebius console."
    if name in {"APIConnectionError", "APITimeoutError", "ConnectError"} or "connection" in low or "timed out" in low:
        return "Could not reach the LLM endpoint (network or proxy). Check your internet connection or FS_LLM_BASE_URL."
    return f"{name}: {text[:200]}"


def _is_endpoint_problem(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in {"APIConnectionError", "APITimeoutError", "NotFoundError"} or "connection" in str(exc).lower()


def check_llm() -> LLMStatus:
    """Send a one-token request. Tries the fallback Nebius endpoint if the first is unreachable."""
    global _working_base_url
    provider, model = config.llm_provider(), config.LLM_MODEL
    if not config.llm_available():
        return LLMStatus(False, provider, model, "", config.missing_key_hint())
    last_exc: Exception | None = None
    urls = config.llm_base_urls()
    for url in urls:
        try:
            chat = _build(model, config.llm_api_key(), url, 0.0, 15.0, 0)
            chat.bind(max_tokens=1).invoke([("user", "ping")])
            _working_base_url = url
            return LLMStatus(True, provider, model, url or "api.openai.com", "Connected")
        except Exception as exc:  # noqa: BLE001 - every provider error becomes a readable status
            last_exc = exc
            if not _is_endpoint_problem(exc):
                break  # bad key or quota: another endpoint will not help
    return LLMStatus(False, provider, model, urls[0], explain_error(last_exc))


def reset() -> None:
    """Forget cached clients and the working endpoint (after the key or model changes)."""
    global _working_base_url
    _working_base_url = None
    _build.cache_clear()

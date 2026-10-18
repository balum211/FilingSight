"""Chat model factory. Nebius AI Studio (OpenAI-compatible) by default, any
OpenAI-compatible endpoint via FS_LLM_BASE_URL, or OpenAI with OPENAI_API_KEY."""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0, model: str | None = None):
    if not config.llm_available():
        raise RuntimeError("No LLM key set. Add NEBIUS_API_KEY (or OPENAI_API_KEY) to .env.")
    from langchain_openai import ChatOpenAI

    kwargs = dict(
        model=model or config.LLM_MODEL,
        api_key=config.llm_api_key(),
        temperature=temperature,
        timeout=config.LLM_TIMEOUT,
        max_retries=2,
    )
    if config.LLM_BASE_URL:
        kwargs["base_url"] = config.LLM_BASE_URL
    return ChatOpenAI(**kwargs)

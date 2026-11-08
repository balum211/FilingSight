"""Local chat model: a small open model that runs on CPU with no API key.

Default is Qwen/Qwen2.5-0.5B-Instruct (Apache 2.0) through Hugging Face
transformers. `get_chat_model()` returns an object with the same `.invoke()`
shape as a LangChain chat model, so the LangGraph flow does not change.

`check_llm()` reports whether the model is ready and, if not, why, so the app can
show a clear status instead of silently falling back.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

from . import config

_lock = threading.Lock()


@dataclass
class LLMStatus:
    ok: bool
    provider: str
    model: str
    message: str
    loaded: bool = False


class LocalChatModel:
    """Minimal LangChain-style wrapper: invoke(messages) -> object with .content."""

    def __init__(self, model_id: str, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        self.model.eval()

    @staticmethod
    def _to_chat(messages) -> list[dict]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        role_map = {"human": "user", "ai": "assistant"}
        return [{"role": role_map.get(r, r), "content": c} for r, c in messages]

    def invoke(self, messages, max_new_tokens: int | None = None):
        chat = self._to_chat(messages)
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with _lock, self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or config.LLM_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return SimpleNamespace(content=text.strip())


@lru_cache(maxsize=1)
def _load(model_id: str, dtype: str) -> LocalChatModel:
    return LocalChatModel(model_id, dtype)


def is_loaded() -> bool:
    return _load.cache_info().currsize > 0


def get_chat_model(temperature: float = 0.0, model: str | None = None):
    """Return the shared local model (loaded once per process). temperature is ignored: decoding is greedy."""
    if not config.llm_available():
        raise RuntimeError(explain_unavailable())
    return _load(model or config.LLM_MODEL, config.LLM_DTYPE)


def explain_unavailable() -> str:
    if not config.LLM_ENABLED:
        return "The local model is switched off (FS_LLM=off)."
    if not config.llm_installed():
        return "torch and transformers are not installed. Run pip install -r requirements.txt."
    return "The local model is not available."


def explain_error(exc: Exception) -> str:
    """Turn a load or generation error into a reason a person can act on."""
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    if isinstance(exc, MemoryError) or "out of memory" in low or "cannot allocate" in low:
        return "Not enough memory to run the local model on this server. Answers use cited passages instead."
    if "connection" in low or "resolve" in low or "offline" in low or "huggingface.co" in low:
        return "Could not download the model from Hugging Face (network). Answers use cited passages instead."
    if "not installed" in low or name == "ModuleNotFoundError":
        return explain_unavailable()
    return f"{name}: {text[:200]}"


def check_llm(load: bool = False) -> LLMStatus:
    """Report readiness. With load=True, load the model and run a one-token generation."""
    model = config.LLM_MODEL
    if not config.llm_available():
        return LLMStatus(False, "local", model, explain_unavailable())
    if not load:
        msg = "Loaded and ready" if is_loaded() else "Loads on the first question (about 1 GB download, once)"
        return LLMStatus(True, "local", model, msg, loaded=is_loaded())
    try:
        t = time.perf_counter()
        get_chat_model().invoke("Say OK.", max_new_tokens=1)
        return LLMStatus(True, "local", model, f"Loaded and ready ({time.perf_counter() - t:.0f}s)", loaded=True)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a readable status
        return LLMStatus(False, "local", model, explain_error(exc))


def reset() -> None:
    _load.cache_clear()

"""Local chat model: a small open model that runs on CPU with no API key.

Default is Qwen2.5-0.5B-Instruct (Apache 2.0), int8 ONNX export, run with ONNX
Runtime (already used by Chroma for embeddings), so there is no torch install.
`get_chat_model()` returns an object with the same `.invoke()`
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
    """Greedy decoding with ONNX Runtime on CPU (int8 weights, no torch).

    Uses the ONNX export of the model (onnx-community/*), the fast `tokenizers`
    library and a ChatML prompt. Exposes LangChain's invoke(messages) -> .content.
    """

    def __init__(self, repo_id: str, onnx_file: str):
        import json

        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.np = np
        model_path = hf_hub_download(repo_id, onnx_file)
        try:  # large exports keep weights in a side file next to the graph
            hf_hub_download(repo_id, onnx_file + "_data")
        except Exception:  # noqa: BLE001 - most quantized exports are a single file
            pass
        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo_id, "tokenizer.json"))
        cfg = json.loads(open(hf_hub_download(repo_id, "config.json"), encoding="utf-8").read())
        self.kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        self.head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.past_names = [n for n in self.inputs if n.startswith("past_key_values")]
        self.kv_dtype = np.float16 if any(self.inputs[n].type == "tensor(float16)" for n in self.past_names) else np.float32
        self.eos = {i for i in (self.tokenizer.token_to_id("<|im_end|>"), self.tokenizer.token_to_id("<|endoftext|>")) if i is not None}

    @staticmethod
    def _to_chat(messages) -> list[dict]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        role_map = {"human": "user", "ai": "assistant"}
        return [{"role": role_map.get(r, r), "content": c} for r, c in messages]

    @staticmethod
    def _chatml(chat: list[dict]) -> str:
        if chat[0]["role"] != "system":
            chat = [{"role": "system", "content": "You are a helpful assistant."}] + chat
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in chat)
        return body + "<|im_start|>assistant\n"

    def invoke(self, messages, max_new_tokens: int | None = None):
        np = self.np
        ids = self.tokenizer.encode(self._chatml(self._to_chat(messages)), add_special_tokens=False).ids
        past = {n: np.zeros((1, self.kv_heads, 0, self.head_dim), dtype=self.kv_dtype) for n in self.past_names}
        step_ids, total, generated = ids, len(ids), []
        with _lock:
            for _ in range(max_new_tokens or config.LLM_MAX_NEW_TOKENS):
                feed = {"input_ids": np.array([step_ids], dtype=np.int64),
                        "attention_mask": np.ones((1, total), dtype=np.int64)}
                if "position_ids" in self.inputs:
                    feed["position_ids"] = np.arange(total - len(step_ids), total, dtype=np.int64)[None, :]
                feed.update(past)
                out = dict(zip(self.outputs, self.session.run(None, feed)))
                nxt = int(out["logits"][0, -1].argmax())
                if nxt in self.eos:
                    break
                generated.append(nxt)
                past = {n: out[n.replace("past_key_values", "present")] for n in self.past_names}
                step_ids, total = [nxt], total + 1
        return SimpleNamespace(content=self.tokenizer.decode(generated, skip_special_tokens=True).strip())


@lru_cache(maxsize=1)
def _load(model_id: str, onnx_file: str) -> LocalChatModel:
    return LocalChatModel(model_id, onnx_file)


def is_loaded() -> bool:
    return _load.cache_info().currsize > 0


def get_chat_model(temperature: float = 0.0, model: str | None = None):
    """Return the shared local model (loaded once per process). temperature is ignored: decoding is greedy."""
    if not config.llm_available():
        raise RuntimeError(explain_unavailable())
    return _load(model or config.LLM_MODEL, config.LLM_ONNX_FILE)


def explain_unavailable() -> str:
    if not config.LLM_ENABLED:
        return "The local model is switched off (FS_LLM=off)."
    if not config.llm_installed():
        return "onnxruntime, tokenizers or huggingface_hub is not installed. Run pip install -r requirements.txt."
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
        msg = "Loaded and ready" if is_loaded() else "Loads on the first question (about 500 MB download, once)"
        return LLMStatus(True, "local", model, msg, loaded=is_loaded())
    try:
        t = time.perf_counter()
        get_chat_model().invoke("Say OK.", max_new_tokens=1)
        return LLMStatus(True, "local", model, f"Loaded and ready ({time.perf_counter() - t:.0f}s)", loaded=True)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a readable status
        return LLMStatus(False, "local", model, explain_error(exc))


def reset() -> None:
    _load.cache_clear()

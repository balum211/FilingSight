"""Central settings. Everything can be overridden with environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:  # optional: load the project's .env when python-dotenv is installed.
    # Load it by absolute path so it works no matter which folder Streamlit is started from.
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = Path(os.getenv("FS_INDEX_DIR", ROOT / ".index"))


@dataclass(frozen=True)
class Filing:
    """One source document in the corpus."""

    company: str
    ticker: str
    form: str
    fiscal_year: int
    file: str
    # The 10-K body ends where the Exhibit Index starts. Pages after it are
    # exhibits (credit agreements, indentures) that only add retrieval noise.
    body_last_page: int
    # Financial years that the filing reports figures for (current + 2 prior).
    covered_years: tuple[int, ...] = field(default=())


FILINGS: tuple[Filing, ...] = (
    Filing("Uber", "UBER", "10-K", 2021, "uber_2021.pdf", body_last_page=149, covered_years=(2019, 2020, 2021)),
    Filing("Lyft", "LYFT", "10-K", 2021, "lyft_2021.pdf", body_last_page=124, covered_years=(2019, 2020, 2021)),
)

COMPANIES = tuple(f.company for f in FILINGS)
COVERED_YEARS = sorted({y for f in FILINGS for y in f.covered_years})


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# Chunking. all-MiniLM-L6-v2 truncates at 256 word-pieces (~1,000 characters of
# 10-K prose), so chunks are sized to fit inside the embedding window.
FIXED_CHUNK_SIZE = _env_int("FS_FIXED_CHUNK_SIZE", 900)
FIXED_CHUNK_OVERLAP = _env_int("FS_FIXED_CHUNK_OVERLAP", 150)
SEMANTIC_BREAKPOINT_PERCENTILE = _env_float("FS_SEMANTIC_PERCENTILE", 85)
SEMANTIC_MAX_CHARS = _env_int("FS_SEMANTIC_MAX_CHARS", 1100)
SEMANTIC_MIN_CHARS = _env_int("FS_SEMANTIC_MIN_CHARS", 200)

# Retrieval
CANDIDATES_K = _env_int("FS_CANDIDATES_K", 20)  # per retriever, before fusion
TOP_K = _env_int("FS_TOP_K", 5)  # chunks handed to the LLM
RRF_K = _env_int("FS_RRF_K", 60)
# Weighted RRF. 10-K questions lean on exact figures and names, so BM25 gets
# more say than dense similarity (tuned in eval/run_retrieval_eval.py).
DENSE_WEIGHT = _env_float("FS_DENSE_WEIGHT", 1.0)
BM25_WEIGHT = _env_float("FS_BM25_WEIGHT", 1.0)

# Reranker: "local" (no download, runs anywhere), "cross-encoder"
# (sentence-transformers ms-marco model), or "llm" (uses the chat model).
RERANKER = os.getenv("FS_RERANKER", "local")
CROSS_ENCODER_MODEL = os.getenv("FS_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# "I don't know" gate: if the best reranked chunk scores below this, refuse.
MIN_CONFIDENCE = _env_float("FS_MIN_CONFIDENCE", 0.60)  # local reranker scale
MIN_CONFIDENCE_CE = _env_float("FS_MIN_CONFIDENCE_CE", 0.05)  # cross-encoder sigmoid scale

# Default strategy used by the app (the evaluation picks the winner).
DEFAULT_CHUNKING = os.getenv("FS_CHUNKING", "semantic")

# LLM (OpenAI-compatible). Nebius (Token Factory, formerly AI Studio) is the
# default provider for the course and serves the Llama 3.3 70B model.
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Older Nebius AI Studio endpoint, tried automatically if the new one is unreachable.
NEBIUS_LEGACY_BASE_URL = "https://api.studio.nebius.com/v1/"
DEFAULT_NEBIUS_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

_PLACEHOLDERS = {"", "your_key_here", "your-nebius-key", "changeme", "xxx", "sk-..."}


def _clean_key(value: str | None) -> str:
    """Strip whitespace and stray quotes that sneak in when keys are pasted into .env."""
    key = (value or "").strip().strip('"').strip("'").strip()
    return "" if key.lower() in _PLACEHOLDERS else key


NEBIUS_API_KEY = _clean_key(os.getenv("NEBIUS_API_KEY"))
OPENAI_API_KEY = _clean_key(os.getenv("OPENAI_API_KEY"))
LLM_BASE_URL = os.getenv("FS_LLM_BASE_URL", "").strip()
LLM_MODEL = os.getenv("FS_LLM_MODEL", "").strip() or (DEFAULT_NEBIUS_MODEL if NEBIUS_API_KEY or not OPENAI_API_KEY else "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("FS_JUDGE_MODEL", LLM_MODEL)
LLM_TIMEOUT = _env_float("FS_LLM_TIMEOUT", 60)


def set_runtime_key(api_key: str, provider: str = "nebius") -> None:
    """Use a key entered at runtime (for example pasted into the app sidebar)."""
    global NEBIUS_API_KEY, OPENAI_API_KEY, LLM_MODEL
    key = _clean_key(api_key)
    if provider == "openai":
        OPENAI_API_KEY, NEBIUS_API_KEY = key, ""
        if LLM_MODEL == DEFAULT_NEBIUS_MODEL:
            LLM_MODEL = "gpt-4.1-mini"
    else:
        NEBIUS_API_KEY = key
        if not os.getenv("FS_LLM_MODEL"):
            LLM_MODEL = DEFAULT_NEBIUS_MODEL


def llm_api_key() -> str:
    return NEBIUS_API_KEY or OPENAI_API_KEY


def llm_provider() -> str:
    if NEBIUS_API_KEY:
        return "Nebius"
    if OPENAI_API_KEY:
        return "OpenAI"
    return "none"


def llm_base_urls() -> list[str]:
    """Endpoints to try, in order. An explicit FS_LLM_BASE_URL always wins."""
    if LLM_BASE_URL:
        return [LLM_BASE_URL]
    if NEBIUS_API_KEY:
        return [NEBIUS_BASE_URL, NEBIUS_LEGACY_BASE_URL]
    return [""]  # OpenAI default endpoint


def llm_available() -> bool:
    return bool(llm_api_key())


def missing_key_hint() -> str:
    """Explain why no key was found, including the common .env.example mistake."""
    env_file = ROOT / ".env"
    example = ROOT / ".env.example"
    example_has_key = False
    if example.exists():
        for line in example.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("NEBIUS_API_KEY=") and _clean_key(line.split("=", 1)[1]):
                example_has_key = True
    if example_has_key and not (env_file.exists() and "NEBIUS_API_KEY" in env_file.read_text(errors="ignore")):
        return ("A key is in .env.example, but the app only reads .env. Copy .env.example to .env "
                "(and remove the key from .env.example so it is not pushed to GitHub).")
    if not env_file.exists():
        return "No .env file found. Copy .env.example to .env and set NEBIUS_API_KEY."
    return "NEBIUS_API_KEY is empty in .env. Paste your Nebius key after the = sign."


# Latency budget from the one-liner (seconds, end to end).
LATENCY_BUDGET_S = _env_float("FS_LATENCY_BUDGET_S", 8)


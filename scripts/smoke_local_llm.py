"""Smoke test for the free answer model (no API key): answers three questions end to end.

    python scripts/smoke_local_llm.py
"""
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingsight import config, llm  # noqa: E402
from filingsight.graph import ask  # noqa: E402

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


from filingsight.index import load_index  # noqa: E402

load_index("semantic")
print(f"peak RSS after index {rss():.0f} MB")
status = llm.check_llm()  # no ping: the free tier allows one reply every 45 seconds per IP
print(f"peak RSS after model {rss():.0f} MB")
print(f"model={config.LLM_MODEL_LABEL} url={config.LLM_URL} ok={status.ok} msg={status.message}")
assert status.ok, status.message
for i, q in enumerate(["What was Uber's total revenue in 2021?",
                       "What did Lyft sell to Woven Planet?",
                       "Compare Uber's and Lyft's revenue growth in 2021."]):
    if i:
        time.sleep(47)  # stay inside the free tier's one reply per 45 seconds
    t = time.perf_counter()
    s = ask(q, use_llm=True)
    print(f"\nQ: {q}\n  route={s['route']} generator={s.get('generator')} citations={s.get('citations')} "
          f"llm_error={s.get('llm_error')} {time.perf_counter() - t:.1f}s\n  A: {s['answer'][:600]!r}")
    assert s.get("generator") == "llm", s.get("llm_error")
    assert s["route"] == "answered" and s.get("citations"), "the free model gave no cited answer"
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"\npeak RSS {rss_mb:.0f} MB")

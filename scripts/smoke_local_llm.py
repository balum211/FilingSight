"""Smoke test for the local answer model: loads it, answers three questions, reports memory and latency.

    python scripts/smoke_local_llm.py
"""
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingsight import config, llm  # noqa: E402
from filingsight.graph import ask  # noqa: E402

status = llm.check_llm(load=True)
print(f"model={config.LLM_MODEL} dtype={config.LLM_DTYPE} ok={status.ok} msg={status.message}")
assert status.ok, status.message
for q in ["What was Uber's total revenue in 2021?",
          "What did Lyft sell to Woven Planet?",
          "Compare Uber's and Lyft's revenue growth in 2021."]:
    t = time.perf_counter()
    s = ask(q, use_llm=True)
    print(f"\nQ: {q}\n  route={s['route']} generator={s.get('generator')} citations={s.get('citations')} "
          f"llm_error={s.get('llm_error')} {time.perf_counter() - t:.1f}s\n  A: {s['answer'][:600]}")
    assert s.get("generator") == "llm", s.get("llm_error")
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"\npeak RSS {rss_mb:.0f} MB")

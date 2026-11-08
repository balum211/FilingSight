"""Times the free endpoint with different prompt sizes and request options (diagnostic)."""
import time, requests
URL = "https://text.pollinations.ai/openai"
src = "\n\n".join(f"[{i}] Uber 10-K FY2021 page {50+i}\n" + ("Revenue increased $6.3 billion, or 57%, primarily attributable to Gross Bookings. " * 14)[:1200] for i in range(1, 6))
user = f"Sources:\n{src}\n\nQuestion: What was Uber's total revenue in 2021?"
sys_ = "Answer only from the sources. Cite every sentence with [n]. If the answer is not in the sources reply NOT_FOUND."
for label, extra, u in [("short", {}, "Sources:\n[1] Uber revenue was $17,455 million in 2021.\n\nQuestion: Uber revenue 2021?"),
                        ("full plain", {}, user),
                        ("full seed+private", {"seed": 42, "private": True}, user),
                        ("full max_tokens", {"max_tokens": 400}, user),
                        ("full reasoning low", {"reasoning_effort": "low"}, user),
                        ("openai-fast", {"model": "openai-fast"}, user)]:
    body = {"model": "openai", "messages": [{"role": "system", "content": sys_}, {"role": "user", "content": u}], **extra}
    t = time.perf_counter()
    try:
        r = requests.post(URL, json=body, timeout=90)
        txt = r.text[:200].replace("\n", " ")
        print(f"{label}: {r.status_code} {time.perf_counter()-t:.1f}s {txt}", flush=True)
    except Exception as e:
        print(f"{label}: {type(e).__name__} {time.perf_counter()-t:.1f}s", flush=True)

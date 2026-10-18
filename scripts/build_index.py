"""Rebuild data/processed/* (chunks + embeddings) for both chunking strategies.

    python scripts/build_index.py            # both strategies
    python scripts/build_index.py fixed      # just one
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingsight.chunking import STRATEGIES  # noqa: E402
from filingsight.index import build_and_save  # noqa: E402

if __name__ == "__main__":
    for strategy in sys.argv[1:] or STRATEGIES:
        t = time.time()
        n = build_and_save(strategy)
        print(f"{strategy:9s} {n:5d} chunks  {time.time() - t:6.1f}s")

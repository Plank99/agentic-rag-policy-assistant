"""Standalone helper to (re)build the FAISS index without launching the UI.

    python scripts/build_index.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.vectorstore import load_or_build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild even if the corpus is unchanged.")
    args = parser.parse_args()

    store = load_or_build_index(force_rebuild=args.force)
    print(f"Index ready with {store.index.ntotal} vectors.")


if __name__ == "__main__":
    main()

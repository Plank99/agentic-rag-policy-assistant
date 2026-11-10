"""Central, environment-driven configuration for the agent."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

POLICY_DOCS_DIR = Path(os.getenv("POLICY_DOCS_DIR", PROJECT_ROOT / "app" / "data" / "policies"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / "app" / "data" / "index"))
TICKETS_DB_PATH = Path(os.getenv("TICKETS_DB_PATH", PROJECT_ROOT / "app" / "data" / "tickets.db"))

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MAX_REWRITES = int(os.getenv("RAG_MAX_REWRITES", "2"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

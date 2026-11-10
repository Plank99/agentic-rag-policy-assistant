"""Load and chunk the policy markdown corpus.

Scalability note: documents are split with source + section metadata preserved on every
chunk, so the corpus can grow (more files, larger files) without changing the ingestion
code — only the chunk count grows. Swapping the in-memory FAISS store for a managed vector
DB (pgvector, Pinecone, etc.) for a larger corpus is a change confined to vectorstore.py;
this module never needs to know about the storage backend.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, POLICY_DOCS_DIR


@dataclass
class Chunk:
    content: str
    source: str
    chunk_id: int


def load_markdown_files(docs_dir: Path = POLICY_DOCS_DIR) -> list[tuple[str, str]]:
    """Return a list of (filename, raw_text) for every .md file in docs_dir."""
    docs_dir = Path(docs_dir)
    files = sorted(docs_dir.glob("*.md"))
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def chunk_documents(
    documents: list[tuple[str, str]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks: list[Chunk] = []
    for filename, text in documents:
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append(Chunk(content=piece, source=filename, chunk_id=i))
    return chunks


def corpus_fingerprint(docs_dir: Path = POLICY_DOCS_DIR) -> str:
    """Hash of file names + contents, used to decide whether the index must be rebuilt."""
    docs_dir = Path(docs_dir)
    hasher = hashlib.sha256()
    for f in sorted(docs_dir.glob("*.md")):
        hasher.update(f.name.encode("utf-8"))
        hasher.update(f.read_bytes())
    return hasher.hexdigest()

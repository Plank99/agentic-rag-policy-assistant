"""FAISS-backed vector store wrapper.

This is the one module that knows about the storage backend. Everything else in the RAG
subgraph talks to it through `load_or_build_index` / `similarity_search`, so a future move
to a different vector database only touches this file.
"""
from __future__ import annotations

from pathlib import Path

from app.config import INDEX_DIR, POLICY_DOCS_DIR, RAG_TOP_K
from app.rag.ingest import chunk_documents, corpus_fingerprint, load_markdown_files

_FINGERPRINT_FILE = "corpus.fingerprint"

_embeddings_cache = {}


def get_embeddings():
    """Lazily construct (and cache) the local sentence-transformers embedding model."""
    from app.config import EMBEDDING_MODEL

    if "embeddings" not in _embeddings_cache:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings_cache["embeddings"] = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_cache["embeddings"]


def _build_index(docs_dir: Path, embeddings):
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    documents = load_markdown_files(docs_dir)
    chunks = chunk_documents(documents)
    lc_docs = [
        Document(page_content=c.content, metadata={"source": c.source, "chunk_id": c.chunk_id})
        for c in chunks
    ]
    return FAISS.from_documents(lc_docs, embeddings)


def load_or_build_index(
    docs_dir: Path = POLICY_DOCS_DIR, index_dir: Path = INDEX_DIR, force_rebuild: bool = False
):
    """Load a cached FAISS index from disk, rebuilding it if the corpus changed."""
    from langchain_community.vectorstores import FAISS

    docs_dir = Path(docs_dir)
    index_dir = Path(index_dir)
    embeddings = get_embeddings()
    fingerprint_path = index_dir / _FINGERPRINT_FILE
    current_fingerprint = corpus_fingerprint(docs_dir)

    index_is_fresh = (
        not force_rebuild
        and (index_dir / "index.faiss").exists()
        and fingerprint_path.exists()
        and fingerprint_path.read_text(encoding="utf-8").strip() == current_fingerprint
    )

    if index_is_fresh:
        return FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )

    store = _build_index(docs_dir, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    fingerprint_path.write_text(current_fingerprint, encoding="utf-8")
    return store


def similarity_search(store, query: str, k: int = RAG_TOP_K) -> list[dict]:
    results = store.similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
        }
        for doc, score in results
    ]

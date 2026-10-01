"""
rag.py
------
Knowledge-grounded retrieval pipeline for evaluating interview answers.

Pipeline:
1. Load .txt files from knowledge_base/
2. Split each into overlapping chunks
3. Embed chunks with Sentence-Transformers (via embeddings.py)
4. Build a FAISS index (Inner Product on L2-normalized vectors == cosine similarity)
5. At evaluation time, retrieve the top-k chunks most relevant to a
   question + answer pair, to ground the LLM's evaluation in real reference
   material instead of relying purely on the model's parametric knowledge.

If FAISS, the embedding model, or the knowledge_base folder is unavailable,
retrieve_context() returns an empty list and the rest of the app continues
without knowledge grounding (clearly labeled as such in the UI).
"""

import os
import re

from src import embeddings

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")

_index = None
_chunks = None
_build_error = None

CHUNK_SIZE = 600  # characters
CHUNK_OVERLAP = 100


def _chunk_text(text: str, source: str):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _load_knowledge_base():
    all_chunks = []
    if not os.path.isdir(KB_DIR):
        return all_chunks
    for filename in sorted(os.listdir(KB_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(KB_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            topic = filename.replace(".txt", "").replace("_", " ").title()
            all_chunks.extend(_chunk_text(content, topic))
        except Exception:
            continue
    return all_chunks


def build_index(force_rebuild: bool = False):
    """Builds (or rebuilds) the FAISS index from the knowledge base. Cached
    at module level so it only runs once per process (Streamlit also wraps
    the caller with st.cache_resource -- see app.py)."""
    global _index, _chunks, _build_error

    if _index is not None and not force_rebuild:
        return True

    try:
        import faiss
        import numpy as np
    except Exception as e:
        _build_error = f"FAISS is not available: {e}"
        return False

    chunks = _load_knowledge_base()
    if not chunks:
        _build_error = "Knowledge base folder is empty or missing."
        return False

    texts = [c["text"] for c in chunks]
    vectors = embeddings.embed_texts(texts)
    if vectors is None:
        _build_error = "Embedding model unavailable; cannot build RAG index."
        return False

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors, dtype="float32"))

    _index = index
    _chunks = chunks
    _build_error = None
    return True


def is_ready() -> bool:
    return _index is not None


def get_build_error():
    return _build_error


def retrieve_context(query: str, top_k: int = 3):
    """Returns a list of {text, source, score} dicts for the top_k most
    relevant knowledge-base chunks to the query. Returns [] if the index is
    not available -- callers must handle this gracefully."""
    if not query.strip():
        return []
    if _index is None:
        ready = build_index()
        if not ready:
            return []

    import numpy as np

    query_vec = embeddings.embed_texts([query])
    if query_vec is None:
        return []

    query_vec = np.ascontiguousarray(query_vec, dtype="float32")
    scores, indices = _index.search(query_vec, min(top_k, len(_chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_chunks):
            continue
        results.append(
            {
                "text": _chunks[idx]["text"],
                "source": _chunks[idx]["source"],
                "score": round(float(score), 3),
            }
        )
    return results


def format_context_for_prompt(context_chunks: list) -> str:
    if not context_chunks:
        return "No grounded reference material was retrieved for this topic."
    lines = []
    for c in context_chunks:
        lines.append(f"[Source: {c['source']}] {c['text']}")
    return "\n\n".join(lines)

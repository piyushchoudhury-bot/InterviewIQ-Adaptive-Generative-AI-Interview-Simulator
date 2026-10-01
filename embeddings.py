"""
embeddings.py
-------------
Wraps the Sentence-Transformers model used for:
1. Semantic similarity between resume and JD text (Resume-JD match score).
2. Generating embeddings for the RAG knowledge base (see rag.py).

We use the small, fast 'all-MiniLM-L6-v2' model (384-dim embeddings) which
is a good balance of speed and quality for a local B.Tech project and does
not require a GPU.
"""

import numpy as np

_model = None
_model_error = None

MODEL_NAME = "all-MiniLM-L6-v2"


def get_model():
    """Lazily load the Sentence-Transformers model (cached at module level)."""
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
        return _model
    except Exception as e:
        _model_error = str(e)
        return None


def is_available() -> bool:
    return get_model() is not None


def embed_texts(texts):
    """Returns an (n, dim) numpy array of L2-normalized embeddings, or None
    if the embedding model could not be loaded."""
    model = get_model()
    if model is None:
        return None
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return embeddings / norms


def cosine_similarity_score(text_a: str, text_b: str):
    """Returns a similarity score in [0, 100], or None if embeddings are
    unavailable (caller should fall back to a keyword-based estimate)."""
    if not text_a.strip() or not text_b.strip():
        return None
    embeddings = embed_texts([text_a, text_b])
    if embeddings is None:
        return None
    sim = float(np.dot(embeddings[0], embeddings[1]))
    sim = max(0.0, min(1.0, (sim + 1) / 2)) if sim < 0 else min(1.0, sim)
    return round(sim * 100, 1)


def keyword_overlap_score(text_a: str, text_b: str):
    """Fallback similarity using simple word-overlap (Jaccard-like) when the
    embedding model cannot be loaded (e.g., no internet to download it)."""
    def tokenize(t):
        return set(w.strip(".,()[]{}:;\"'").lower() for w in t.split() if len(w.strip(".,()[]{}:;\"'")) > 2)

    set_a, set_b = tokenize(text_a), tokenize(text_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return round((intersection / union) * 100, 1) if union else 0.0

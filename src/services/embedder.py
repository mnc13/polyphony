"""
Shared sentence embedder (all-MiniLM-L6-v2, 384-d, normalized).

Single lazy-loaded instance reused everywhere embeddings are needed: vector DB
indexing/search, P-MER novelty scoring, and premise dedup. Normalized vectors so
cosine similarity is a plain dot product (matches PROCLAIM's setup).
"""
from __future__ import annotations

import numpy as np

from config.settings import get_settings

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(get_settings().hp.embedding_model)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, dim) float32 array of L2-normalized embeddings."""
    if not texts:
        return np.zeros((0, get_settings().hp.embedding_dim), dtype="float32")
    vecs = _get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def max_cosine(vec: np.ndarray, pool: np.ndarray) -> float:
    """Max cosine similarity of `vec` against rows of `pool` (both normalized)."""
    if pool.shape[0] == 0:
        return 0.0
    return float(np.max(pool @ vec))

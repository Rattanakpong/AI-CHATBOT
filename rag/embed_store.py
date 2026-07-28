"""
Vector store: turn chunks into vectors and support similarity search over them.

Implements (per Final Project Brief, Sections 2 & 5):
  Req 3 - Embeddings: real embedding model via sentence-transformers
          (all-MiniLM-L6-v2, local, free, no API key). TF-IDF is kept only as
          a comparison baseline for the evaluation write-up.
  Req 4 - Vector search: cosine similarity over the embedding matrix
          (in-memory; fine below a few thousand chunks per the brief).

Design decisions:
  * Pluggable backends behind one `VectorStore` interface, so app.py never
    changes when the embedding model does (LLO2: justifiable modularity).
  * Embeddings are cached to disk (.cache/) keyed by a hash of the corpus +
    backend name, so Streamlit restarts and live demos are instant.
"""

import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np

from .ingest import Chunk

CACHE_DIR = ".cache"


# ---------------------------------------------------------------------------
# Embedding backends — each exposes .name and .encode(list[str]) -> np.ndarray
# ---------------------------------------------------------------------------

class TfidfBackend:
    """Baseline from the Week 14 lab. Kept for evaluation comparison only."""
    name = "tfidf"
    default_threshold = 0.15  # TF-IDF cosine scores run lower than dense embeddings

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            matrix = self.vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            matrix = self.vectorizer.transform(texts)
        return np.asarray(matrix.todense(), dtype=np.float32)


class SentenceTransformerBackend:
    """Real embeddings — the required backend for the final submission."""
    name = "sentence-transformers"
    default_threshold = 0.30

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = f"st-{model_name}"

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float32,
        )


class OpenAIBackend:
    """Optional API backend (Section 5 'Stronger Option'). Needs OPENAI_API_KEY."""
    name = "openai"
    default_threshold = 0.30

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        self.name = f"openai-{model_name}"

    def encode(self, texts: List[str]) -> np.ndarray:
        # OpenAI accepts batches; keep batches modest to stay under limits.
        vectors = []
        for i in range(0, len(texts), 100):
            resp = self.client.embeddings.create(model=self.model_name,
                                                 input=texts[i:i + 100])
            vectors.extend(d.embedding for d in resp.data)
        return np.asarray(vectors, dtype=np.float32)


def make_backend(backend: str = "auto"):
    """
    Backend selection:
      "auto"  -> sentence-transformers if installed, else TF-IDF (with warning)
      "st"    -> sentence-transformers (raises if missing)
      "tfidf" -> TF-IDF baseline
      "openai"-> OpenAI embeddings API
    """
    if backend == "tfidf":
        return TfidfBackend()
    if backend == "openai":
        return OpenAIBackend()
    if backend in ("st", "sentence-transformers"):
        return SentenceTransformerBackend()
    # auto
    try:
        return SentenceTransformerBackend()
    except ImportError:
        print("[embed_store] sentence-transformers not installed; "
              "falling back to TF-IDF baseline. Run: pip install sentence-transformers")
        return TfidfBackend()


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def _cosine_sim_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against every row of `matrix`."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q


class VectorStore:
    def __init__(self, backend: str = "auto", cache: bool = True):
        self.backend = make_backend(backend)
        self.cache = cache
        self.matrix: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []

    # -- caching ----------------------------------------------------------
    def _cache_path(self, texts: List[str]) -> str:
        h = hashlib.sha256(("||".join(texts) + self.backend.name).encode()).hexdigest()[:16]
        os.makedirs(CACHE_DIR, exist_ok=True)
        return os.path.join(CACHE_DIR, f"emb_{self.backend.name}_{h}.npy")

    # -- interface --------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunk text (using the disk cache when possible)."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        path = self._cache_path(texts) if self.cache else None
        if path and os.path.exists(path) and not isinstance(self.backend, TfidfBackend):
            self.matrix = np.load(path)
            return

        self.matrix = self.backend.encode(texts)
        # TF-IDF vocabularies aren't portable across runs, so don't cache them.
        if path and not isinstance(self.backend, TfidfBackend):
            np.save(path, self.matrix)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, cosine_similarity) pairs for a query string."""
        if self.matrix is None:
            raise RuntimeError("VectorStore.build() must be called before query().")
        query_vec = self.backend.encode([query_text])[0]
        scores = _cosine_sim_matrix(query_vec, self.matrix)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]

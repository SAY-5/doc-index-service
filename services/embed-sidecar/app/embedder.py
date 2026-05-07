"""Embedder implementations.

Two backends are provided:

* ``HashEmbedder`` is a deterministic bag-of-token-hashes embedder. It is
  not semantic — its only job is to produce identically-shaped vectors so
  the API and storage layers can be exercised end-to-end in CI without
  pulling a 90 MB model into the test container. Cosine similarity over
  hashed bags still has just enough signal that "the same query repeated"
  ranks above an unrelated query, which is what the unit tests assert.

* ``SentenceTransformersEmbedder`` wraps the eponymous library and serves
  the production path. It is loaded lazily so the unit tests don't have
  to import torch.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .settings import Settings


class Embedder(Protocol):
    """Embedder protocol."""

    @property
    def name(self) -> str:
        """Identifier returned in /embed responses and /healthz."""

    @property
    def dim(self) -> int:
        """Output dimensionality."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalised vector per input text."""


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests and CI."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._name = f"hash-bag-{dim}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            out.append(self._embed_one(t).tolist())
        return out

    def _embed_one(self, text: str) -> NDArray[np.float32]:
        vec = np.zeros(self._dim, dtype=np.float32)
        # Lowercase, split on whitespace, drop empties.
        tokens = [tok for tok in text.lower().split() if tok]
        if not tokens:
            # Constant non-zero vector for empty inputs so downstream
            # cosine math doesn't blow up. The sign pattern is arbitrary.
            vec[0] = 1.0
            return vec
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return cast(NDArray[np.float32], vec / norm)


class SentenceTransformersEmbedder:
    """Production embedder backed by sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        # Imported lazily so HashEmbedder users don't pull torch.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._name = model_name
        out_dim = int(self._model.get_sentence_embedding_dimension())
        if out_dim != 384:
            raise ValueError(
                f"model {model_name!r} produces dim {out_dim}, but the schema is fixed at 384"
            )
        self._dim = out_dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return cast(list[list[float]], arr.astype("float32").tolist())


def build_embedder(settings: Settings) -> Embedder:
    """Factory: pick the embedder that matches settings."""
    if settings.embedder_kind == "hash":
        return HashEmbedder(dim=settings.dim)
    return SentenceTransformersEmbedder(model_name=settings.model_name)

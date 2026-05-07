"""Cross-encoder reranker implementations.

Two backends mirror the embedder split:

* ``HeuristicReranker`` — a Python clone of the Go
  ``rerank.HeuristicReranker``. Useful for sidecar-only tests and as a
  safe fallback when no model is configured.
* ``CrossEncoderReranker`` — wraps ``sentence_transformers.CrossEncoder``
  with the ``cross-encoder/ms-marco-MiniLM-L-6-v2`` checkpoint by
  default. Loaded lazily so importers that don't need it (notably the
  unit-test environment) don't pay the torch import.

Both implementations satisfy the ``Reranker`` protocol so the FastAPI
factory can return whichever the settings select.
"""

from __future__ import annotations

import math
import re
from typing import Protocol, cast

from .settings import Settings

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


class Reranker(Protocol):
    """Reranker protocol."""

    @property
    def name(self) -> str:
        """Identifier returned in /rerank responses."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one float per passage. Higher is better."""


class HeuristicReranker:
    """Lexical-overlap reranker with a length penalty.

    Output scores are roughly comparable to the Go heuristic but the two
    are *not* required to be identical — the Go side computes a different
    ranking on a different host, and the only invariant the API enforces
    is that the scores are deterministic and finite for finite input.
    """

    name = "heuristic-overlap"

    def __init__(self, target_snippet_tokens: int = 60) -> None:
        self._target = max(1, target_snippet_tokens)

    def score(self, query: str, passages: list[str]) -> list[float]:
        q_tokens = _tokens(query)
        target = self._target
        out: list[float] = []
        for p in passages:
            s_tokens = _tokens(p)
            overlap = len(q_tokens & s_tokens)
            if q_tokens and s_tokens:
                denom = math.sqrt(len(q_tokens) * len(s_tokens))
            else:
                denom = 1.0
            base = overlap / denom
            excess = max(0, len(s_tokens) - target)
            penalty = 1.0 / (1.0 + excess / target)
            out.append(base * penalty)
        return out


class CrossEncoderReranker:
    """Production reranker backed by sentence-transformers' CrossEncoder."""

    def __init__(self, model_name: str) -> None:
        # Imported lazily so unit-test environments (which use the
        # heuristic) never have to install torch.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self._name = model_name

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        raw = self._model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        return cast(list[float], raw.astype("float64").tolist())


def _tokens(s: str) -> set[str]:
    if not s:
        return set()
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(s)}


def build_reranker(settings: Settings) -> Reranker:
    """Pick the reranker backend that matches settings.

    The same EMBEDDER_KIND env var selects both stages because they ship
    in lockstep — a CI image with ``hash`` embeddings would not
    otherwise have the model weights available for cross-encoder rerank.
    """
    if settings.embedder_kind == "hash":
        return HeuristicReranker()
    # The cross-encoder model name is currently a constant rather than a
    # second env var; if a future deployment needs to override it the
    # plumbing is small and obvious.
    return CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

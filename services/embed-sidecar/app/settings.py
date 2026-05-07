"""Runtime configuration for the embedding sidecar.

Everything is read from environment variables so the same image runs in
local docker-compose, the bench harness, and CI without rebuilds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Frozen view of the env-derived configuration."""

    embedder_kind: str
    """Either ``hf`` for sentence-transformers or ``hash`` for the deterministic stub."""

    model_name: str
    """Sentence-transformers model id, used only when embedder_kind == 'hf'."""

    dim: int
    """Output dimensionality. Pinned at 384 to match the Postgres column."""

    @classmethod
    def from_env(cls) -> Settings:
        kind = os.environ.get("EMBEDDER_KIND", "hash").strip().lower()
        if kind not in {"hf", "hash"}:
            raise ValueError(f"unknown EMBEDDER_KIND: {kind!r}")
        model = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return cls(embedder_kind=kind, model_name=model, dim=384)

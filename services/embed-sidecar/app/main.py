"""FastAPI app for the embedding sidecar."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embedder import Embedder, build_embedder
from .settings import Settings


class EmbedRequest(BaseModel):
    """Request body for ``POST /embed``."""

    texts: list[str] = Field(..., description="One or more strings to embed")


class EmbedResponse(BaseModel):
    """Response body for ``POST /embed``."""

    embeddings: list[list[float]]
    model: str
    dim: int


class HealthResponse(BaseModel):
    """Response body for ``GET /healthz``."""

    model: str
    ready: bool


def create_app(embedder: Embedder | None = None) -> FastAPI:
    """Build the FastAPI app, accepting an injected embedder for tests."""
    app = FastAPI(title="doc-index-service embed sidecar")
    settings = Settings.from_env()

    state: dict[str, Embedder | None] = {"embedder": embedder}

    def get_embedder() -> Embedder:
        e = state["embedder"]
        if e is None:
            e = build_embedder(settings)
            state["embedder"] = e
        return e

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest) -> EmbedResponse:
        if not req.texts:
            return EmbedResponse(embeddings=[], model=get_embedder().name, dim=get_embedder().dim)
        for i, t in enumerate(req.texts):
            if not isinstance(t, str):
                raise HTTPException(status_code=400, detail=f"texts[{i}] is not a string")
        e = get_embedder()
        vectors = e.embed(req.texts)
        return EmbedResponse(embeddings=vectors, model=e.name, dim=e.dim)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        e = get_embedder()
        return HealthResponse(model=e.name, ready=True)

    return app


app = create_app()

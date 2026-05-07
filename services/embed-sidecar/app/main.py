"""FastAPI app for the embedding sidecar."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embedder import Embedder, build_embedder
from .reranker import Reranker, build_reranker
from .settings import Settings


class EmbedRequest(BaseModel):
    """Request body for ``POST /embed``."""

    texts: list[str] = Field(..., description="One or more strings to embed")


class EmbedResponse(BaseModel):
    """Response body for ``POST /embed``."""

    embeddings: list[list[float]]
    model: str
    dim: int


class RerankRequest(BaseModel):
    """Request body for ``POST /rerank``."""

    query: str = Field(..., description="Query string")
    passages: list[str] = Field(..., description="Candidate passages to score")


class RerankResponse(BaseModel):
    """Response body for ``POST /rerank``. ``scores`` is parallel to the
    request's ``passages`` list and is *not* sorted — the Go side does
    its own ordering so it can keep the prior-rank tie-break stable."""

    scores: list[float]
    model: str


class HealthResponse(BaseModel):
    """Response body for ``GET /healthz``."""

    model: str
    ready: bool


def create_app(embedder: Embedder | None = None, reranker: Reranker | None = None) -> FastAPI:
    """Build the FastAPI app, accepting injected components for tests."""
    app = FastAPI(title="doc-index-service embed sidecar")
    settings = Settings.from_env()

    state: dict[str, Embedder | Reranker | None] = {
        "embedder": embedder,
        "reranker": reranker,
    }

    def get_embedder() -> Embedder:
        e = state["embedder"]
        if e is None:
            e = build_embedder(settings)
            state["embedder"] = e
        return e  # type: ignore[return-value]

    def get_reranker() -> Reranker:
        r = state["reranker"]
        if r is None:
            r = build_reranker(settings)
            state["reranker"] = r
        return r  # type: ignore[return-value]

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

    @app.post("/rerank", response_model=RerankResponse)
    def rerank(req: RerankRequest) -> RerankResponse:
        if not req.passages:
            return RerankResponse(scores=[], model=get_reranker().name)
        for i, p in enumerate(req.passages):
            if not isinstance(p, str):
                raise HTTPException(status_code=400, detail=f"passages[{i}] is not a string")
        r = get_reranker()
        scores = r.score(req.query, req.passages)
        return RerankResponse(scores=scores, model=r.name)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        e = get_embedder()
        return HealthResponse(model=e.name, ready=True)

    return app


app = create_app()

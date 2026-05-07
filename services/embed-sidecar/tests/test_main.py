"""HTTP-level tests for the embedding sidecar."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.embedder import HashEmbedder
from app.main import create_app
from app.reranker import HeuristicReranker


@pytest.fixture
def client() -> AsyncClient:
    app = create_app(embedder=HashEmbedder(dim=384), reranker=HeuristicReranker())
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_embed_endpoint_round_trip(client: AsyncClient) -> None:
    async with client as c:
        resp = await c.post("/embed", json={"texts": ["hello", "world"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"].startswith("hash-bag-")
    assert body["dim"] == 384
    assert len(body["embeddings"]) == 2
    assert all(len(v) == 384 for v in body["embeddings"])


async def test_embed_endpoint_empty_list(client: AsyncClient) -> None:
    async with client as c:
        resp = await c.post("/embed", json={"texts": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["embeddings"] == []
    assert body["dim"] == 384


async def test_embed_endpoint_validates_payload(client: AsyncClient) -> None:
    async with client as c:
        resp = await c.post("/embed", json={"wrong_key": ["x"]})
    assert resp.status_code == 422


async def test_healthz(client: AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["model"]


async def test_rerank_endpoint_round_trip(client: AsyncClient) -> None:
    async with client as c:
        resp = await c.post(
            "/rerank",
            json={"query": "raft leader", "passages": ["raft leader election", "noise"]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"]
    assert len(body["scores"]) == 2
    # The relevant passage must outscore the noise one — that's the only
    # ranking guarantee the heuristic makes.
    assert body["scores"][0] > body["scores"][1]


async def test_rerank_endpoint_empty_passages(client: AsyncClient) -> None:
    async with client as c:
        resp = await c.post("/rerank", json={"query": "x", "passages": []})
    assert resp.status_code == 200
    assert resp.json()["scores"] == []


async def test_rerank_endpoint_validates_payload(client: AsyncClient) -> None:
    async with client as c:
        resp = await c.post("/rerank", json={"query": "x"})
    assert resp.status_code == 422

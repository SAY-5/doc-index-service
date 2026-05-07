"""HTTP-level tests for the embedding sidecar."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.embedder import HashEmbedder
from app.main import create_app


@pytest.fixture
def client() -> AsyncClient:
    app = create_app(embedder=HashEmbedder(dim=384))
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

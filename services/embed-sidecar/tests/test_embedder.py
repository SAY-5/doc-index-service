"""Unit tests for the HashEmbedder."""

from __future__ import annotations

import math

import pytest

from app.embedder import HashEmbedder


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def test_hash_embedder_dim_and_shape() -> None:
    e = HashEmbedder(dim=384)
    out = e.embed(["hello world", "another sentence"])
    assert len(out) == 2
    assert all(len(v) == 384 for v in out)


def test_hash_embedder_is_deterministic() -> None:
    e1 = HashEmbedder()
    e2 = HashEmbedder()
    a = e1.embed(["the quick brown fox"])
    b = e2.embed(["the quick brown fox"])
    assert a == b


def test_identical_inputs_have_cosine_one() -> None:
    e = HashEmbedder()
    a, b = e.embed(["the optimizer rewrites the predicate", "the optimizer rewrites the predicate"])
    assert cosine(a, b) == pytest.approx(1.0, abs=1e-6)


def test_similar_inputs_outscore_unrelated() -> None:
    e = HashEmbedder()
    a, b, c = e.embed(
        [
            "the optimizer rewrites the predicate",
            "the optimizer reorders the predicate",  # shares 4/5 tokens
            "the kernel flushes the working set",  # disjoint
        ]
    )
    assert cosine(a, b) > cosine(a, c)


def test_empty_string_does_not_break() -> None:
    e = HashEmbedder()
    [v] = e.embed([""])
    assert len(v) == 384
    # Any non-zero vector is fine; we just don't want all zeros.
    assert any(x != 0.0 for x in v)


def test_repeat_call_is_idempotent() -> None:
    e = HashEmbedder()
    v1 = e.embed(["repeated text"])
    v2 = e.embed(["repeated text"])
    assert v1 == v2

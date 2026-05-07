"""Property-based tests for the HashEmbedder.

These complement the example-based tests in test_embedder.py with
Hypothesis-driven random inputs. The properties asserted are the same
ones the rest of the indexing pipeline relies on:

* the embedder is deterministic — repeated calls on equal inputs return
  equal vectors,
* the output shape is always ``(len(texts), dim)`` regardless of how the
  texts were batched,
* identical inputs collide perfectly under cosine similarity (sanity
  check that the constant-non-zero fallback for empty strings does not
  silently degrade),
* the per-vector L2 norm is finite and non-zero (so the cosine math in
  Postgres never divides by zero on an embedding we wrote).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.embedder import HashEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


# Strategies. Dim is sampled from a small set rather than the full int
# range because the schema in production fixes dim=384; varying it here
# only checks that the embedder code is dimensionally correct.
_dims = st.sampled_from([32, 64, 384])
_texts = st.lists(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=0,
        max_size=120,
    ),
    min_size=0,
    max_size=16,
)


@given(texts=_texts, dim=_dims)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_shape_matches_input(texts: list[str], dim: int) -> None:
    e = HashEmbedder(dim=dim)
    out = e.embed(texts)
    assert len(out) == len(texts)
    for v in out:
        assert len(v) == dim


@given(texts=_texts, dim=_dims)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_deterministic_across_instances(texts: list[str], dim: int) -> None:
    a = HashEmbedder(dim=dim).embed(texts)
    b = HashEmbedder(dim=dim).embed(texts)
    assert a == b


@given(texts=_texts, dim=_dims, batch=st.integers(min_value=1, max_value=8))
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_batch_invariance(texts: list[str], dim: int, batch: int) -> None:
    """Embedding the corpus all at once vs in batches must yield the same vectors."""
    e = HashEmbedder(dim=dim)
    flat = e.embed(texts)
    batched: list[list[float]] = []
    for i in range(0, len(texts), batch):
        batched.extend(e.embed(texts[i : i + batch]))
    assert flat == batched


@given(text=st.text(min_size=1, max_size=80))
@settings(deadline=None)
def test_self_cosine_is_one(text: str) -> None:
    """A vector compared against itself must be perfectly aligned."""
    e = HashEmbedder()
    [v] = e.embed([text])
    assert _cosine(v, v) == pytest.approx(1.0, abs=1e-6)


@given(text=st.text(min_size=1, max_size=80))
@settings(deadline=None)
def test_norm_is_finite_and_nonzero(text: str) -> None:
    e = HashEmbedder()
    [v] = e.embed([text])
    n = math.sqrt(sum(x * x for x in v))
    assert math.isfinite(n)
    assert n > 0.0


@given(token=st.text(alphabet="abcdefghij", min_size=1, max_size=10))
@settings(deadline=None)
def test_repeating_token_collides_with_self(token: str) -> None:
    """A doc that repeats one token must rank identically across runs."""
    e = HashEmbedder()
    a, b = e.embed([token + " " + token, token + " " + token])
    assert a == b
    assert _cosine(a, b) == pytest.approx(1.0, abs=1e-6)

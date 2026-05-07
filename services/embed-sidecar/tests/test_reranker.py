"""Unit tests for the HeuristicReranker."""

from __future__ import annotations

from app.reranker import HeuristicReranker


def test_returns_one_score_per_passage() -> None:
    r = HeuristicReranker()
    out = r.score("kafka topics", ["alpha", "kafka topics", "beta gamma"])
    assert len(out) == 3


def test_higher_overlap_scores_higher() -> None:
    r = HeuristicReranker()
    s_match, s_nomatch = r.score(
        "redis cluster failover", ["redis cluster failover diag", "lorem ipsum dolor"]
    )
    assert s_match > s_nomatch


def test_length_penalty_demotes_long_irrelevant_padding() -> None:
    # Use *distinct* tokens so the snippet's token set actually grows; a
    # repeated single word would dedupe to one entry and dodge the
    # penalty. That's an intentional property of the heuristic — overlap
    # is set-based — but here we want the penalty to bite.
    distinct_filler = " ".join(f"filler{i}" for i in range(200))
    r = HeuristicReranker()
    short, long_ = r.score(
        "raft leader election",
        [
            "raft leader election timing",
            "raft leader election " + distinct_filler,
        ],
    )
    assert short > long_


def test_empty_passages_yields_empty_scores() -> None:
    r = HeuristicReranker()
    assert r.score("anything", []) == []


def test_empty_query_does_not_crash() -> None:
    r = HeuristicReranker()
    out = r.score("", ["anything", "else"])
    assert out == [0.0, 0.0]


def test_case_insensitive() -> None:
    r = HeuristicReranker()
    upper = r.score("GraphQL Schema", ["GRAPHQL SCHEMA validation"])[0]
    lower = r.score("graphql schema", ["graphql schema validation"])[0]
    assert upper == lower

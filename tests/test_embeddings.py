"""Embeddings tests: unit behavior plus retrieval-quality checks."""

from __future__ import annotations

import math

import pytest

from selflearn.embeddings import (
    Bm25Index,
    NgramTfidfIndex,
    OllamaEmbedder,
    OpenAICompatEmbedder,
    char_ngrams,
    gram_cosine,
    tokenize,
)
from tests.mock_servers import MockServer, OllamaMockHandler, OpenAIMockHandler

DOCS = {
    "coffee": "user drinks oat milk latte every morning",
    "city": "user moved from Berlin to Lisbon in March",
    "dog": "user has a golden retriever named Biscuit",
    "work": "user is a backend engineer working with Postgres",
    "hobby": "user plays chess on weekends and likes puzzles",
}


def _filled_bm25() -> Bm25Index:
    idx = Bm25Index()
    for doc_id, text in DOCS.items():
        idx.add(doc_id, text)
    return idx


def _filled_ngram() -> NgramTfidfIndex:
    idx = NgramTfidfIndex()
    for doc_id, text in DOCS.items():
        idx.add(doc_id, text)
    return idx


def test_tokenize_and_ngrams() -> None:
    assert tokenize("Hello, World 123!") == ["hello", "world", "123"]
    grams = char_ngrams("ab", n_min=2, n_max=2)
    assert all(len(g) == 2 for g in grams)
    assert grams


def test_bm25_exact_match_wins() -> None:
    idx = _filled_bm25()
    top = idx.rank("oat milk latte")
    assert top[0][0] == "coffee"


def test_bm25_empty_query_scores_zero() -> None:
    idx = _filled_bm25()
    assert idx.rank("") == []
    assert idx.score("latte", "missing") == 0.0


def test_bm25_replace_and_remove() -> None:
    idx = _filled_bm25()
    idx.add("coffee", "user hates coffee now")
    assert idx.rank("hates coffee")[0][0] == "coffee"
    idx.remove("coffee")
    assert "coffee" not in [d for d, _ in idx.rank("coffee")]
    assert len(idx) == 4
    idx.remove("nope")  # never raises


def test_ngram_near_duplicate_scores_high() -> None:
    idx = _filled_ngram()
    # Morphological variant still matches the latte doc best.
    top = idx.rank("drinking oat-milk lattes each morning")
    assert top[0][0] == "coffee"
    assert top[0][1] > 0.3


def test_ngram_cosine_self_is_one() -> None:
    idx = NgramTfidfIndex()
    idx.add("a", "exact same words here")
    assert idx.cosine("exact same words here", "a") == pytest.approx(1.0)
    assert idx.cosine("totally unrelated zebra quasar", "a") < 0.5
    assert idx.cosine("anything", "missing") == 0.0


def test_ngram_vector_normalized() -> None:
    idx = _filled_ngram()
    vec = idx.vector("some query text")
    assert len(vec) == 512
    assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)


def test_gram_cosine_pairwise() -> None:
    """Pairwise TF cosine: no IDF, so near-identical strings score high."""
    assert gram_cosine("lives in lisbon", "lives in lisbon") == pytest.approx(1.0)
    assert gram_cosine("lives in lisbon!", "lives in  lisbon") > 0.7
    assert gram_cosine("likes chess", "lives in lisbon") < 0.4
    assert gram_cosine("", "x") == 0.0


def test_retrieval_quality_both_indexes() -> None:
    """Each labeled query must rank its doc first in both indexes."""
    queries = {
        "oat milk latte morning": "coffee",
        "moved Berlin Lisbon": "city",
        "golden retriever Biscuit": "dog",
        "backend Postgres engineer": "work",
        "chess weekends puzzles": "hobby",
    }
    bm25, ngram = _filled_bm25(), _filled_ngram()
    for query, expected in queries.items():
        assert bm25.rank(query)[0][0] == expected, query
        assert ngram.rank(query)[0][0] == expected, query


def test_hybrid_covers_weak_spots() -> None:
    """A query failing one signal still wins on max-of-both (used by retrieval)."""
    bm25, ngram = _filled_bm25(), _filled_ngram()
    queries = ["latte", "Lisbon move", "Biscuit dog", "Postgres", "chess"]
    for query in queries:
        best_bm25 = bm25.rank(query, top_k=1)
        best_vec = ngram.rank(query, top_k=1)
        assert best_bm25 or best_vec, query


@pytest.fixture()
def openai_server() -> MockServer:
    srv = MockServer(OpenAIMockHandler)
    yield srv
    srv.stop()


@pytest.fixture()
def ollama_server() -> MockServer:
    srv = MockServer(OllamaMockHandler)
    yield srv
    srv.stop()


def test_openai_compat_embedder(openai_server: MockServer) -> None:
    emb = OpenAICompatEmbedder(base_url=openai_server.base_url + "/v1", api_key="k", model="m")
    vecs = emb.embed(["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == 3
    assert emb.dim == 3
    sent = openai_server.requests[-1]["body"]
    assert sent["input"] == ["a", "b"] and sent["model"] == "m"


def test_ollama_embedder(ollama_server: MockServer) -> None:
    emb = OllamaEmbedder(base_url=ollama_server.base_url, model="m")
    vecs = emb.embed(["a"])
    assert len(vecs) == 1 and len(vecs[0]) == 2
    assert emb.dim == 2
    sent = ollama_server.requests[-1]["body"]
    assert sent["input"] == ["a"]


def test_embedder_error_mapping(openai_server: MockServer) -> None:
    from selflearn.llm.types import BadRequestError

    emb = OpenAICompatEmbedder(base_url=openai_server.base_url + "/v1", api_key="k", model="")
    with pytest.raises(BadRequestError):
        emb.embed(["a"])

"""From-scratch hybrid text similarity: BM25 plus hashed char n-gram TF-IDF.

No API, no dependencies. BM25 catches exact rare-term matches; the char
n-gram vector catches near-duplicates and morphological variants
("running" ~ "run"). Retrieval blends both (see memory.retrieval).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def char_ngrams(text: str, *, n_min: int = 3, n_max: int = 5) -> list[str]:
    """Character n-grams over lowercased text with boundary markers."""
    padded = f" {text.lower()} "
    out: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(len(padded) - n + 1):
            out.append(padded[i : i + n])
    return out


def _hash_bucket(term: str, buckets: int) -> int:
    digest = hashlib.md5(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % buckets


def gram_cosine(a: str, b: str, *, buckets: int = 512, n_min: int = 3, n_max: int = 5) -> float:
    """Pairwise TF cosine over hashed char n-grams. No IDF: for dedup, not ranking."""
    ca: Counter[int] = Counter()
    for gram in char_ngrams(a, n_min=n_min, n_max=n_max):
        ca[_hash_bucket(gram, buckets)] += 1
    cb: Counter[int] = Counter()
    for gram in char_ngrams(b, n_min=n_min, n_max=n_max):
        cb[_hash_bucket(gram, buckets)] += 1
    if not ca or not cb:
        return 0.0
    dot = sum(tf * cb.get(bucket, 0) for bucket, tf in ca.items())
    denom = math.sqrt(sum(tf * tf for tf in ca.values()) * sum(tf * tf for tf in cb.values()))
    return dot / denom if denom > 0 else 0.0


class Bm25Index:
    """Okapi BM25 over word tokens. Incremental: add docs, then score."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._df: Counter[str] = Counter()

    def add(self, doc_id: str, text: str) -> None:
        """Add or replace a document."""
        if doc_id in self._docs:
            for term in set(self._docs[doc_id]):
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]
        tokens = tokenize(text)
        self._docs[doc_id] = tokens
        for term in set(tokens):
            self._df[term] += 1

    def remove(self, doc_id: str) -> None:
        """Remove a document if present."""
        tokens = self._docs.pop(doc_id, None)
        if tokens is None:
            return
        for term in set(tokens):
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]

    def __len__(self) -> int:
        return len(self._docs)

    def score(self, query: str, doc_id: str) -> float:
        """BM25 score of one doc for the query. 0.0 when unknown/empty."""
        tokens = self._docs.get(doc_id)
        if not tokens:
            return 0.0
        q_terms = tokenize(query)
        if not q_terms:
            return 0.0
        n = len(self._docs)
        avg_len = sum(len(t) for t in self._docs.values()) / n
        tf = Counter(tokens)
        total = 0.0
        for term in q_terms:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            freq = tf.get(term, 0)
            denom = freq + self.k1 * (1 - self.b + self.b * len(tokens) / max(avg_len, 1e-9))
            total += idf * (freq * (self.k1 + 1)) / max(denom, 1e-9)
        return total

    def rank(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """All docs with score > 0, best first, capped at top_k."""
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self._docs]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class NgramTfidfIndex:
    """Hashed character n-gram TF-IDF vectors with cosine similarity.

    Fixed bucket count => fixed dim, incremental, stdlib only. IDF is
    recomputed from current doc frequencies at query time (cheap at our
    scale: one pass over buckets touched by the query).
    """

    def __init__(self, *, buckets: int = 512, n_min: int = 3, n_max: int = 5) -> None:
        self.buckets = buckets
        self.n_min = n_min
        self.n_max = n_max
        self._docs: dict[str, Counter[int]] = {}
        self._df: Counter[int] = Counter()

    def add(self, doc_id: str, text: str) -> None:
        """Add or replace a document."""
        if doc_id in self._docs:
            for bucket in set(self._docs[doc_id]):
                self._df[bucket] -= 1
                if self._df[bucket] <= 0:
                    del self._df[bucket]
        vec: Counter[int] = Counter()
        for gram in char_ngrams(text, n_min=self.n_min, n_max=self.n_max):
            vec[_hash_bucket(gram, self.buckets)] += 1
        self._docs[doc_id] = vec
        for bucket in set(vec):
            self._df[bucket] += 1

    def remove(self, doc_id: str) -> None:
        """Remove a document if present."""
        vec = self._docs.pop(doc_id, None)
        if vec is None:
            return
        for bucket in set(vec):
            self._df[bucket] -= 1
            if self._df[bucket] <= 0:
                del self._df[bucket]

    def __len__(self) -> int:
        return len(self._docs)

    def _idf(self, bucket: int) -> float:
        n = len(self._docs)
        df = self._df.get(bucket, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    def vector(self, text: str) -> list[float]:
        """TF-IDF hashed vector for arbitrary text, L2-normalized."""
        counts: Counter[int] = Counter()
        for gram in char_ngrams(text, n_min=self.n_min, n_max=self.n_max):
            counts[_hash_bucket(gram, self.buckets)] += 1
        vec = [0.0] * self.buckets
        for bucket, tf in counts.items():
            vec[bucket] = (1 + math.log(tf)) * self._idf(bucket)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def cosine(self, query: str, doc_id: str) -> float:
        """Cosine similarity between query text and a stored doc."""
        doc_vec = self._docs.get(doc_id)
        if not doc_vec:
            return 0.0
        q_counts: Counter[int] = Counter()
        for gram in char_ngrams(query, n_min=self.n_min, n_max=self.n_max):
            q_counts[_hash_bucket(gram, self.buckets)] += 1
        dot = 0.0
        q_norm_sq = 0.0
        d_norm_sq = 0.0
        for bucket, tf in q_counts.items():
            qw = (1 + math.log(tf)) * self._idf(bucket)
            q_norm_sq += qw * qw
            if bucket in doc_vec:
                dwt = (1 + math.log(doc_vec[bucket])) * self._idf(bucket)
                dot += qw * dwt
        for bucket, tf in doc_vec.items():
            dw = (1 + math.log(tf)) * self._idf(bucket)
            d_norm_sq += dw * dw
        denom = math.sqrt(q_norm_sq * d_norm_sq)
        return dot / denom if denom > 0 else 0.0

    def rank(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """All docs with similarity > 0, best first, capped at top_k."""
        scored = [(doc_id, self.cosine(query, doc_id)) for doc_id in self._docs]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


__all__ = ["Bm25Index", "NgramTfidfIndex", "char_ngrams", "tokenize"]

"""Embeddings: local hybrid default plus optional API providers."""

from selflearn.embeddings.api import OllamaEmbedder, OpenAICompatEmbedder
from selflearn.embeddings.base import Embedder
from selflearn.embeddings.hybrid import (
    Bm25Index,
    NgramTfidfIndex,
    char_ngrams,
    gram_cosine,
    tokenize,
)

__all__ = [
    "Bm25Index",
    "Embedder",
    "NgramTfidfIndex",
    "OllamaEmbedder",
    "OpenAICompatEmbedder",
    "char_ngrams",
    "gram_cosine",
    "tokenize",
]

"""Pluggable embedding interface. Local hybrid default needs no API."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """One interface for turning texts into vectors."""

    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text, in order."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]


__all__ = ["Embedder"]

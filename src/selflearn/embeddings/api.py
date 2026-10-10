"""Optional API embedding providers: OpenAI-compatible and Ollama.

Anthropic has no embeddings API, so there is no Anthropic embedder.
Local default lives in hybrid.py and needs no key.
"""

from __future__ import annotations

from selflearn.embeddings.base import Embedder
from selflearn.llm.base import decode_json, http_post_json, map_http_error, parse_retry_after


class OpenAICompatEmbedder(Embedder):
    """Any OpenAI-compatible embeddings endpoint. urllib only."""

    def __init__(
        self, *, base_url: str, api_key: str = "", model: str = "", timeout_s: float = 60.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """POST {base}/embeddings. Raises typed errors on failure."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "selflearn/0.1.0 (urllib)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        status, resp_headers, body = http_post_json(
            f"{self.base_url}/embeddings",
            headers,
            {"input": texts, "model": self.model},
            timeout_s=self.timeout_s,
            provider="openai_compatible",
        )
        if status != 200:
            raise map_http_error(
                status,
                body,
                provider="openai_compatible",
                retry_after_s=parse_retry_after(resp_headers.get("retry-after")),
            )
        data = decode_json(body, provider="openai_compatible")
        items = data.get("data", [])
        vecs = [list(map(float, item.get("embedding", []))) for item in items]
        if len(vecs) != len(texts):
            from selflearn.llm.types import ParseError

            raise ParseError(
                "openai_compatible: embedding count mismatch", provider="openai_compatible"
            )
        if vecs:
            self.dim = len(vecs[0])
        return vecs


class OllamaEmbedder(Embedder):
    """Local Ollama embeddings via POST /api/embed."""

    def __init__(
        self, *, base_url: str = "http://localhost:11434", model: str = "", timeout_s: float = 120.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """POST {base}/api/embed. Raises typed errors on failure."""
        status, resp_headers, body = http_post_json(
            f"{self.base_url}/api/embed",
            {"Content-Type": "application/json", "User-Agent": "selflearn/0.1.0 (urllib)"},
            {"model": self.model, "input": texts},
            timeout_s=self.timeout_s,
            provider="ollama",
        )
        if status != 200:
            raise map_http_error(
                status,
                body,
                provider="ollama",
                retry_after_s=parse_retry_after(resp_headers.get("retry-after")),
            )
        data = decode_json(body, provider="ollama")
        vecs = [[float(v) for v in row] for row in data.get("embeddings", [])]
        if len(vecs) != len(texts):
            from selflearn.llm.types import ParseError

            raise ParseError("ollama: embedding count mismatch", provider="ollama")
        if vecs:
            self.dim = len(vecs[0])
        return vecs


__all__ = ["OllamaEmbedder", "OpenAICompatEmbedder"]

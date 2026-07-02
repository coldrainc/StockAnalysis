from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import requests
from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_SERVICE_URL = "http://127.0.0.1:18210"
EmbeddingProvider = Literal["openai", "local", "service"]


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    provider: EmbeddingProvider = "openai"
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int = 64
    device: str | None = None
    local_files_only: bool = False
    service_url: str | None = None
    timeout: float = 60.0


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        kwargs = {}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.api_key:
            kwargs["api_key"] = config.api_key
        self.client = OpenAI(**kwargs)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            response = self.client.embeddings.create(
                model=self.config.model,
                input=batch,
            )
            vectors.extend(item.embedding for item in response.data)
        return [normalize_vector(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class LocalEmbeddingClient:
    """SentenceTransformers-backed local embedding client."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "本地 embedding 需要安装 sentence-transformers，请先运行 pip install -e '.[dev]'。"
            ) from exc

        kwargs = {}
        if self.config.device:
            kwargs["device"] = self.config.device
        if self.config.local_files_only:
            kwargs["local_files_only"] = True
        return SentenceTransformer(self.config.model, **kwargs)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            encoded = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors.extend([list(map(float, vector)) for vector in encoded])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class EmbeddingServiceClient:
    """HTTP client for a local or remote embedding service."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.base_url = (config.service_url or DEFAULT_EMBEDDING_SERVICE_URL).rstrip("/")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            response = requests.post(
                f"{self.base_url}/embed",
                json={"texts": batch},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            vectors.extend(payload["vectors"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()


def create_embedding_client(config: EmbeddingConfig):
    if config.provider == "service":
        return EmbeddingServiceClient(config)
    if config.provider == "local":
        return LocalEmbeddingClient(config)
    return EmbeddingClient(config)


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

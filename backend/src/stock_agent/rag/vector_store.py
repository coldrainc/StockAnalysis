from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from stock_agent.rag.embedding import cosine_similarity


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str = "json"
    path: Path | None = None
    collection_name: str = "stock_agent"
    url: str | None = None
    api_key: str | None = None
    vector_size: int | None = None
    recreate_collection: bool = False
    batch_size: int = 256


@dataclass(frozen=True)
class VectorSearchHit:
    id: str
    score: float


class VectorStore(Protocol):
    @property
    def metadata(self) -> dict[str, Any]:
        ...

    def is_available(self) -> bool:
        ...

    def upsert(
        self,
        records: list[tuple[str, list[float], dict[str, Any]]],
        metadata: dict[str, Any],
    ) -> None:
        ...

    def search(self, query_vector: list[float], top_k: int = 80) -> list[VectorSearchHit]:
        ...


class JsonVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._payload: dict[str, Any] | None = None
        self._vectors: dict[str, list[float]] | None = None

    @property
    def payload(self) -> dict[str, Any]:
        if self._payload is None:
            if not self.path.exists():
                self._payload = {}
            else:
                self._payload = json.loads(self.path.read_text(encoding="utf-8"))
        return self._payload

    @property
    def metadata(self) -> dict[str, Any]:
        payload = self.payload
        return {
            "vector_store": "json",
            "embedding_provider": payload.get("embedding_provider", "openai"),
            "embedding_model": payload.get("embedding_model"),
            "chunk_count": payload.get("chunk_count"),
            "path": str(self.path),
        }

    @property
    def vectors(self) -> dict[str, list[float]]:
        if self._vectors is None:
            self._vectors = {
                item["id"]: list(item["vector"])
                for item in self.payload.get("vectors", [])
            }
        return self._vectors

    def is_available(self) -> bool:
        return self.path.exists() and bool(self.vectors)

    def upsert(
        self,
        records: list[tuple[str, list[float], dict[str, Any]]],
        metadata: dict[str, Any],
    ) -> None:
        payload = {
            **metadata,
            "vector_store": "json",
            "chunk_count": len(records),
            "vectors": [
                {"id": record_id, "vector": vector}
                for record_id, vector, _payload in records
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self._payload = payload
        self._vectors = {record_id: vector for record_id, vector, _payload in records}

    def search(self, query_vector: list[float], top_k: int = 80) -> list[VectorSearchHit]:
        scored = [
            VectorSearchHit(record_id, cosine_similarity(query_vector, vector))
            for record_id, vector in self.vectors.items()
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]


class QdrantVectorStore:
    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant 向量库需要安装 qdrant-client，请先运行 pip install -e '.[dev]'。"
            ) from exc

        self.client = QdrantClient(
            url=config.url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=config.api_key or os.getenv("QDRANT_API_KEY"),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "vector_store": "qdrant",
            "collection_name": self.config.collection_name,
            "url": self.config.url or os.getenv("QDRANT_URL", "http://localhost:6333"),
        }

    def is_available(self) -> bool:
        try:
            return self.client.collection_exists(self.config.collection_name)
        except Exception:
            return False

    def upsert(
        self,
        records: list[tuple[str, list[float], dict[str, Any]]],
        metadata: dict[str, Any],
    ) -> None:
        if not records:
            return

        from qdrant_client.models import Distance, PointStruct, VectorParams

        vector_size = self.config.vector_size or len(records[0][1])
        if self.client.collection_exists(self.config.collection_name):
            if self.config.recreate_collection:
                self.client.delete_collection(self.config.collection_name)

        if not self.client.collection_exists(self.config.collection_name):
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        batch_size = max(self.config.batch_size, 1)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            points = [
                PointStruct(
                    id=start + index,
                    vector=vector,
                    payload={
                        **payload,
                        "chunk_id": record_id,
                        "vector_store_metadata": metadata,
                    },
                )
                for index, (record_id, vector, payload) in enumerate(batch)
            ]
            self.client.upsert(collection_name=self.config.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int = 80) -> list[VectorSearchHit]:
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits: list[VectorSearchHit] = []
        for item in response.points:
            payload = item.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                hits.append(VectorSearchHit(str(chunk_id), float(item.score)))
        return hits


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    if config.provider == "qdrant":
        return QdrantVectorStore(config)
    if not config.path:
        raise ValueError("JSON vector store requires a path.")
    return JsonVectorStore(config.path)

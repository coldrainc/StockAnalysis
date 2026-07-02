from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from stock_agent.rag.embedding import (
    DEFAULT_EMBEDDING_SERVICE_URL,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
)


@dataclass(frozen=True)
class AppSettings:
    knowledge_base_path: Path = Path("knowledge_base/stock_universe")
    rag_index_path: Path = Path(".stock_agent/rag_index.json")
    rag_vector_path: Path = Path(".stock_agent/rag_vectors.json")
    vector_store_metadata_path: Path = Path(".stock_agent/vector_store.json")
    memory_path: Path = Path(".stock_agent/memory")
    vector_store: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "stock_agent"
    embedding_provider: str = "service"
    embedding_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL
    embedding_service_url: str = DEFAULT_EMBEDDING_SERVICE_URL
    rag_top_k: int = 4
    rag_max_chars: int = 6000


def load_settings() -> AppSettings:
    return AppSettings(
        knowledge_base_path=Path(
            os.getenv("STOCK_KNOWLEDGE_BASE", "knowledge_base/stock_universe")
        ),
        rag_index_path=Path(os.getenv("STOCK_RAG_INDEX", ".stock_agent/rag_index.json")),
        rag_vector_path=Path(os.getenv("STOCK_RAG_VECTORS", ".stock_agent/rag_vectors.json")),
        vector_store_metadata_path=Path(
            os.getenv("STOCK_VECTOR_STORE_METADATA", ".stock_agent/vector_store.json")
        ),
        memory_path=Path(os.getenv("STOCK_MEMORY_PATH", ".stock_agent/memory")),
        vector_store=os.getenv("STOCK_VECTOR_STORE", "qdrant"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "stock_agent"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "service"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL),
        embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", DEFAULT_EMBEDDING_SERVICE_URL),
        rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
        rag_max_chars=int(os.getenv("RAG_MAX_CHARS", "6000")),
    )

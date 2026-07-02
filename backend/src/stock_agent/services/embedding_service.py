from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from stock_agent.rag.embedding import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    EmbeddingConfig,
    LocalEmbeddingClient,
)


class EmbeddingServiceSettings(BaseModel):
    model: str = DEFAULT_LOCAL_EMBEDDING_MODEL
    batch_size: int = 64
    device: str | None = None
    local_files_only: bool = True


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    vectors: list[list[float]]


def create_app(settings: EmbeddingServiceSettings | None = None) -> FastAPI:
    service_settings = settings or EmbeddingServiceSettings()
    state: dict[str, LocalEmbeddingClient] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["client"] = LocalEmbeddingClient(
            EmbeddingConfig(
                provider="local",
                model=service_settings.model,
                batch_size=service_settings.batch_size,
                device=service_settings.device,
                local_files_only=service_settings.local_files_only,
            )
        )
        yield
        state.clear()

    app = FastAPI(title="Stock Agent Embedding Service", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": service_settings.model,
            "provider": "local",
        }

    @app.post("/embed", response_model=EmbedResponse)
    def embed(request: EmbedRequest) -> EmbedResponse:
        vectors = state["client"].embed_texts(request.texts)
        dimensions = len(vectors[0]) if vectors else 0
        return EmbedResponse(
            model=service_settings.model,
            dimensions=dimensions,
            vectors=vectors,
        )

    @app.post("/embed-query", response_model=EmbedResponse)
    def embed_query(request: EmbedRequest) -> EmbedResponse:
        text = request.texts[0]
        vector = state["client"].embed_query(text)
        return EmbedResponse(
            model=service_settings.model,
            dimensions=len(vector),
            vectors=[vector],
        )

    return app


app = create_app()

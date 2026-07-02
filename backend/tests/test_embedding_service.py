from fastapi.testclient import TestClient

from stock_agent.services.embedding_service import EmbeddingServiceSettings, create_app


class FakeLocalEmbeddingClient:
    def __init__(self, config) -> None:
        self.config = config

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def test_embedding_service_embeds_texts(monkeypatch) -> None:
    import stock_agent.services.embedding_service as service_module

    monkeypatch.setattr(service_module, "LocalEmbeddingClient", FakeLocalEmbeddingClient)
    app = create_app(EmbeddingServiceSettings(model="fake-model"))

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post("/embed", json={"texts": ["RAG", "AgentLoop"]})

    assert health.json()["model"] == "fake-model"
    assert response.status_code == 200
    assert response.json()["vectors"] == [[3.0, 1.0], [9.0, 1.0]]
    assert response.json()["dimensions"] == 2

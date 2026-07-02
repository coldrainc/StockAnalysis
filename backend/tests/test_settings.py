from stock_agent.core.settings import load_settings


def test_load_settings_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "service")
    monkeypatch.setenv("RAG_TOP_K", "8")

    settings = load_settings()

    assert settings.qdrant_collection == "test_collection"
    assert settings.embedding_provider == "service"
    assert settings.rag_top_k == 8

from pathlib import Path

from stock_agent.rag.knowledge_base import MarkdownKnowledgeBase
from stock_agent.rag.rag_index import PersistentRagIndex, RagIndexer
from stock_agent.rag.vector_store import JsonVectorStore, VectorSearchHit


class FakeEmbeddingConfig:
    model = "fake-embedding"
    provider = "local"


class FakeEmbeddingClient:
    config = FakeEmbeddingConfig()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "rag" in lowered or "检索" in lowered else 0.0,
            1.0 if "agent" in lowered or "循环" in lowered else 0.0,
            1.0,
        ]


class BiasedVectorStore(JsonVectorStore):
    def search(self, query_vector: list[float], top_k: int = 80) -> list[VectorSearchHit]:
        payload = self.payload
        hits = []
        for item in payload.get("vectors", []):
            score = 0.95 if item["id"] == "chunk-000001" else 0.10
            hits.append(VectorSearchHit(item["id"], score))
        return hits[:top_k]


def test_rag_index_builds_and_searches(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text(
        "# RAG\n\nRAG 是检索增强生成，包含检索、重排和生成。",
        encoding="utf-8",
    )
    (docs / "agent.md").write_text(
        "# AgentLoop\n\nAgentLoop 需要维护状态、工具调用和终止条件。",
        encoding="utf-8",
    )
    index_path = tmp_path / "rag_index.json"

    payload = RagIndexer(docs, index_path).build()
    index = PersistentRagIndex(index_path)
    hits = index.search("RAG 检索生成", top_k=1)

    assert payload["chunk_count"] >= 2
    assert index_path.exists()
    assert len(hits) == 1
    assert hits[0].chunk.source == Path("docs/rag.md")


def test_knowledge_base_prefers_persistent_index(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("# RAG\n\nRAG 是检索增强生成。", encoding="utf-8")
    index_path = tmp_path / "rag_index.json"
    RagIndexer(docs, index_path).build()

    kb = MarkdownKnowledgeBase(docs, index_path=index_path)

    assert kb.retrieval_mode == "bm25"
    assert kb.is_loaded is False
    assert "Score:" in kb.context_for("RAG", top_k=1)


def test_rag_index_includes_multiple_roots(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    memory = tmp_path / "memory"
    docs.mkdir()
    memory.mkdir()
    (docs / "rag.md").write_text("# RAG\n\nRAG 是检索增强生成。", encoding="utf-8")
    (memory / "session.md").write_text(
        "# 历史股票研究可复用知识\n\n## Q1\n\n用户输入：招商银行需要关注净息差和资产质量。",
        encoding="utf-8",
    )
    index_path = tmp_path / "rag_index.json"

    RagIndexer([docs, memory], index_path).build()
    index = PersistentRagIndex(index_path)

    assert index.search("招商银行 净息差", top_k=1)[0].chunk.source == Path("memory/session.md")


def test_rag_index_builds_vectors_and_uses_hybrid_mode(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("# RAG\n\nRAG 是检索增强生成。", encoding="utf-8")
    index_path = tmp_path / "rag_index.json"
    vector_path = tmp_path / "rag_vectors.json"
    embedding_client = FakeEmbeddingClient()

    RagIndexer(
        docs,
        index_path,
        vector_path=vector_path,
        embedding_client=embedding_client,
    ).build()
    index = PersistentRagIndex(
        index_path,
        vector_path=vector_path,
        embedding_client=embedding_client,
    )

    assert vector_path.exists()
    assert '"embedding_provider": "local"' in vector_path.read_text(encoding="utf-8")
    assert index.vector_metadata["embedding_provider"] == "local"
    assert index.vector_metadata["embedding_model"] == "fake-embedding"
    assert index.retrieval_mode == "hybrid-bm25-vector"
    assert "hybrid=" in index.context_for("RAG", top_k=1)


def test_rag_index_uses_vector_store_abstraction(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("# RAG\n\nRAG 是检索增强生成。", encoding="utf-8")
    index_path = tmp_path / "rag_index.json"
    vector_path = tmp_path / "rag_vectors.json"
    vector_store = JsonVectorStore(vector_path)
    embedding_client = FakeEmbeddingClient()

    RagIndexer(
        docs,
        index_path,
        embedding_client=embedding_client,
        vector_store=vector_store,
    ).build()
    index = PersistentRagIndex(
        index_path,
        embedding_client=embedding_client,
        vector_store=JsonVectorStore(vector_path),
    )

    assert vector_store.is_available()
    assert index.vector_metadata["vector_store"] == "json"
    assert index.search("RAG", top_k=1)[0].reason.startswith("hybrid=")


def test_hybrid_search_prefers_explicit_company_identifier(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "688981_中芯国际.md").write_text(
        "# 中芯国际\n\n## RAG 元数据\n\n- 公司：中芯国际\n- 资料标签：#needs_verification\n\n半导体 晶圆 代工。",
        encoding="utf-8",
    )
    (docs / "920179_凯德石英.md").write_text(
        "# 凯德石英\n\n## RAG 元数据\n\n- 公司：凯德石英\n- 资料标签：#needs_verification\n\n半导体材料。",
        encoding="utf-8",
    )
    index_path = tmp_path / "rag_index.json"
    vector_path = tmp_path / "rag_vectors.json"
    vector_store = BiasedVectorStore(vector_path)

    RagIndexer(
        docs,
        index_path,
        embedding_client=FakeEmbeddingClient(),
        vector_store=vector_store,
    ).build()
    index = PersistentRagIndex(
        index_path,
        embedding_client=FakeEmbeddingClient(),
        vector_store=BiasedVectorStore(vector_path),
    )

    hit = index.search("中芯国际 半导体 晶圆 资料标签", top_k=1)[0]

    assert hit.chunk.source == Path("docs/688981_中芯国际.md")

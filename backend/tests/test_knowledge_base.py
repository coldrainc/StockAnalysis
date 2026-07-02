from pathlib import Path

from stock_agent.rag.knowledge_base import MarkdownKnowledgeBase


def test_markdown_knowledge_base_retrieves_relevant_chunk(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text(
        "# RAG\n\nRAG uses retrieval and generation.\n\n## Chunking\n\nChunk size affects recall.",
        encoding="utf-8",
    )
    (docs / "agent.md").write_text(
        "# Agent\n\nAgentLoop controls planning and tool use.",
        encoding="utf-8",
    )

    kb = MarkdownKnowledgeBase(docs)
    assert kb.is_loaded is False
    results = kb.search("RAG retrieval", top_k=1)

    assert kb.is_loaded is True
    assert len(results) == 1
    assert results[0].source == Path("rag.md")
    assert "retrieval" in results[0].content


def test_markdown_knowledge_base_returns_context(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text(
        "# RAG\n\nRAG 通过检索知识库内容，并结合生成模型回答问题。\n\n"
        "Embedding 可以把文本转换成向量，用向量数据库做语义召回。",
        encoding="utf-8",
    )

    kb = MarkdownKnowledgeBase(docs)
    assert kb.is_loaded is False
    context = kb.context_for("RAG 向量数据库 Embedding", top_k=2)

    assert kb.is_loaded is True
    assert "Source:" in context
    assert "RAG" in context or "Embedding" in context

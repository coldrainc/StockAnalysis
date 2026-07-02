from pathlib import Path

from stock_agent.rag.knowledge_base import MarkdownKnowledgeBase
from stock_agent.interfaces.terminal import (
    TerminalCommandKind,
    parse_terminal_command,
    render_web_search_results,
    render_search_results,
)


def test_parse_search_command() -> None:
    command = parse_terminal_command("/search RAG 检索")

    assert command.kind == TerminalCommandKind.KB_SEARCH
    assert command.payload == "RAG 检索"


def test_parse_chinese_search_command() -> None:
    command = parse_terminal_command("搜索 AgentLoop")

    assert command.kind == TerminalCommandKind.KB_SEARCH
    assert command.payload == "AgentLoop"


def test_parse_web_search_command() -> None:
    command = parse_terminal_command("/web RAG 最新优化")

    assert command.kind == TerminalCommandKind.WEB_SEARCH
    assert command.payload == "RAG 最新优化"


def test_render_search_results(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("# RAG\n\nRAG 包含检索、重排和生成。", encoding="utf-8")
    kb = MarkdownKnowledgeBase(docs)

    text = render_search_results(kb, "RAG", top_k=1)

    assert "知识库搜索：RAG" in text
    assert "rag.md" in text
    assert "检索" in text


def test_render_web_search_results() -> None:
    text = render_web_search_results("标题：示例\n链接：https://example.test", "RAG")

    assert "联网搜索：RAG" in text
    assert "https://example.test" in text

from stock_agent.services.web_search import WebSearchClient, WebSearchResult


def test_web_search_context_uses_cached_results(monkeypatch) -> None:
    client = WebSearchClient()

    def fake_search(query: str, top_k: int):
        return [WebSearchResult(title="RAG", url="https://example.test", snippet="检索增强生成")]

    monkeypatch.setattr(client, "search", fake_search)

    context = client.context_for("RAG")

    assert "标题：RAG" in context
    assert "检索增强生成" in context


def test_web_search_returns_empty_context_on_request_failure(monkeypatch) -> None:
    client = WebSearchClient()

    def fail_search(query: str, top_k: int):
        import requests

        raise requests.ConnectionError("network down")

    monkeypatch.setattr(client, "_search_duckduckgo_html", fail_search)

    assert client.context_for("RAG") == "未检索到联网搜索结果。"

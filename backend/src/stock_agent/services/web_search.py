from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import quote_plus

import requests


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str

    def render(self) -> str:
        return f"标题：{self.title}\n链接：{self.url}\n摘要：{self.snippet}".strip()


class WebSearchClient:
    """Small web search facade with API providers and a no-key HTML fallback."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout
        self._cache: dict[tuple[str, int], list[WebSearchResult]] = {}

    def search(self, query: str, top_k: int = 3) -> list[WebSearchResult]:
        query = query.strip()
        if not query:
            return []
        cache_key = (query, top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            results = self._search_with_configured_provider(query, top_k)
            if not results:
                results = self._search_duckduckgo_html(query, top_k)
        except requests.RequestException:
            results = []
        self._cache[cache_key] = results
        return results

    def context_for(self, query: str, top_k: int = 3, max_chars: int = 2400) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "未检索到联网搜索结果。"

        rendered: list[str] = []
        total = 0
        for result in results:
            text = result.render()
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining <= 0:
                    break
                text = text[:remaining].rstrip()
            rendered.append(text)
            total += len(text)
        return "\n\n---\n\n".join(rendered)

    def _search_with_configured_provider(
        self, query: str, top_k: int
    ) -> list[WebSearchResult]:
        if os.getenv("TAVILY_API_KEY"):
            return self._search_tavily(query, top_k)
        if os.getenv("BRAVE_SEARCH_API_KEY"):
            return self._search_brave(query, top_k)
        if os.getenv("SEARXNG_URL"):
            return self._search_searxng(query, top_k)
        return []

    def _search_tavily(self, query: str, top_k: int) -> list[WebSearchResult]:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.getenv("TAVILY_API_KEY"),
                "query": query,
                "max_results": top_k,
                "search_depth": "basic",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [
            WebSearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", "")),
            )
            for item in data.get("results", [])[:top_k]
        ]

    def _search_brave(self, query: str, top_k: int) -> list[WebSearchResult]:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": top_k},
            headers={"X-Subscription-Token": os.getenv("BRAVE_SEARCH_API_KEY", "")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("web", {}).get("results", [])
        return [
            WebSearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("description", "")),
            )
            for item in items[:top_k]
        ]

    def _search_searxng(self, query: str, top_k: int) -> list[WebSearchResult]:
        base_url = os.getenv("SEARXNG_URL", "").rstrip("/")
        response = requests.get(
            f"{base_url}/search",
            params={"q": query, "format": "json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [
            WebSearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", "")),
            )
            for item in data.get("results", [])[:top_k]
        ]

    def _search_duckduckgo_html(self, query: str, top_k: int) -> list[WebSearchResult]:
        response = requests.get(
            f"https://duckduckgo.com/html/?q={quote_plus(query)}",
            headers={"User-Agent": "stock-agent/0.1"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        html = response.text
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)".*?>(?P<title>.*?)</a>.*?'
            r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
            re.S,
        )
        results: list[WebSearchResult] = []
        for match in pattern.finditer(html):
            results.append(
                WebSearchResult(
                    title=self._clean_html(match.group("title")),
                    url=unescape(match.group("url")),
                    snippet=self._clean_html(match.group("snippet")),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _clean_html(self, value: str) -> str:
        value = re.sub(r"<.*?>", "", value)
        value = unescape(value)
        return re.sub(r"\s+", " ", value).strip()

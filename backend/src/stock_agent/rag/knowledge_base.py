from __future__ import annotations

import re
from pathlib import Path

from stock_agent.rag.knowledge_types import KnowledgeChunk
from stock_agent.rag.rag_index import PersistentRagIndex


class MarkdownKnowledgeBase:
    """Knowledge base facade: persistent RAG index first, Markdown fallback second."""

    def __init__(
        self,
        root: Path,
        chunk_size: int = 2400,
        index_path: Path | None = None,
        vector_path: Path | None = None,
        embedding_client=None,
        vector_store=None,
    ) -> None:
        self.root = root
        self.chunk_size = chunk_size
        self.index_path = index_path
        self.index = (
            PersistentRagIndex(
                index_path,
                vector_path=vector_path,
                embedding_client=embedding_client,
                vector_store=vector_store,
            )
            if index_path and index_path.exists()
            else None
        )
        self._chunks: list[KnowledgeChunk] | None = None

    @property
    def chunks(self) -> list[KnowledgeChunk]:
        if self._chunks is None:
            self._chunks = self._load_chunks()
        return self._chunks

    @property
    def is_loaded(self) -> bool:
        return self._chunks is not None

    def estimated_file_count(self) -> int:
        if not self.root.exists():
            return 0
        return sum(1 for _ in self.root.rglob("*.md"))

    @property
    def retrieval_mode(self) -> str:
        if self.index:
            return self.index.retrieval_mode
        return "markdown-fallback"

    def search(self, query: str, top_k: int = 4) -> list[KnowledgeChunk]:
        if self.index:
            return [hit.chunk for hit in self.index.search(query, top_k=top_k)]

        terms = self._tokenize(query)
        if not terms:
            return self.chunks[:top_k]

        scored: list[tuple[int, KnowledgeChunk]] = []
        for chunk in self.chunks:
            haystack = self._normalize(f"{chunk.heading}\n{chunk.content}")
            score = sum(haystack.count(term) for term in terms)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            return [chunk for _, chunk in scored[:top_k]]
        return self.chunks[:top_k]

    def context_for(self, query: str, top_k: int = 4, max_chars: int = 6000) -> str:
        if self.index:
            return self.index.context_for(query, top_k=top_k, max_chars=max_chars)

        rendered: list[str] = []
        total = 0
        for chunk in self.search(query, top_k=top_k):
            text = chunk.render()
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining <= 0:
                    break
                text = text[:remaining].rstrip()
            rendered.append(text)
            total += len(text)
        return "\n\n---\n\n".join(rendered)

    def _load_chunks(self) -> list[KnowledgeChunk]:
        if not self.root.exists():
            return []

        chunks: list[KnowledgeChunk] = []
        for path in sorted(self.root.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            chunks.extend(self._split_markdown(path, text))
        return chunks

    def _split_markdown(self, path: Path, text: str) -> list[KnowledgeChunk]:
        sections = re.split(r"(?m)^(#{1,4}\s+.+)$", text)
        chunks: list[KnowledgeChunk] = []
        current_heading = path.stem

        for section in sections:
            if not section.strip():
                continue
            if section.lstrip().startswith("#"):
                current_heading = section.strip().lstrip("#").strip()
                continue
            for part in self._split_by_size(section.strip()):
                chunks.append(
                    KnowledgeChunk(
                        source=path.relative_to(self.root),
                        heading=current_heading,
                        content=part,
                    )
                )
        return chunks

    def _split_by_size(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for paragraph in paragraphs:
            if current and current_len + len(paragraph) > self.chunk_size:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(paragraph)
            current_len += len(paragraph)
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _tokenize(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        return [term for term in re.split(r"\W+", normalized) if len(term) >= 2]

    def _normalize(self, text: str) -> str:
        return text.lower()

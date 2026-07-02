from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stock_agent.rag.knowledge_types import KnowledgeChunk
from stock_agent.rag.vector_store import JsonVectorStore, VectorStore


INDEX_VERSION = 1


@dataclass(frozen=True)
class IndexedChunk:
    id: str
    source: str
    heading: str
    content: str
    tokens: list[str]
    metadata: dict[str, Any]

    def to_knowledge_chunk(self) -> KnowledgeChunk:
        return KnowledgeChunk(
            source=Path(self.source),
            heading=self.heading,
            content=self.content,
        )


@dataclass(frozen=True)
class SearchHit:
    chunk: KnowledgeChunk
    score: float
    reason: str


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: list[float]


class RagIndexer:
    def __init__(
        self,
        source_root: Path | list[Path],
        index_path: Path,
        chunk_size: int = 1800,
        vector_path: Path | None = None,
        embedding_client: Any | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.source_roots = [source_root] if isinstance(source_root, Path) else source_root
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.vector_path = vector_path
        self.embedding_client = embedding_client
        self.vector_store = vector_store or (
            JsonVectorStore(vector_path) if vector_path else None
        )

    def build(self) -> dict[str, Any]:
        from stock_agent.rag.knowledge_base import MarkdownKnowledgeBase

        chunks: list[IndexedChunk] = []
        doc_freq: Counter[str] = Counter()
        total_length = 0

        for index, chunk in enumerate(self._iter_chunks()):
            text = f"{chunk.heading}\n{chunk.content}"
            tokens = tokenize(text)
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
            total_length += len(tokens)
            chunks.append(
                IndexedChunk(
                    id=f"chunk-{index:06d}",
                    source=str(chunk.source),
                    heading=chunk.heading,
                    content=chunk.content,
                    tokens=tokens,
                    metadata={
                        "source": str(chunk.source),
                        "heading": chunk.heading,
                        "token_count": len(tokens),
                    },
                )
            )

        payload = {
            "version": INDEX_VERSION,
            "source_roots": [str(root) for root in self.source_roots],
            "chunk_size": self.chunk_size,
            "chunk_count": len(chunks),
            "avg_doc_len": total_length / max(len(chunks), 1),
            "doc_freq": dict(doc_freq),
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if self.vector_store and self.embedding_client and chunks:
            self._build_vector_index(chunks)
        return payload

    def _build_vector_index(self, chunks: list[IndexedChunk]) -> None:
        texts = [embedding_text(chunk) for chunk in chunks]
        vectors = self.embedding_client.embed_texts(texts)
        metadata = {
            "version": INDEX_VERSION,
            "embedding_provider": getattr(self.embedding_client.config, "provider", "openai"),
            "embedding_model": self.embedding_client.config.model,
            "chunk_count": len(chunks),
        }
        records = [
            (
                chunk.id,
                vector,
                {
                    "source": chunk.source,
                    "heading": chunk.heading,
                    "token_count": chunk.metadata.get("token_count"),
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if self.vector_store:
            self.vector_store.upsert(records, metadata)

    def _iter_chunks(self):
        from stock_agent.rag.knowledge_base import MarkdownKnowledgeBase

        for root in self.source_roots:
            if not root.exists():
                continue
            kb = MarkdownKnowledgeBase(root, chunk_size=self.chunk_size)
            for chunk in kb.chunks:
                yield KnowledgeChunk(
                    source=Path(root.name) / chunk.source,
                    heading=chunk.heading,
                    content=chunk.content,
                )


class PersistentRagIndex:
    def __init__(
        self,
        index_path: Path,
        vector_path: Path | None = None,
        embedding_client: Any | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.index_path = index_path
        self.vector_path = vector_path
        self.embedding_client = embedding_client
        self.vector_store = vector_store or (
            JsonVectorStore(vector_path) if vector_path else None
        )
        self._payload: dict[str, Any] | None = None
        self._chunks: list[IndexedChunk] | None = None

    @property
    def is_available(self) -> bool:
        return self.index_path.exists()

    @property
    def payload(self) -> dict[str, Any]:
        if self._payload is None:
            self._payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        return self._payload

    @property
    def chunks(self) -> list[IndexedChunk]:
        if self._chunks is None:
            self._chunks = [
                IndexedChunk(
                    id=item["id"],
                    source=item["source"],
                    heading=item["heading"],
                    content=item["content"],
                    tokens=list(item["tokens"]),
                    metadata=dict(item["metadata"]),
                )
                for item in self.payload.get("chunks", [])
            ]
        return self._chunks

    @property
    def vectors(self) -> dict[str, list[float]]:
        if isinstance(self.vector_store, JsonVectorStore):
            return self.vector_store.vectors
        return {}

    @property
    def vector_metadata(self) -> dict[str, Any]:
        if not self.vector_store:
            return {}
        return self.vector_store.metadata

    @property
    def retrieval_mode(self) -> str:
        if self.vector_store and self.vector_store.is_available() and self.embedding_client:
            return "hybrid-bm25-vector"
        return "bm25"

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        query_tokens = expand_query_tokens(tokenize(query))
        if not query_tokens:
            return []

        bm25_scores: list[tuple[float, IndexedChunk]] = []
        for chunk in self.chunks:
            score = self._bm25_score(query_tokens, chunk)
            if score > 0:
                bm25_scores.append((score, chunk))

        if self.vector_store and self.vector_store.is_available() and self.embedding_client:
            scored = self._hybrid_scores(query, bm25_scores, query_tokens)
        else:
            scored = bm25_scores

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = self._mmr_select(scored, query_tokens, top_k)
        return [
            SearchHit(
                chunk=chunk.to_knowledge_chunk(),
                score=score,
                reason=self._score_reason(score),
            )
            for score, chunk in selected
        ]

    def _hybrid_scores(
        self,
        query: str,
        bm25_scores: list[tuple[float, IndexedChunk]],
        query_tokens: list[str],
    ) -> list[tuple[float, IndexedChunk]]:
        from stock_agent.rag.embedding import cosine_similarity

        bm25_by_id = {chunk.id: score for score, chunk in bm25_scores}
        max_bm25 = max(bm25_by_id.values(), default=1.0)
        try:
            query_vector = self.embedding_client.embed_query(query)
        except Exception:
            return bm25_scores

        candidates: dict[str, IndexedChunk] = {}
        for score, chunk in sorted(bm25_scores, key=lambda item: item[0], reverse=True)[:80]:
            candidates[chunk.id] = chunk
        if self.vector_store:
            for hit in self.vector_store.search(query_vector, top_k=80):
                chunk = self._chunk_by_id(hit.id)
                if chunk:
                    candidates.setdefault(chunk.id, chunk)
        for chunk in self.chunks:
            if any(token in chunk.tokens for token in query_tokens):
                candidates.setdefault(chunk.id, chunk)

        scored: list[tuple[float, IndexedChunk]] = []
        vector_scores = {}
        if self.vector_store:
            vector_scores = {
                hit.id: hit.score for hit in self.vector_store.search(query_vector, top_k=200)
            }
        for chunk in candidates.values():
            vector = self.vectors.get(chunk.id)
            vector_score = vector_scores.get(chunk.id)
            if vector_score is None:
                vector_score = cosine_similarity(query_vector, vector) if vector else 0.0
            sparse_score = bm25_by_id.get(chunk.id, 0.0) / max_bm25
            exact_score = exact_identifier_bonus(query, chunk)
            hybrid_score = 0.55 * vector_score + 0.45 * sparse_score + exact_score
            scored.append((hybrid_score, chunk))
        return scored

    def _chunk_by_id(self, chunk_id: str) -> IndexedChunk | None:
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def _score_reason(self, score: float) -> str:
        if self.retrieval_mode == "hybrid-bm25-vector":
            return f"hybrid={score:.3f}"
        return f"bm25={score:.3f}"

    def context_for(self, query: str, top_k: int = 4, max_chars: int = 6000) -> str:
        rendered: list[str] = []
        total = 0
        for hit in self.search(query, top_k=top_k):
            text = (
                f"Source: {hit.chunk.source}\n"
                f"Heading: {hit.chunk.heading}\n"
                f"Score: {hit.score:.3f} ({hit.reason})\n"
                f"{hit.chunk.content}"
            ).strip()
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining <= 0:
                    break
                text = text[:remaining].rstrip()
            rendered.append(text)
            total += len(text)
        return "\n\n---\n\n".join(rendered)

    def _bm25_score(self, query_tokens: list[str], chunk: IndexedChunk) -> float:
        k1 = 1.5
        b = 0.75
        avg_doc_len = float(self.payload.get("avg_doc_len", 1.0)) or 1.0
        doc_freq = self.payload.get("doc_freq", {})
        total_docs = max(int(self.payload.get("chunk_count", 1)), 1)
        counts = Counter(chunk.tokens)
        doc_len = max(len(chunk.tokens), 1)

        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            df = int(doc_freq.get(token, 0))
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def _mmr_select(
        self,
        scored: list[tuple[float, IndexedChunk]],
        query_tokens: list[str],
        top_k: int,
    ) -> list[tuple[float, IndexedChunk]]:
        selected: list[tuple[float, IndexedChunk]] = []
        candidates = scored[: max(top_k * 6, top_k)]
        query_set = set(query_tokens)

        while candidates and len(selected) < top_k:
            best_index = 0
            best_value = float("-inf")
            selected_sources = {chunk.source for _, chunk in selected}
            for index, (score, chunk) in enumerate(candidates):
                diversity_penalty = 0.25 if chunk.source in selected_sources else 0.0
                overlap_bonus = 0.05 * len(query_set.intersection(chunk.tokens))
                value = score + overlap_bonus - diversity_penalty
                if value > best_value:
                    best_index = index
                    best_value = value
            selected.append(candidates.pop(best_index))
        return selected


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    latin = re.findall(r"[a-z0-9][a-z0-9_+-]{1,}", normalized)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    chinese_bigrams: list[str] = []
    for phrase in chinese:
        chinese_bigrams.extend(phrase[index : index + 2] for index in range(len(phrase) - 1))
    return latin + chinese + chinese_bigrams


def expand_query_tokens(tokens: list[str]) -> list[str]:
    synonyms = {
        "rag": ["retrieval", "augmented", "generation", "检索", "增强", "生成"],
        "agentloop": ["agent", "loop", "循环", "状态", "工具"],
        "harness": ["评估", "测试", "回归", "护栏"],
        "rerank": ["重排", "召回", "相关性"],
    }
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(synonyms.get(token, []))
    return expanded


GENERIC_EXACT_TERMS = {
    "资料标签",
    "市场分类",
    "股票代码",
    "数据来源",
    "基本面",
    "科技相关",
    "动态刷新",
    "行情快照",
    "公告线索",
    "风险清单",
    "半导体",
    "锂电池",
    "机器人",
    "自动化",
    "新能源",
}


def exact_identifier_bonus(query: str, chunk: IndexedChunk) -> float:
    """Prefer the named company/code document when the user explicitly names it."""

    terms = exact_identifier_terms(query)
    if not terms:
        return 0.0

    source_text = chunk.source.lower()
    heading_text = chunk.heading.lower()
    content_text = chunk.content.lower()
    bonus = 0.0
    for term in terms:
        normalized = term.lower()
        if normalized in source_text:
            bonus = max(bonus, 0.35)
        if normalized in heading_text:
            bonus = max(bonus, 0.25)
        if f"公司：{term}" in chunk.content or f"股票代码：{term}" in chunk.content:
            bonus = max(bonus, 0.25)
        if re.fullmatch(r"\d{6}(?:\.(?:sh|sz|bj))?", normalized) and normalized in content_text:
            bonus = max(bonus, 0.30)
    return bonus


def exact_identifier_terms(query: str) -> list[str]:
    terms: list[str] = []
    for term in re.findall(r"\d{6}(?:\.(?:SH|SZ|BJ))?", query, flags=re.IGNORECASE):
        terms.append(term)
    for term in re.findall(r"[\u4e00-\u9fff]{4,}", query):
        if term in GENERIC_EXACT_TERMS:
            continue
        terms.append(term)
    return list(dict.fromkeys(terms))


def embedding_text(chunk: IndexedChunk) -> str:
    return f"{chunk.heading}\n{chunk.content}".strip()

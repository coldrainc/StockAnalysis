from pathlib import Path

from stock_agent.rag.knowledge_types import KnowledgeChunk
from stock_agent.rag.rag_eval import RagEvalCase, load_eval_cases, run_rag_eval


class FakeKnowledgeBase:
    def search(self, query: str, top_k: int):
        return [
            KnowledgeChunk(
                source=Path("docs/03-rag-system/README.md"),
                heading="RAG",
                content="RAG 包含检索、召回、重排和生成。",
            )
        ]


def test_run_rag_eval_passes_expected_source_and_terms() -> None:
    results = run_rag_eval(
        FakeKnowledgeBase(),
        [
            RagEvalCase(
                query="RAG 怎么优化",
                expected_sources=["03-rag-system"],
                must_contain=["召回"],
            )
        ],
    )

    assert results[0].ok is True


def test_load_eval_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        '[{"query":"q","expected_sources":["s"],"must_contain":["m"]}]',
        encoding="utf-8",
    )

    cases = load_eval_cases(path)

    assert cases[0].query == "q"
    assert cases[0].expected_sources == ["s"]

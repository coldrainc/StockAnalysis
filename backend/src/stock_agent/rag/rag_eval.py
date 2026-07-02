from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RagEvalCase:
    query: str
    expected_sources: list[str]
    must_contain: list[str]


@dataclass(frozen=True)
class RagEvalResult:
    query: str
    ok: bool
    matched_sources: list[str]
    missing_sources: list[str]
    missing_terms: list[str]


def load_eval_cases(path: Path) -> list[RagEvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[RagEvalCase] = []
    for item in payload:
        cases.append(
            RagEvalCase(
                query=str(item["query"]),
                expected_sources=[str(source) for source in item.get("expected_sources", [])],
                must_contain=[str(term) for term in item.get("must_contain", [])],
            )
        )
    return cases


def run_rag_eval(kb: Any, cases: list[RagEvalCase], top_k: int = 4) -> list[RagEvalResult]:
    results: list[RagEvalResult] = []
    for case in cases:
        hits = kb.search(case.query, top_k=top_k) if kb else []
        sources = [str(hit.source) for hit in hits]
        combined = "\n".join(f"{hit.heading}\n{hit.content}" for hit in hits)
        matched_sources = [
            expected for expected in case.expected_sources if any(expected in source for source in sources)
        ]
        missing_sources = [
            expected for expected in case.expected_sources if expected not in matched_sources
        ]
        missing_terms = [term for term in case.must_contain if term not in combined]
        results.append(
            RagEvalResult(
                query=case.query,
                ok=not missing_sources and not missing_terms,
                matched_sources=matched_sources,
                missing_sources=missing_sources,
                missing_terms=missing_terms,
            )
        )
    return results

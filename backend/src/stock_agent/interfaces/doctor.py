from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str


def run_doctor(
    *,
    index_path: Path,
    vector_store_metadata_path: Path,
    embedding_service_url: str,
    qdrant_url: str,
    qdrant_collection: str,
) -> list[CheckResult]:
    results = [
        _check_file("RAG index", index_path),
        _check_file("Vector store metadata", vector_store_metadata_path),
        _check_embedding_service(embedding_service_url),
        _check_qdrant(qdrant_url, qdrant_collection),
    ]
    return results


def _check_file(name: str, path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(name, False, f"missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return CheckResult(name, False, f"invalid json: {exc}")
    count = payload.get("chunk_count") or payload.get("embedding_model") or "ok"
    return CheckResult(name, True, f"{path} ({count})")


def _check_embedding_service(url: str) -> CheckResult:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return CheckResult("EmbeddingService", False, str(exc))
    return CheckResult(
        "EmbeddingService",
        payload.get("status") == "ok",
        f"{url} model={payload.get('model')}",
    )


def _check_qdrant(url: str, collection: str) -> CheckResult:
    try:
        response = requests.get(f"{url.rstrip('/')}/collections/{collection}", timeout=5)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result", {})
    except Exception as exc:
        return CheckResult("Qdrant", False, str(exc))
    return CheckResult(
        "Qdrant",
        payload.get("status") == "ok",
        f"{url} collection={collection} points={result.get('points_count')}",
    )

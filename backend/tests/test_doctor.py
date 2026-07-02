import json
from pathlib import Path

from stock_agent.interfaces.doctor import run_doctor


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_run_doctor_checks_files_and_services(monkeypatch, tmp_path: Path) -> None:
    index_path = tmp_path / "rag_index.json"
    metadata_path = tmp_path / "vector_store.json"
    index_path.write_text(json.dumps({"chunk_count": 2}), encoding="utf-8")
    metadata_path.write_text(json.dumps({"embedding_model": "fake"}), encoding="utf-8")

    def fake_get(url: str, timeout: int):
        if url.endswith("/health"):
            return FakeResponse({"status": "ok", "model": "fake"})
        return FakeResponse({"status": "ok", "result": {"points_count": 2}})

    monkeypatch.setattr("stock_agent.interfaces.doctor.requests.get", fake_get)

    results = run_doctor(
        index_path=index_path,
        vector_store_metadata_path=metadata_path,
        embedding_service_url="http://embedding",
        qdrant_url="http://qdrant",
        qdrant_collection="collection",
    )

    assert all(result.ok for result in results)

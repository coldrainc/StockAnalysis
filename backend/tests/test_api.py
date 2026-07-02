from fastapi.testclient import TestClient
from types import SimpleNamespace

from stock_agent.api import _response, create_app


def test_api_health_identifies_stock_agent() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["app"] == "stock-agent"
    assert payload["domain"] == "stock_quant_research"
    assert "daily_quant_picks" in payload["capabilities"]
    assert "portfolio_analysis" in payload["capabilities"]


def test_api_offline_session_uses_stock_research_opening() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/sessions",
            json={"offline": True, "web_search": False},
        )

    assert response.status_code == 200
    message = response.json()["message"]
    assert "股票量化研究" in message
    assert "持仓分析" in message
    assert "候选人" not in message
    assert "Python 工程实践" not in message


def test_api_long_response_creates_document(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    long_message = "长文分析。" * 420
    result = SimpleNamespace(
        message=long_message,
        state=SimpleNamespace(completed=False),
        fallback_used=False,
        guardrail_findings=[],
    )

    response = _response("session-1", result)

    assert response.document is not None
    assert response.analysis_trace is not None
    assert response.document.analysis_trace is not None
    assert "完整研究文档" in response.message
    assert long_message in response.document.content
    assert "## 分析流程" in response.document.content
    assert "长文分析" in response.document.title


def test_api_extracts_public_analysis_trace_from_response() -> None:
    result = SimpleNamespace(
        message=(
            "## 分析流程\n"
            "1. 识别目标：读取用户持仓和周期。\n"
            "2. 检索 RAG：匹配每日候选池和逐股文档。\n\n"
            "## 结论\n"
            "维持观察，等待行情和公告核验。"
        ),
        state=SimpleNamespace(completed=False, stage=SimpleNamespace(value="recommendation")),
        fallback_used=False,
        guardrail_findings=[],
    )

    response = _response("session-2", result)

    assert response.analysis_trace is not None
    assert response.analysis_trace.steps[0].title == "识别目标"
    assert "## 分析流程" not in response.message
    assert "## 结论" in response.message

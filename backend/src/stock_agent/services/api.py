from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stock_agent.agent.agent_loop import AgentLoop
from stock_agent.interfaces.cli import (
    default_vector_path,
    load_config,
    load_embedding_client_for_existing_vectors,
    load_knowledge_base,
    load_vector_store_for_run,
    persist_result,
)
from stock_agent.core.config import StockConfig
from stock_agent.core.conversation_store import ConversationStore
from stock_agent.core.codex_config import load_codex_model_config
from stock_agent.core.settings import load_settings
from stock_agent.services.web_search import WebSearchClient


class SessionRequest(BaseModel):
    offline: bool = False
    web_search: bool = False


class MessageRequest(BaseModel):
    message: str


class AnalysisTraceStep(BaseModel):
    title: str
    detail: str = ""
    status: str = "completed"


class AnalysisTrace(BaseModel):
    summary: str
    steps: list[AnalysisTraceStep]
    warnings: list[str] = []
    source: str = "public_summary"


class ChatDocument(BaseModel):
    id: str
    title: str
    path: str
    content: str
    created_at: str
    analysis_trace: AnalysisTrace | None = None


class ChatResponse(BaseModel):
    session_id: str
    message: str
    completed: bool
    fallback_used: bool = False
    guardrails: list[str] = []
    document: ChatDocument | None = None
    analysis_trace: AnalysisTrace | None = None


@dataclass
class ApiSession:
    loop: AgentLoop
    config: StockConfig
    store: ConversationStore


sessions: dict[str, ApiSession] = {}
RESPONSE_DOCUMENT_THRESHOLD = 1600
RESEARCH_DOC_ROOT = Path(".stock_agent/research_docs")


def create_app() -> FastAPI:
    app = FastAPI(title="Stock Agent API")

    @app.get("/health")
    def health() -> dict:
        settings = load_settings()
        model_config = load_codex_model_config(Path.cwd())
        return {
            "status": "ok",
            "app": "stock-agent",
            "service": "stock-agent-api",
            "domain": "stock_quant_research",
            "capabilities": [
                "daily_quant_picks",
                "portfolio_analysis",
                "rag_vector_refresh",
                "a_share_technology_rag",
                "multi_agent_orchestration",
            ],
            "default_api_url": "http://127.0.0.1:18220",
            "model_provider": model_config.provider,
            "model": model_config.model,
            "model_base_url": model_config.base_url,
            "qdrant_url": settings.qdrant_url,
            "embedding_service_url": settings.embedding_service_url,
        }

    @app.post("/sessions", response_model=ChatResponse)
    def create_session(request: SessionRequest) -> ChatResponse:
        from stock_agent.agent.harness import LangChainStockHarness, ScriptedStockHarness

        config = load_config(None)
        store = ConversationStore()
        codex_model_config = load_codex_model_config(__import__("pathlib").Path.cwd())
        embedding_client = load_embedding_client_for_existing_vectors(default_vector_path())
        vector_store = load_vector_store_for_run(default_vector_path())
        kb = load_knowledge_base(None, embedding_client=embedding_client, vector_store=vector_store)
        web_search = WebSearchClient() if request.web_search else None
        api_key = codex_model_config.api_key

        if request.offline or not api_key:
            harness = ScriptedStockHarness(config, knowledge_base=kb)
        else:
            harness = LangChainStockHarness(
                config,
                knowledge_base=kb,
                web_search=web_search,
                model=codex_model_config.model or "gpt-4o-mini",
                base_url=codex_model_config.base_url,
                api_key=api_key,
                wire_api=codex_model_config.wire_api,
            )

        loop = AgentLoop(config, harness)
        result = loop.start()
        session_id = str(uuid4())
        sessions[session_id] = ApiSession(loop=loop, config=config, store=store)
        persist_result(store, config, result, "start")
        return _response(session_id, result)

    @app.post("/sessions/{session_id}/messages", response_model=ChatResponse)
    def send_message(session_id: str, request: MessageRequest) -> ChatResponse:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session not found")
        result = session.loop.step(request.message)
        persist_result(session.store, session.config, result, "turn")
        return _response(session_id, result)

    @app.get("/sessions/{session_id}/transcript")
    def transcript(session_id: str) -> dict:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session not found")
        return {"transcript": session.loop.state.transcript()}

    return app


def _response(session_id: str, result) -> ChatResponse:
    message, extracted_trace = _split_analysis_trace(result.message)
    analysis_trace = extracted_trace or _default_analysis_trace(result)
    document = None
    if len(message) > RESPONSE_DOCUMENT_THRESHOLD:
        document = _persist_chat_document(session_id, message, analysis_trace)
        message = (
            f"已生成完整研究文档：《{document.title}》。\n\n"
            "这次分析内容较长，聊天框仅保留入口；请在桌面端打开完整文档查看全文。"
        )
    return ChatResponse(
        session_id=session_id,
        message=message,
        completed=result.state.completed,
        fallback_used=result.fallback_used,
        guardrails=[finding.message for finding in result.guardrail_findings or []],
        document=document,
        analysis_trace=analysis_trace,
    )


def _persist_chat_document(
    session_id: str,
    text: str,
    analysis_trace: AnalysisTrace | None,
) -> ChatDocument:
    RESEARCH_DOC_ROOT.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    document_id = f"{session_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    title = _document_title(text)
    file_path = RESEARCH_DOC_ROOT / f"{_safe_slug(document_id)}.md"
    content = "\n".join(
        [
            f"# {title}",
            "",
            f"- 会话：{session_id}",
            f"- 生成时间：{created_at}",
            "- 展示方式：聊天框只保留入口，以下为完整输出。",
            "",
            "---",
            "",
            _render_analysis_trace(analysis_trace),
            "",
            text.strip(),
            "",
        ]
    )
    file_path.write_text(content, encoding="utf-8")
    return ChatDocument(
        id=document_id,
        title=title,
        path=str(file_path),
        content=content,
        created_at=created_at,
        analysis_trace=analysis_trace,
    )


def _split_analysis_trace(text: str) -> tuple[str, AnalysisTrace | None]:
    heading = re.search(r"(?im)^#{1,3}\s*(分析流程|思考流程|推理摘要|研究流程)\s*$", text)
    if not heading:
        return text.strip(), None

    content_start = heading.end()
    next_heading = re.search(r"(?m)^#{1,3}\s+.+$", text[content_start:])
    content_end = content_start + next_heading.start() if next_heading else len(text)
    section = text[content_start:content_end].strip()
    cleaned = f"{text[: heading.start()].rstrip()}\n\n{text[content_end:].lstrip()}".strip()
    steps, summary = _parse_analysis_trace_section(section)
    if not steps:
        return cleaned or text.strip(), None
    return cleaned, AnalysisTrace(summary=summary, steps=steps)


def _parse_analysis_trace_section(section: str) -> tuple[list[AnalysisTraceStep], str]:
    steps: list[AnalysisTraceStep] = []
    summary = "本回答展示的是可审计的分析步骤摘要，不包含模型内部隐藏推理。"
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet = re.match(r"^(?:[-*]|\d+[.、)])\s*(.+)$", line)
        if not bullet:
            if len(line) > 8 and summary.startswith("本回答展示"):
                summary = _clean_trace_text(line)
            continue
        content = _clean_trace_text(bullet.group(1))
        title, detail = _split_trace_step(content)
        if title:
            steps.append(AnalysisTraceStep(title=title[:48], detail=detail[:180]))
        if len(steps) >= 8:
            break
    return steps, summary


def _split_trace_step(content: str) -> tuple[str, str]:
    for separator in ("：", ":"):
        if separator in content:
            title, detail = content.split(separator, 1)
            return title.strip(), detail.strip()
    return content.strip(), ""


def _clean_trace_text(value: str) -> str:
    return re.sub(r"[*_`]+", "", value).strip()


def _default_analysis_trace(result) -> AnalysisTrace:
    state = getattr(result, "state", None)
    stage = getattr(getattr(state, "stage", None), "value", None) or str(getattr(state, "stage", ""))
    warnings = [finding.message for finding in getattr(result, "guardrail_findings", None) or []]
    if getattr(result, "fallback_used", False):
        warnings.append("模型调用不可用或返回异常，本轮使用了保守回退回答。")
    steps = [
        AnalysisTraceStep(
            title="识别研究目标",
            detail=f"读取用户输入、当前对话阶段和持仓约束；当前阶段：{stage or 'unknown'}。",
        ),
        AnalysisTraceStep(
            title="多Agent分工",
            detail="DataAgent、QuantAgent、FundamentalAgent、CatalystAgent、PortfolioAgent、RiskAgent 与 SupervisorAgent 按场景协作。",
        ),
        AnalysisTraceStep(
            title="检索研究上下文",
            detail="优先使用每日候选池、逐股文档、RAG 向量库、行情刷新和已选择持仓文件。",
        ),
        AnalysisTraceStep(
            title="核验资料可信度",
            detail="区分公告/交易所资料、第三方数据、用户输入和需要刷新或复核的标签。",
        ),
        AnalysisTraceStep(
            title="交叉分析",
            detail="把量价、基本面、催化剂、估值约束、组合适配和风险项放在同一框架下比较。",
        ),
        AnalysisTraceStep(
            title="输出可执行框架",
            detail="给出观察分层、触发条件、失效条件、风险纪律和资料核验提醒。",
        ),
    ]
    return AnalysisTrace(
        summary="公开分析流程摘要，用于展示本轮回答如何组织证据与结论。",
        steps=steps,
        warnings=warnings,
        source="generated_fallback",
    )


def _render_analysis_trace(trace: AnalysisTrace | None) -> str:
    if trace is None:
        return ""
    lines = ["## 分析流程", "", trace.summary]
    for index, step in enumerate(trace.steps, start=1):
        detail = f"：{step.detail}" if step.detail else ""
        lines.append(f"{index}. {step.title}{detail}")
    if trace.warnings:
        lines.append("")
        lines.append("### 流程提醒")
        lines.extend(f"- {warning}" for warning in trace.warnings)
    return "\n".join(lines)


def _document_title(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned[:42]
    return "股票研究长文分析"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "research_document"


app = create_app()

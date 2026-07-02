import { useEffect, useMemo, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { createSession, choosePortfolio, getHealth, getWorkspace, runDailyPicks, sendMessage } from "./api/stockAgent";
import { DashboardPage } from "./pages/DashboardPage";
import { DocumentPage } from "./pages/DocumentPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { PortfolioPage } from "./pages/PortfolioPage";
import { ReportPage } from "./pages/ReportPage";
import { ResearchPage } from "./pages/ResearchPage";
import type { ChatDocument, ChatMessage, DailyPick, HealthPayload, PageId, PortfolioPreview, WorkspaceSnapshot } from "./types";

const pageTitles: Record<PageId, string> = {
  overview: "量化总览",
  opportunities: "机会池",
  research: "研究台",
  portfolio: "持仓分析",
  knowledge: "知识库与核验",
  archive: "研究档案"
};

export function App() {
  const [activePage, setActivePage] = useState<PageId>("overview");
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [workspace, setWorkspace] = useState<WorkspaceSnapshot | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioPreview | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<ChatDocument[]>([]);
  const [activeDocumentId, setActiveDocumentId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [offline, setOffline] = useState(false);
  const [webSearch, setWebSearch] = useState(false);
  const [syncVectorStore, setSyncVectorStore] = useState(true);
  const [busy, setBusy] = useState(false);
  const [typingLabel, setTypingLabel] = useState("");

  useEffect(() => {
    void refreshHealth();
    void refreshWorkspace();
  }, []);

  const modeText = useMemo(() => {
    if (busy) return "运行中";
    if (sessionId) return offline ? "离线模式" : "模型模式";
    return "准备中";
  }, [busy, offline, sessionId]);

  async function refreshHealth() {
    try {
      setHealth(await getHealth());
    } catch (error) {
      setHealth({
        status: "fail",
        app: "",
        service: "",
        domain: "",
        embedding_service_url: "请先运行 ./stock api",
        qdrant_url: error instanceof Error ? error.message : "API 未启动"
      });
    }
  }

  async function refreshWorkspace() {
    try {
      const snapshot = await getWorkspace();
      setWorkspace(snapshot);
      setPortfolio((current) => current || snapshot.defaultPortfolio || null);
    } catch (error) {
      addSystemMessage(`读取工作区失败：${errorText(error)}`);
    }
  }

  async function handleCreateSession() {
    if (busy) return null;
    setBusy(true);
    setTypingLabel("正在创建会话...");
    setMessages([]);
    try {
      const response = await createSession({ offline, webSearch });
      setSessionId(response.session_id);
      setActivePage("research");
      addAgentMessage(response.message, response.document, response.analysis_trace);
      renderGuardrails(response.guardrails);
      return response.session_id;
    } catch (error) {
      addSystemMessage(`创建会话失败：${errorText(error)}`);
      return null;
    } finally {
      setBusy(false);
      setTypingLabel("");
    }
  }

  async function ensureSession() {
    if (sessionId) return sessionId;
    return handleCreateSession();
  }

  async function handleSend(text: string) {
    if (!text.trim() || busy) return;
    const activeSessionId = await ensureSession();
    if (!activeSessionId) return;
    addUserMessage(text);
    setBusy(true);
    setTypingLabel("量化分析师正在研究...");
    try {
      const response = await sendMessage({ sessionId: activeSessionId, message: text });
      addAgentMessage(response.message, response.document, response.analysis_trace);
      renderGuardrails(response.guardrails);
    } catch (error) {
      addSystemMessage(`发送失败：${errorText(error)}`);
    } finally {
      setBusy(false);
      setTypingLabel("");
    }
  }

  async function handleChoosePortfolio() {
    if (busy) return;
    try {
      const selected = await choosePortfolio();
      if (!selected) return;
      setPortfolio(selected);
      setActivePage("portfolio");
    } catch (error) {
      addSystemMessage(`读取持仓失败：${errorText(error)}`);
    }
  }

  async function handleRunDailyPicks() {
    if (busy) return;
    setBusy(true);
    setTypingLabel("正在刷新每日候选池...");
    try {
      const response = await runDailyPicks({
        portfolio: portfolio?.path,
        maxCandidates: 800,
        topK: 30,
        syncVectorStore
      });
      setWorkspace(response.workspace);
      addSystemMessage(`每日候选池已刷新。${syncVectorStore ? "A股 RAG/向量库已同步。" : "已跳过向量库同步。"}`);
      setActivePage("opportunities");
    } catch (error) {
      addSystemMessage(`刷新失败：${errorText(error)}`);
    } finally {
      setBusy(false);
      setTypingLabel("");
    }
  }

  function analyzePick(pick: DailyPick) {
    void handleSend(
      `请详细分析 ${pick.name}（${pick.code}）：解释它进入今日候选池的评分逻辑、上涨触发条件、失效条件、主要风险、资料核验点，并说明是否适合加入观察仓。`
    );
  }

  function addUserMessage(text: string) {
    setMessages((items) => [...items, makeMessage("user", text)]);
  }

  function addAgentMessage(text: string, document?: ChatDocument | null, analysisTrace = document?.analysis_trace || null) {
    if (document) {
      setDocuments((items) => [document, ...items.filter((item) => item.id !== document.id)]);
      setActiveDocumentId(document.id);
    }
    setMessages((items) => [...items, makeMessage("agent", text, document, analysisTrace)]);
  }

  function addSystemMessage(text: string) {
    setMessages((items) => [...items, makeMessage("system", text)]);
  }

  function renderGuardrails(guardrails?: string[]) {
    if (!guardrails?.length) return;
    addSystemMessage(`系统提醒：${guardrails.join("；")}`);
  }

  const activeDocument = useMemo(
    () => documents.find((document) => document.id === activeDocumentId) || documents[0] || null,
    [activeDocumentId, documents]
  );

  return (
    <main className="app-shell">
      <Sidebar
        activePage={activePage}
        busy={busy}
        health={health}
        offline={offline}
        portfolio={portfolio}
        syncVectorStore={syncVectorStore}
        webSearch={webSearch}
        workspace={workspace}
        onChoosePortfolio={handleChoosePortfolio}
        onNewSession={handleCreateSession}
        onOfflineChange={setOffline}
        onPageChange={setActivePage}
        onPrompt={handleSend}
        onRunDailyPicks={handleRunDailyPicks}
        onSyncVectorStoreChange={setSyncVectorStore}
        onWebSearchChange={setWebSearch}
      />

      <section className="workspace">
        <header className="topbar">
          <div>
            <div className="eyebrow">Quant Research</div>
            <h2>{pageTitles[activePage]}</h2>
          </div>
          <div className="topbar-actions">
            <span className="session-pill">{sessionId ? `会话 ${sessionId.slice(0, 8)}` : "未开始"}</span>
            <span className="mode-pill">{modeText}</span>
          </div>
        </header>

        {activePage === "overview" && (
          <DashboardPage
            workspace={workspace}
            onAnalyzePick={analyzePick}
            onOpenOpportunities={() => setActivePage("opportunities")}
            onOpenResearch={() => setActivePage("research")}
            onRefreshDaily={() => void handleRunDailyPicks()}
          />
        )}
        {activePage === "opportunities" && (
          <ReportPage
            report={workspace?.dailyReport || null}
            onAskReport={() =>
              void handleSend("请解读当前桌面端展示的每日报告，输出今日强关注 Top、候选分层、资料核验提醒、触发条件和失效条件。")
            }
            onAnalyzePick={analyzePick}
          />
        )}
        {activePage === "research" && (
          <ResearchPage
            busy={busy}
            messages={messages}
            onOpenDocument={(document) => {
              setActiveDocumentId(document.id);
              setActivePage("archive");
            }}
            typingLabel={typingLabel}
            onStartSession={() => void handleCreateSession()}
            onSend={handleSend}
          />
        )}
        {activePage === "portfolio" && (
          <PortfolioPage
            portfolio={portfolio}
            report={workspace?.dailyReport || null}
            onAnalyze={() =>
              void handleSend("请基于我刚选择的持仓文件和最新 daily-picks 做持仓诊断，重点分析浮盈浮亏、仓位集中、主题暴露、候选池匹配度和加减仓前置条件。")
            }
          />
        )}
        {activePage === "knowledge" && <KnowledgePage workspace={workspace} />}
        {activePage === "archive" && <DocumentPage documents={documents} selected={activeDocument} onSelect={setActiveDocumentId} />}
      </section>
    </main>
  );
}

function makeMessage(
  role: ChatMessage["role"],
  text: string,
  document?: ChatDocument | null,
  analysisTrace = document?.analysis_trace || null
): ChatMessage {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    text,
    createdAt: new Date().toISOString(),
    document,
    analysisTrace
  };
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

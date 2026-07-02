import { BarChart3, BookOpen, BriefcaseBusiness, Database, FileText, MessageSquarePlus, Search } from "lucide-react";
import type { ComponentType } from "react";
import type { HealthPayload, PageId, PortfolioPreview, WorkspaceSnapshot } from "../types";
import { compactPath, formatDateTime } from "../utils/format";

const quickPrompts = [
  {
    label: "今日强关注",
    prompt: "请基于最新 daily-picks 总结今日强关注股票，按评分、上涨触发条件、失效条件、风险标签输出。"
  },
  {
    label: "持仓诊断",
    prompt: "请读取最新量化候选池，并结合我的持仓做组合风险、主题暴露、加减仓前置条件分析。"
  },
  {
    label: "资料核验",
    prompt: "请列出最新 RAG 中所有 #needs_verification 或 #needs_refresh 的关键信息，并说明哪些资料回答时需要提醒我核验。"
  }
];

const navItems: Array<{ id: PageId; label: string; icon: ComponentType<{ size?: number }> }> = [
  { id: "overview", label: "总览", icon: BarChart3 },
  { id: "opportunities", label: "机会池", icon: Search },
  { id: "research", label: "研究台", icon: BookOpen },
  { id: "portfolio", label: "持仓", icon: BriefcaseBusiness },
  { id: "knowledge", label: "知识库", icon: Database },
  { id: "archive", label: "档案", icon: FileText }
];

interface SidebarProps {
  activePage: PageId;
  health: HealthPayload | null;
  workspace: WorkspaceSnapshot | null;
  portfolio: PortfolioPreview | null;
  offline: boolean;
  webSearch: boolean;
  syncVectorStore: boolean;
  busy: boolean;
  onPageChange: (page: PageId) => void;
  onNewSession: () => void;
  onChoosePortfolio: () => void;
  onRunDailyPicks: () => void;
  onPrompt: (prompt: string) => void;
  onOfflineChange: (value: boolean) => void;
  onWebSearchChange: (value: boolean) => void;
  onSyncVectorStoreChange: (value: boolean) => void;
}

export function Sidebar(props: SidebarProps) {
  const statusOk = props.health?.app === "stock-agent" && props.health?.service === "stock-agent-api";
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">SA</div>
        <div>
          <h1>Stock Agent</h1>
          <p>量化推荐工作台</p>
        </div>
      </div>

      <button className="new-chat-btn" disabled={props.busy} onClick={props.onNewSession}>
        <MessageSquarePlus size={17} />
        新建研究
      </button>

      <nav className="page-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={`nav-btn ${props.activePage === item.id ? "active" : ""}`}
              onClick={() => props.onPageChange(item.id)}
            >
              <Icon size={15} />
              {item.label}
            </button>
          );
        })}
      </nav>

      <section className="side-section">
        <div className="section-title">每日工作流</div>
        <div className="workflow-card">
          <button className="secondary-btn" disabled={props.busy} onClick={props.onChoosePortfolio}>
            选择持仓 CSV/JSON
          </button>
          <div className="portfolio-path" title={props.portfolio?.path || ""}>
            {props.portfolio ? compactPath(props.portfolio.path) : "未选择持仓文件"}
          </div>
          <label className="switch-row compact">
            <input
              checked={props.syncVectorStore}
              type="checkbox"
              onChange={(event) => props.onSyncVectorStoreChange(event.currentTarget.checked)}
            />
            <span>同步向量库</span>
          </label>
          <button className="primary-action" disabled={props.busy} onClick={props.onRunDailyPicks}>
            刷新候选池
          </button>
        </div>
      </section>

      <section className="side-section">
        <div className="section-title">运行状态</div>
        <div className="health-card">
          <div className="health-main">
            <span className={`status-dot ${statusOk ? "ok" : "fail"}`} />
            <div>
              <strong>{statusOk ? "已连接" : "API 未启动"}</strong>
              <p>本地服务连接状态</p>
            </div>
          </div>
          <div className="service-list">
            <ServiceLine label="Embedding" value={props.health?.embedding_service_url || "-"} />
            <ServiceLine label="Qdrant" value={props.health?.qdrant_url || "-"} />
            <ServiceLine
              label="向量库"
              value={`${props.workspace?.vectorStore.provider || "-"} / ${
                props.workspace?.vectorStore.collection || "-"
              }`}
            />
            <ServiceLine
              label="日报"
              value={props.workspace?.dailyReport.exists ? formatDateTime(props.workspace.dailyReport.modifiedAt) : "未生成"}
            />
          </div>
        </div>
      </section>

      <section className="side-section">
        <div className="section-title">会话选项</div>
        <label className="switch-row">
          <input
            checked={props.offline}
            type="checkbox"
            onChange={(event) => props.onOfflineChange(event.currentTarget.checked)}
          />
          <span>离线模式</span>
        </label>
        <label className="switch-row">
          <input
            checked={props.webSearch}
            type="checkbox"
            onChange={(event) => props.onWebSearchChange(event.currentTarget.checked)}
          />
          <span>联网搜索</span>
        </label>
      </section>

      <section className="side-section">
        <div className="section-title">快捷研究</div>
        <div className="quick-stack">
          {quickPrompts.map((item) => (
            <button key={item.label} className="quick-btn" disabled={props.busy} onClick={() => props.onPrompt(item.prompt)}>
              {item.label}
            </button>
          ))}
        </div>
      </section>
    </aside>
  );
}

function ServiceLine({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <small>{value}</small>
    </div>
  );
}

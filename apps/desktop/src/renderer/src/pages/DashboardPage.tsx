import { CandidateCard } from "../components/CandidateCard";
import { MetricCard } from "../components/MetricCard";
import type { DailyPick, WorkspaceSnapshot } from "../types";
import { formatDateTime, formatInteger } from "../utils/format";

interface DashboardPageProps {
  workspace: WorkspaceSnapshot | null;
  onAnalyzePick: (pick: DailyPick) => void;
  onOpenOpportunities: () => void;
  onOpenResearch: () => void;
  onRefreshDaily: () => void;
}

export function DashboardPage(props: DashboardPageProps) {
  const report = props.workspace?.dailyReport;
  const vector = props.workspace?.vectorStore;
  const picks = report?.picks || [];
  return (
    <section className="page active" id="page-overview">
      <section className="dashboard">
        <div className="metrics-strip">
          <MetricCard label="日报时间" value={report?.generatedAt || "-"} />
          <MetricCard label="扫描股票" value={formatInteger(report?.scannedCount)} />
          <MetricCard label="候选数" value={formatInteger(report?.topK)} />
          <MetricCard label="RAG Chunks" value={formatInteger(vector?.chunkCount)} />
        </div>
        <div className="workflow-overview">
          <div className="panel-title-row">
            <div>
              <div className="panel-title">每日量化流程</div>
              <p>行情刷新、RAG 同步、机会筛选和持仓诊断拆开执行，结论进入研究台与档案。</p>
            </div>
          </div>
          <div className="process-grid">
            <button className="process-step" onClick={props.onRefreshDaily}>
              <span>01</span>
              <strong>刷新候选池</strong>
              <small>更新行情、报告和向量库</small>
            </button>
            <button className="process-step" onClick={props.onOpenOpportunities}>
              <span>02</span>
              <strong>查看机会池</strong>
              <small>按评分、触发和风险分层</small>
            </button>
            <button className="process-step" onClick={props.onOpenResearch}>
              <span>03</span>
              <strong>进入研究台</strong>
              <small>对话生成分析流程与长文</small>
            </button>
          </div>
        </div>
        <div className="picks-panel">
          <div className="panel-title-row">
            <div>
              <div className="panel-title">今日候选 Top</div>
              <p>
                来自 latest_daily_picks.md，最后更新 {report?.modifiedAt ? formatDateTime(report.modifiedAt) : "-"}。
              </p>
            </div>
            <button className="ghost-btn" onClick={props.onOpenOpportunities}>
              打开机会池
            </button>
          </div>
          <div className="picks-list">
            {picks.length ? (
              picks.slice(0, 6).map((pick, index) => (
                <CandidateCard key={`${pick.code}-${index}`} pick={pick} index={index} onAnalyze={props.onAnalyzePick} />
              ))
            ) : (
              <div className="empty-picks">还没有每日候选。</div>
            )}
          </div>
        </div>
      </section>
    </section>
  );
}

import { useMemo, useState } from "react";
import type { DailyPick, DailyReport } from "../types";
import { formatInteger, renderMarkdown } from "../utils/format";

interface ReportPageProps {
  report: DailyReport | null;
  onAskReport: () => void;
  onAnalyzePick: (pick: DailyPick) => void;
}

export function ReportPage({ report, onAskReport, onAnalyzePick }: ReportPageProps) {
  const picks = report?.picks || [];
  const [selectedCode, setSelectedCode] = useState("");
  const selected = useMemo(
    () => picks.find((pick) => pick.code === selectedCode) || picks[0],
    [picks, selectedCode]
  );
  return (
    <section className="page active" id="page-opportunities">
      <div className="page-grid report-grid">
        <div className="report-sidebar">
          <div className="panel-title-row">
            <div>
              <div className="panel-title">候选列表</div>
              <p>点击任一股票查看规则摘要。</p>
            </div>
          </div>
          <div className="report-pick-list">
            {picks.length ? (
              picks.map((pick, index) => (
                <button
                  key={`${pick.code}-${index}`}
                  className={`report-pick ${selected?.code === pick.code ? "active" : ""}`}
                  onClick={() => setSelectedCode(pick.code)}
                >
                  <span>
                    {index + 1}. {pick.name} <small>{pick.code}</small>
                  </span>
                  <strong>{pick.score || "-"}</strong>
                </button>
              ))
            ) : (
              <div className="empty-picks">暂无候选。</div>
            )}
          </div>
        </div>
        <div className="report-reader">
          <div className="reader-toolbar">
            <div>
              <div className="panel-title">每日量化推荐观察池</div>
              <p>
                {report?.generatedAt || "-"} · 扫描 {formatInteger(report?.scannedCount)} · Top{" "}
                {formatInteger(report?.topK)}
              </p>
            </div>
            <button className="ghost-btn" onClick={onAskReport}>
              让 Agent 解读
            </button>
          </div>
          <ReportDetail pick={selected} onAnalyzePick={onAnalyzePick} />
          <article
            className="markdown-view"
            dangerouslySetInnerHTML={{
              __html: report?.content ? renderMarkdown(report.content) : '<div class="empty-picks">还没有可展示的每日报告。</div>'
            }}
          />
        </div>
      </div>
    </section>
  );
}

function ReportDetail({ pick, onAnalyzePick }: { pick?: DailyPick; onAnalyzePick: (pick: DailyPick) => void }) {
  if (!pick) {
    return <div className="report-detail"><div className="empty-picks">请选择候选股票。</div></div>;
  }
  return (
    <div className="report-detail">
      <div className="detail-card">
        <div className="detail-head">
          <div>
            <h3>
              {pick.name}（{pick.code}）
            </h3>
            <p>
              {pick.rating || "-"} · {pick.pctChange || "-"} · {pick.turnover || "-"}
            </p>
          </div>
          <strong>{pick.score || "-"}</strong>
        </div>
        <div className="detail-actions">
          <button className="ghost-btn" onClick={() => onAnalyzePick(pick)}>
            生成逐股分析
          </button>
        </div>
        <dl className="detail-list compact-list">
          <div><dt>最新价</dt><dd>{pick.price || "-"}</dd></div>
          <div><dt>成交额</dt><dd>{pick.amount || "-"}</dd></div>
          <div><dt>推荐逻辑</dt><dd>{pick.logic || "-"}</dd></div>
          <div><dt>风险提示</dt><dd>{pick.risk || "需核验公告和行情"}</dd></div>
        </dl>
      </div>
    </div>
  );
}

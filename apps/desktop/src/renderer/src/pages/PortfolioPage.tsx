import type { DailyReport, PortfolioPreview } from "../types";
import { compactPath, renderMarkdown } from "../utils/format";

interface PortfolioPageProps {
  portfolio: PortfolioPreview | null;
  report: DailyReport | null;
  onAnalyze: () => void;
}

export function PortfolioPage({ portfolio, report, onAnalyze }: PortfolioPageProps) {
  return (
    <section className="page active" id="page-portfolio">
      <div className="page-grid portfolio-grid">
        <div className="info-panel">
          <div className="panel-title">持仓文件</div>
          <p className="muted-copy">
            {portfolio ? `${compactPath(portfolio.path)} · ${portfolio.totalRows} 条持仓记录` : "未选择持仓文件。"}
          </p>
          <button className="primary-action" onClick={onAnalyze}>
            用当前持仓做诊断
          </button>
        </div>
        <div className="info-panel">
          <div className="panel-title">持仓预览</div>
          <PortfolioTable portfolio={portfolio} />
        </div>
        <div className="info-panel portfolio-report-panel">
          <div className="panel-title">报告中的持仓分析</div>
          <div
            className="markdown-view compact-markdown"
            dangerouslySetInnerHTML={{
              __html: report?.portfolioSection
                ? renderMarkdown(report.portfolioSection)
                : '<div class="empty-picks">日报中还没有持仓分析。选择持仓文件并刷新候选池后会出现在这里。</div>'
            }}
          />
        </div>
      </div>
    </section>
  );
}

function PortfolioTable({ portfolio }: { portfolio: PortfolioPreview | null }) {
  if (!portfolio) {
    return <div className="portfolio-table"><div className="empty-picks">选择 CSV/JSON 后在这里预览。</div></div>;
  }
  if (!portfolio.rows.length) {
    return <div className="portfolio-table"><div className="empty-picks">文件中没有可识别的持仓行。</div></div>;
  }
  return (
    <div className="portfolio-table">
      <table>
        <thead>
          <tr><th>代码</th><th>名称</th><th>数量</th><th>成本价</th><th>名义市值</th><th>备注</th></tr>
        </thead>
        <tbody>
          {portfolio.rows.map((row, index) => (
            <tr key={`${row.code}-${index}`}>
              <td>{row.code || "-"}</td>
              <td>{row.name || "-"}</td>
              <td>{row.shares || "-"}</td>
              <td>{row.costPrice || "-"}</td>
              <td>{row.marketValue || "-"}</td>
              <td>{row.notes || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

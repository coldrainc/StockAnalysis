import type { WorkspaceSnapshot } from "../types";
import { formatInteger } from "../utils/format";

interface KnowledgePageProps {
  workspace: WorkspaceSnapshot | null;
}

export function KnowledgePage({ workspace }: KnowledgePageProps) {
  const vector = workspace?.vectorStore;
  const tags = workspace?.dailyReport.tags?.length
    ? workspace.dailyReport.tags
    : ["#quant_candidate", "#third_party_dataset", "#needs_verification", "#needs_refresh"];
  return (
    <section className="page active" id="page-knowledge">
      <div className="page-grid knowledge-grid">
        <div className="info-panel">
          <div className="panel-title">向量库状态</div>
          <dl className="detail-list">
            <div><dt>后端</dt><dd>{vector?.provider || "-"}</dd></div>
            <div><dt>Collection</dt><dd>{vector?.collection || "-"}</dd></div>
            <div><dt>市场</dt><dd>{vector?.market || "-"}</dd></div>
            <div><dt>Embedding</dt><dd>{vector ? `${vector.embeddingProvider} / ${vector.embeddingModel}` : "-"}</dd></div>
            <div><dt>Chunks</dt><dd>{formatInteger(vector?.chunkCount)}</dd></div>
            <div><dt>RAG 源</dt><dd>{vector?.sourceDir || "-"}</dd></div>
          </dl>
        </div>
        <div className="info-panel">
          <div className="panel-title">资料核验标签</div>
          <div className="tag-list">
            {tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
          <div className="notice-block">
            回答推荐时，凡引用 #needs_verification、#needs_refresh 或 #third_party_dataset，都必须提醒核验公告原文、财报和最新行情。
          </div>
        </div>
      </div>
    </section>
  );
}

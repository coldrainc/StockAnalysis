import type { ChatDocument } from "../types";
import { formatDateTime, renderMarkdown } from "../utils/format";

interface DocumentPageProps {
  documents: ChatDocument[];
  selected: ChatDocument | null;
  onSelect: (documentId: string) => void;
}

export function DocumentPage({ documents, selected, onSelect }: DocumentPageProps) {
  return (
    <section className="page active" id="page-archive">
      <div className="page-grid document-grid">
        <aside className="document-list">
          <div className="panel-title-row">
            <div>
              <div className="panel-title">研究文档</div>
              <p>长回答会自动保存到这里。</p>
            </div>
          </div>
          <div className="report-pick-list">
            {documents.length ? (
              documents.map((document) => (
                <button
                  key={document.id}
                  className={`report-pick ${selected?.id === document.id ? "active" : ""}`}
                  onClick={() => onSelect(document.id)}
                >
                  <span>{document.title}</span>
                  <strong>{formatDateTime(document.created_at)}</strong>
                </button>
              ))
            ) : (
              <div className="empty-picks">暂无研究文档。</div>
            )}
          </div>
        </aside>
        <article className="document-reader">
          {selected ? (
            <>
              <div className="reader-toolbar">
                <div>
                  <div className="panel-title">{selected.title}</div>
                  <p>{selected.path}</p>
                </div>
              </div>
              {selected.analysis_trace && (
                <div className="document-trace">
                  <div className="panel-title">分析流程</div>
                  <p>{selected.analysis_trace.summary}</p>
                  <ol>
                    {selected.analysis_trace.steps.map((step, index) => (
                      <li key={`${step.title}-${index}`}>
                        <strong>{step.title}</strong>
                        {step.detail && <span>{step.detail}</span>}
                      </li>
                    ))}
                  </ol>
                </div>
              )}
              <div
                className="markdown-view"
                dangerouslySetInnerHTML={{
                  __html: renderMarkdown(selected.content)
                }}
              />
            </>
          ) : (
            <div className="empty-picks">让 Agent 生成一份较长分析后，会自动出现在这里。</div>
          )}
        </article>
      </div>
    </section>
  );
}

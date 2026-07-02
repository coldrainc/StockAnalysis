import { FormEvent, useEffect, useRef, useState } from "react";
import { ChevronDown, FileText, Workflow } from "lucide-react";
import type { AnalysisTrace, ChatDocument, ChatMessage } from "../types";
import { formatMessageTime } from "../utils/format";

const promptChips = [
  {
    label: "分析 Top5",
    prompt: "请分析最新日报里评分最高的 5 只股票，按量价、估值、催化剂、风险和观察触发条件输出。"
  },
  {
    label: "候选分层",
    prompt: "请帮我把最新候选池分成强关注、观察、暂不优先，并说明每类的仓位纪律。"
  },
  {
    label: "核验提醒",
    prompt: "请说明当前知识库里哪些信息来自第三方数据，哪些需要我核验公告原文。"
  }
];

interface ChatPanelProps {
  messages: ChatMessage[];
  busy: boolean;
  typingLabel: string;
  onOpenDocument: (document: ChatDocument) => void;
  onStart: () => void;
  onSend: (text: string) => void;
}

export function ChatPanel({ messages, busy, typingLabel, onOpenDocument, onStart, onSend }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const messagesRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages, busy]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    onSend(text);
  };

  return (
    <div className="chat-surface">
      <div className="messages" ref={messagesRef}>
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">SA</div>
            <h3>准备开始量化研究</h3>
            <p>创建新研究后，Agent 会结合每日候选池、逐股文档、向量库和你的持仓约束分析机会与风险。</p>
            <button className="empty-action" disabled={busy} onClick={onStart}>
              开始研究
            </button>
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} message={message} onOpenDocument={onOpenDocument} />
          ))
        )}
        {busy && (
          <div className="typing-stack">
            <div className="typing" data-typing="true">
              {typingLabel}
            </div>
            <div className="live-process">
              <span>读取上下文</span>
              <span>检索 RAG</span>
              <span>核验标签</span>
              <span>组织结论</span>
            </div>
          </div>
        )}
      </div>

      <div className="composer-tools">
        {promptChips.map((item) => (
          <button key={item.label} className="prompt-chip" disabled={busy} onClick={() => onSend(item.prompt)}>
            {item.label}
          </button>
        ))}
      </div>

      <form className="composer" onSubmit={submit}>
        <textarea
          rows={1}
          value={input}
          placeholder="输入股票代码、公司名、持仓约束或想看的机会方向，按 Enter 发送"
          onChange={(event) => setInput(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              submit(event);
            }
          }}
        />
        <button type="submit" disabled={busy || !input.trim()} aria-label="发送">
          <span>发送</span>
        </button>
      </form>
    </div>
  );
}

function MessageBubble({
  message,
  onOpenDocument
}: {
  message: ChatMessage;
  onOpenDocument: (document: ChatDocument) => void;
}) {
  return (
    <div className={`message-row ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "你" : message.role === "system" ? "!" : "AI"}</div>
      <div className="bubble-stack">
        <div className="message-meta">
          {roleLabel(message.role)} · {formatMessageTime(message.createdAt)}
        </div>
        <div className={`message ${message.role}`}>
          {message.text}
          {message.analysisTrace && <AnalysisTracePanel trace={message.analysisTrace} />}
          {message.document && (
            <button className="document-link-btn" onClick={() => onOpenDocument(message.document as ChatDocument)}>
              <FileText size={15} />
              打开完整文档
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function AnalysisTracePanel({ trace }: { trace: AnalysisTrace }) {
  const [open, setOpen] = useState(true);
  if (!trace.steps.length) return null;
  return (
    <div className={`analysis-trace ${open ? "open" : ""}`}>
      <button className="analysis-trace-toggle" onClick={() => setOpen((value) => !value)}>
        <Workflow size={15} />
        <span>分析流程</span>
        <ChevronDown size={15} />
      </button>
      {open && (
        <div className="analysis-trace-body">
          <p>{trace.summary}</p>
          <ol>
            {trace.steps.map((step, index) => (
              <li key={`${step.title}-${index}`}>
                <strong>{step.title}</strong>
                {step.detail && <span>{step.detail}</span>}
              </li>
            ))}
          </ol>
          {trace.warnings.length > 0 && (
            <div className="trace-warnings">
              {trace.warnings.map((warning) => (
                <span key={warning}>{warning}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function roleLabel(role: ChatMessage["role"]) {
  if (role === "user") return "用户";
  if (role === "system") return "系统";
  return "分析师";
}

import { ChatPanel } from "../components/ChatPanel";
import type { ChatDocument, ChatMessage } from "../types";

interface ResearchPageProps {
  messages: ChatMessage[];
  busy: boolean;
  typingLabel: string;
  onOpenDocument: (document: ChatDocument) => void;
  onStartSession: () => void;
  onSend: (text: string) => void;
}

export function ResearchPage(props: ResearchPageProps) {
  return (
    <section className="page active research-page" id="page-research">
      <ChatPanel
        busy={props.busy}
        messages={props.messages}
        onOpenDocument={props.onOpenDocument}
        typingLabel={props.typingLabel}
        onStart={props.onStartSession}
        onSend={props.onSend}
      />
    </section>
  );
}

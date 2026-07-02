export type PageId = "overview" | "opportunities" | "research" | "portfolio" | "knowledge" | "archive";

export type ChatRole = "agent" | "user" | "system";

export interface AnalysisTraceStep {
  title: string;
  detail: string;
  status: string;
}

export interface AnalysisTrace {
  summary: string;
  steps: AnalysisTraceStep[];
  warnings: string[];
  source: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  createdAt: string;
  document?: ChatDocument | null;
  analysisTrace?: AnalysisTrace | null;
}

export interface ChatDocument {
  id: string;
  title: string;
  path: string;
  content: string;
  created_at: string;
  analysis_trace?: AnalysisTrace | null;
}

export interface HealthPayload {
  status: string;
  app: string;
  service: string;
  domain: string;
  capabilities?: string[];
  default_api_url?: string;
  qdrant_url?: string;
  embedding_service_url?: string;
}

export interface ChatResponse {
  session_id: string;
  message: string;
  completed: boolean;
  fallback_used: boolean;
  guardrails: string[];
  document?: ChatDocument | null;
  analysis_trace?: AnalysisTrace | null;
}

export interface DailyPick {
  name: string;
  code: string;
  score: string;
  rating: string;
  price: string;
  pctChange: string;
  amount: string;
  turnover: string;
  logic: string;
  risk: string;
}

export interface DailyReport {
  exists: boolean;
  path: string;
  modifiedAt?: string;
  content: string;
  generatedAt: string;
  scannedCount: number;
  topK: number;
  tags: string[];
  picks: DailyPick[];
  portfolioSection?: string;
}

export interface VectorStoreStatus {
  provider: string;
  collection: string;
  market: string;
  sourceDir: string;
  embeddingProvider: string;
  embeddingModel: string;
  chunkCount: number;
}

export interface WorkspaceSnapshot {
  projectRoot: string;
  defaultPortfolio?: PortfolioPreview | null;
  dailyReport: DailyReport;
  vectorStore: VectorStoreStatus;
}

export interface PortfolioRow {
  code: string;
  name: string;
  shares: string;
  costPrice: string;
  marketValue: string;
  notes: string;
}

export interface PortfolioPreview {
  path: string;
  rows: PortfolioRow[];
  totalRows: number;
}

export interface DailyPicksResult {
  output: string;
  workspace: WorkspaceSnapshot;
}

export interface StockAgentBridge {
  health: () => Promise<HealthPayload>;
  createSession: (payload: { offline: boolean; web_search: boolean }) => Promise<ChatResponse>;
  sendMessage: (payload: { sessionId: string; message: string }) => Promise<ChatResponse>;
  workspace: () => Promise<WorkspaceSnapshot>;
  choosePortfolio: () => Promise<PortfolioPreview | null>;
  runDailyPicks: (payload: {
    portfolio?: string;
    maxCandidates: number;
    topK: number;
    syncVectorStore: boolean;
  }) => Promise<DailyPicksResult>;
}

declare global {
  interface Window {
    stockAgent: StockAgentBridge;
  }
}

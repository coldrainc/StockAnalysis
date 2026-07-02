import type { DailyPicksResult, HealthPayload, PortfolioPreview, WorkspaceSnapshot } from "../types";

export function getHealth(): Promise<HealthPayload> {
  return window.stockAgent.health();
}

export function getWorkspace(): Promise<WorkspaceSnapshot> {
  return window.stockAgent.workspace();
}

export function choosePortfolio(): Promise<PortfolioPreview | null> {
  return window.stockAgent.choosePortfolio();
}

export function runDailyPicks(payload: {
  portfolio?: string;
  maxCandidates: number;
  topK: number;
  syncVectorStore: boolean;
}): Promise<DailyPicksResult> {
  return window.stockAgent.runDailyPicks(payload);
}

export function createSession(payload: { offline: boolean; webSearch: boolean }) {
  return window.stockAgent.createSession({
    offline: payload.offline,
    web_search: payload.webSearch
  });
}

export function sendMessage(payload: { sessionId: string; message: string }) {
  return window.stockAgent.sendMessage(payload);
}

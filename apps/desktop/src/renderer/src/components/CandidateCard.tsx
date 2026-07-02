import type { DailyPick } from "../types";

interface CandidateCardProps {
  pick: DailyPick;
  index: number;
  onAnalyze: (pick: DailyPick) => void;
}

export function CandidateCard({ pick, index, onAnalyze }: CandidateCardProps) {
  return (
    <button className="pick-card" title={`${pick.name} ${pick.code}`} onClick={() => onAnalyze(pick)}>
      <div className="pick-head">
        <span>
          {index + 1}. {pick.name}
        </span>
        <strong>{pick.score || "-"}</strong>
      </div>
      <div className="pick-meta">
        {pick.code} · {pick.rating || "-"} · {pick.pctChange || "-"}
      </div>
      <div className="pick-quote">
        <span>{pick.price || "-"}</span>
        <span>{pick.amount || "-"}</span>
      </div>
      <div className="pick-risk">{pick.risk || "需核验公告和行情"}</div>
    </button>
  );
}

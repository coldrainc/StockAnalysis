from __future__ import annotations

from pydantic import BaseModel, Field

from stock_agent.core.config import AnalysisStage


class AnalysisTurn(BaseModel):
    stage: AnalysisStage
    analyst: str
    user: str | None = None

    @property
    def interviewer(self) -> str:
        return self.analyst

    @property
    def candidate(self) -> str | None:
        return self.user

    @candidate.setter
    def candidate(self, value: str | None) -> None:
        self.user = value


class AnalysisState(BaseModel):
    stage: AnalysisStage = AnalysisStage.INTAKE
    turns: list[AnalysisTurn] = Field(default_factory=list)
    current_focus_index: int = 0
    completed: bool = False

    def transcript(self) -> str:
        lines: list[str] = []
        for index, turn in enumerate(self.turns, start=1):
            lines.append(f"Turn {index} [{turn.stage.value}]")
            lines.append(f"分析师：{turn.analyst}")
            if turn.user:
                lines.append(f"用户：{turn.user}")
        return "\n".join(lines).strip()

    def add_analyst_message(self, stage: AnalysisStage, content: str) -> None:
        self.turns.append(AnalysisTurn(stage=stage, analyst=content))

    def add_user_message(self, content: str) -> None:
        if not self.turns:
            raise ValueError("Cannot add a user message before an analyst turn.")
        self.turns[-1].user = content

    def add_interviewer_message(self, stage: AnalysisStage, content: str) -> None:
        self.add_analyst_message(stage, content)

    def add_candidate_message(self, content: str) -> None:
        self.add_user_message(content)


InterviewTurn = AnalysisTurn
InterviewState = AnalysisState

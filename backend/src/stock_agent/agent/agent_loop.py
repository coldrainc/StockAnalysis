from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from stock_agent.core.config import AnalysisStage, StockConfig
from stock_agent.agent.guardrails import GuardrailFinding
from stock_agent.agent.state import AnalysisState

if TYPE_CHECKING:
    from stock_agent.agent.harness import StockHarness


@dataclass
class LoopResult:
    state: AnalysisState
    message: str
    advanced: bool = True
    guardrail_findings: list[GuardrailFinding] | None = None
    fallback_used: bool = False


class InputKind(str, Enum):
    REQUEST = "request"
    CLARIFYING_QUESTION = "clarifying_question"
    TOO_SHORT = "too_short"


class AgentLoop:
    """Explicit stock research control loop around the LangChain harness."""

    def __init__(self, config: StockConfig, harness: "StockHarness") -> None:
        self.config = config
        self.harness = harness
        self.state = AnalysisState()

    def start(self) -> LoopResult:
        harness_result = self.harness.generate_result(AnalysisStage.INTAKE, self.state)
        message = harness_result.text
        self.state.stage = AnalysisStage.INTAKE
        self.state.add_analyst_message(AnalysisStage.INTAKE, message)
        return LoopResult(
            state=self.state,
            message=message,
            guardrail_findings=harness_result.findings,
            fallback_used=harness_result.fallback_used,
        )

    def step(self, user_message: str) -> LoopResult:
        return self.handle_input(user_message)

    def handle_input(self, user_input: str) -> LoopResult:
        if self.state.completed:
            return LoopResult(self.state, "本轮股票分析已经完成。", advanced=False)
        if not self.state.turns:
            self.start()

        input_check = self.harness.guardrails.check_candidate_input(user_input)
        if input_check.blocked:
            message = self.harness.guardrails.blocked_message(input_check.findings)
            return LoopResult(
                self.state,
                message,
                advanced=False,
                guardrail_findings=input_check.findings,
            )

        cleaned_input = input_check.text
        input_kind = self._classify_input(cleaned_input)
        if input_kind == InputKind.CLARIFYING_QUESTION:
            harness_result = self.harness.respond_to_candidate_question_result(
                cleaned_input, self.state
            )
            return LoopResult(
                self.state,
                harness_result.text,
                advanced=False,
                guardrail_findings=input_check.findings + harness_result.findings,
                fallback_used=harness_result.fallback_used,
            )
        if input_kind == InputKind.TOO_SHORT:
            active_question = self.state.turns[-1].analyst
            message = (
                "这条信息还不足以形成研究结论。请补充股票代码/公司名、市场、持有周期、"
                "风险偏好或你关注的问题。\n\n"
                f"当前需要确认：{active_question}"
            )
            return LoopResult(
                self.state,
                message,
                advanced=False,
                guardrail_findings=input_check.findings,
            )

        self.state.add_user_message(cleaned_input)
        next_stage = self._next_stage()
        harness_result = self.harness.generate_result(next_stage, self.state)
        message = harness_result.text
        self.state.stage = next_stage
        self.state.add_analyst_message(next_stage, message)

        if next_stage == AnalysisStage.RECOMMENDATION:
            self.state.completed = True

        return LoopResult(
            state=self.state,
            message=message,
            guardrail_findings=input_check.findings + harness_result.findings,
            fallback_used=harness_result.fallback_used,
        )

    def _classify_input(self, user_input: str) -> InputKind:
        normalized = user_input.strip().lower()
        if not normalized:
            return InputKind.TOO_SHORT

        question_markers = ("?", "？", "什么是", "是什么意思", "怎么理解", "能解释", "可以解释")
        if any(marker in normalized for marker in question_markers):
            return InputKind.CLARIFYING_QUESTION

        ticker_like = bool(re_search_ticker(normalized))
        if not ticker_like and len(normalized) < 6:
            return InputKind.TOO_SHORT
        return InputKind.REQUEST

    def _next_stage(self) -> AnalysisStage:
        answered_turns = sum(1 for turn in self.state.turns if turn.candidate)
        if answered_turns >= self.config.max_turns:
            return AnalysisStage.RECOMMENDATION

        if self.state.stage == AnalysisStage.INTAKE:
            return AnalysisStage.FUNDAMENTAL
        if self.state.stage == AnalysisStage.FUNDAMENTAL:
            return AnalysisStage.CATALYST
        if self.state.stage == AnalysisStage.CATALYST:
            return AnalysisStage.RISK
        if self.state.stage == AnalysisStage.RISK:
            return AnalysisStage.RECOMMENDATION

        self.state.current_focus_index += 1
        if self.state.current_focus_index >= len(self.config.focus_areas):
            return AnalysisStage.RECOMMENDATION
        return AnalysisStage.FUNDAMENTAL


def re_search_ticker(text: str):
    import re

    return re.search(r"\b([a-z]{1,5}|[0-9]{4,6})(\.[a-z]{1,4})?\b", text)

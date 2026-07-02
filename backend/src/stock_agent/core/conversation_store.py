from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from stock_agent.core.config import StockConfig
from stock_agent.agent.state import AnalysisState, AnalysisTurn


MIN_MEMORY_CHARS = 28
RESEARCH_SIGNALS = (
    "股票",
    "估值",
    "财报",
    "营收",
    "利润",
    "现金流",
    "分红",
    "回购",
    "订单",
    "产能",
    "价格",
    "周期",
    "政策",
    "风险",
    "仓位",
    "止损",
    "行业",
    "科技",
    "材料",
    "贵金属",
    "能源",
    "锂电",
    "银行",
    "PE",
    "PB",
    "ROE",
    "EPS",
    "A股",
    "港股",
    "美股",
)
LOW_VALUE_PATTERNS = (
    "不知道",
    "不会",
    "不清楚",
    "没想过",
    "随便",
    "好的",
    "明白",
    "谢谢",
    "嗯",
    "啊",
    "可以",
    "继续",
    "下一题",
)


@dataclass
class ConversationStore:
    root: Path = Path(".stock_agent/conversations")
    memory_root: Path = Path(".stock_agent/memory")

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = timestamp
        self.jsonl_path = self.root / f"{self.session_id}.jsonl"
        self.markdown_path = self.root / f"{self.session_id}.md"
        self.memory_root.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.memory_root / f"{self.session_id}.md"

    def record_event(self, event_type: str, payload: dict) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": payload,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def save_state(self, config: StockConfig, state: AnalysisState) -> None:
        content = render_transcript_markdown(config, state)
        self.markdown_path.write_text(content, encoding="utf-8")
        self.memory_path.write_text(render_memory_markdown(config, state), encoding="utf-8")


def render_transcript_markdown(config: StockConfig, state: AnalysisState) -> str:
    lines = [
        f"# 股票研究会话 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 用户：{config.investor.name}",
        f"- 关注市场：{config.investor.market}",
        f"- 风险偏好：{config.investor.risk_appetite}",
        f"- 投资周期：{config.investor.investment_horizon}",
        f"- 当前阶段：{state.stage.value}",
        f"- 是否完成：{state.completed}",
        "",
        "## Transcript",
        "",
    ]
    for index, turn in enumerate(state.turns, start=1):
        lines.extend(_render_turn(index, turn))
    return "\n".join(lines).strip() + "\n"


def render_memory_markdown(config: StockConfig, state: AnalysisState) -> str:
    lines = [
        "# 历史股票研究可复用知识",
        "",
        f"- 用户：{config.investor.name}",
        f"- 关注市场：{config.investor.market}",
        "",
    ]
    for index, turn in enumerate(state.turns, start=1):
        decision = memory_decision(turn)
        if not decision.keep:
            continue
        lines.extend(
            [
                f"## Q{index}: {turn.stage.value}",
                "",
                f"分析师输出：{turn.analyst}",
                "",
                f"用户输入：{turn.user}",
                "",
                "可检索要点：",
                f"- 主题：{turn.stage.value}",
                f"- 市场：{config.investor.market}",
                "- 类型：历史股票研究对话",
                f"- 保留原因：{decision.reason}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True)
class MemoryDecision:
    keep: bool
    reason: str


def memory_decision(turn: AnalysisTurn) -> MemoryDecision:
    answer = (turn.user or "").strip()
    if not answer:
        return MemoryDecision(False, "没有用户输入")

    normalized = re.sub(r"\s+", "", answer.lower())
    if len(normalized) < MIN_MEMORY_CHARS:
        return MemoryDecision(False, "回答过短")
    if _looks_like_question(answer):
        return MemoryDecision(False, "用户是在提问或澄清，不作为知识沉淀")
    if _is_low_value(answer):
        return MemoryDecision(False, "回答缺少可复用信息")
    if _is_mostly_repetition(answer):
        return MemoryDecision(False, "回答重复度过高")
    if _research_signal_count(answer) == 0 and len(normalized) < 80:
        return MemoryDecision(False, "缺少股票研究信号且内容较短")
    return MemoryDecision(True, "包含可复用的股票研究信息")


def _looks_like_question(answer: str) -> bool:
    question_markers = ("?", "？", "什么是", "是什么意思", "怎么理解", "能解释", "可以解释")
    return any(marker in answer for marker in question_markers)


def _is_low_value(answer: str) -> bool:
    normalized = re.sub(r"\s+", "", answer.lower())
    return any(pattern.lower() == normalized for pattern in LOW_VALUE_PATTERNS) or any(
        pattern.lower() in normalized for pattern in LOW_VALUE_PATTERNS
    ) and len(normalized) < 50


def _is_mostly_repetition(answer: str) -> bool:
    tokens = re.findall(r"[\u4e00-\u9fff]{1}|[a-zA-Z0-9_+-]+", answer.lower())
    if len(tokens) < 12:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio < 0.25


def _research_signal_count(answer: str) -> int:
    lowered = answer.lower()
    return sum(1 for signal in RESEARCH_SIGNALS if signal.lower() in lowered)


def _render_turn(index: int, turn: AnalysisTurn) -> list[str]:
    lines = [
        f"### Turn {index} [{turn.stage.value}]",
        "",
        f"分析师：{turn.analyst}",
        "",
    ]
    if turn.user:
        lines.extend([f"用户：{turn.user}", ""])
    return lines


InterviewConfig = StockConfig
InterviewState = AnalysisState
InterviewTurn = AnalysisTurn

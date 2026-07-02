from pathlib import Path

from stock_agent.core.config import AnalysisStage, StockConfig
from stock_agent.core.conversation_store import ConversationStore, memory_decision
from stock_agent.agent.state import AnalysisState


def test_conversation_store_saves_transcript_and_memory(tmp_path: Path) -> None:
    store = ConversationStore(
        root=tmp_path / "conversations",
        memory_root=tmp_path / "memory",
    )
    config = StockConfig()
    state = AnalysisState(stage=AnalysisStage.FUNDAMENTAL)
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供股票代码或行业方向。")
    state.add_user_message("请分析宁德时代，重点看锂电池周期、现金流、估值和风险。")

    store.save_state(config, state)

    assert store.markdown_path.exists()
    assert store.memory_path.exists()
    assert "请提供股票代码" in store.markdown_path.read_text(encoding="utf-8")
    assert "历史股票研究" in store.memory_path.read_text(encoding="utf-8")


def test_memory_filters_low_value_turns() -> None:
    state = AnalysisState(stage=AnalysisStage.FUNDAMENTAL)
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供股票代码。")
    state.add_user_message("不知道")

    decision = memory_decision(state.turns[0])

    assert decision.keep is False


def test_memory_filters_clarifying_questions() -> None:
    state = AnalysisState(stage=AnalysisStage.FUNDAMENTAL)
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供股票代码。")
    state.add_user_message("什么是 PB？可以解释一下吗？")

    decision = memory_decision(state.turns[0])

    assert decision.keep is False


def test_memory_keeps_substantive_stock_research_input() -> None:
    state = AnalysisState(stage=AnalysisStage.FUNDAMENTAL)
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供股票代码。")
    state.add_user_message(
        "请分析招商银行，重点看净息差、资产质量、ROE、分红率、估值 PB 和房地产风险，"
        "投资周期一年，风险偏好中等。"
    )

    decision = memory_decision(state.turns[0])

    assert decision.keep is True

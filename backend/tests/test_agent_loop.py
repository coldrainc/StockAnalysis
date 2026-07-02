from stock_agent.agent.agent_loop import AgentLoop
from stock_agent.core.config import AnalysisStage, InvestorProfile, StockConfig
from stock_agent.agent.harness import ScriptedStockHarness


def test_agent_loop_reaches_recommendation_after_research_stages() -> None:
    config = StockConfig(
        investor=InvestorProfile(name="Sam"),
        focus_areas=["科技", "银行"],
        max_turns=10,
    )
    loop = AgentLoop(config, ScriptedStockHarness(config))

    first = loop.start()
    assert first.state.stage == AnalysisStage.INTAKE
    assert "Sam" in first.message
    assert "股票代码" in first.message

    second = loop.step("请分析英伟达，周期六个月，风险偏好中等。")
    assert second.state.stage == AnalysisStage.FUNDAMENTAL

    third = loop.step("我关注 AI 数据中心需求和估值压力。")
    assert third.state.stage == AnalysisStage.CATALYST

    fourth = loop.step("希望看主要风险和观察指标。")
    assert fourth.state.stage == AnalysisStage.RISK

    final = loop.step("可以给出结论和仓位原则。")
    assert final.state.stage == AnalysisStage.RECOMMENDATION
    assert final.state.completed is True


def test_agent_loop_respects_max_turns() -> None:
    config = StockConfig(focus_areas=["科技", "银行"], max_turns=1)
    loop = AgentLoop(config, ScriptedStockHarness(config))
    loop.start()

    result = loop.step("请分析招商银行，投资周期一年，风险偏好中等。")

    assert result.state.stage == AnalysisStage.RECOMMENDATION
    assert result.state.completed is True


def test_agent_loop_answers_clarifying_question_without_advancing() -> None:
    config = StockConfig(focus_areas=["锂电池"], max_turns=3)
    loop = AgentLoop(config, ScriptedStockHarness(config))
    loop.start()

    result = loop.handle_input("什么是锂电池产业链？")

    assert result.advanced is False
    assert result.state.stage == AnalysisStage.INTAKE
    assert len(result.state.turns) == 1
    assert result.state.turns[0].user is None


def test_agent_loop_asks_for_more_detail_on_short_input() -> None:
    config = StockConfig(focus_areas=["能源"], max_turns=3)
    loop = AgentLoop(config, ScriptedStockHarness(config))
    loop.start()

    result = loop.handle_input("不知道")

    assert result.advanced is False
    assert "补充股票代码" in result.message
    assert len(result.state.turns) == 1

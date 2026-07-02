from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from stock_agent.agent.harness import LangChainStockHarness
from stock_agent.agent.multi_agent import AgentRole, AgentTask, MultiAgentOrchestrator
from stock_agent.agent.state import AnalysisState
from stock_agent.core.config import AnalysisStage, StockConfig


def test_multi_agent_report_includes_portfolio_and_uncertain_tags() -> None:
    config = StockConfig(focus_areas=["科技"])
    state = AnalysisState()
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供股票或持仓。")
    state.add_user_message("我的持仓是示例科技股A和示例科技股B，各为样例仓位。")
    orchestrator = MultiAgentOrchestrator(config)

    report = orchestrator.run(
        stage=AnalysisStage.RECOMMENDATION,
        state=state,
        use_case="portfolio_review",
        query="持仓分析 示例科技股A 示例科技股B",
        knowledge_context="示例科技股A逐股文档 #needs_verification #third_party_dataset",
        web_context="未启用联网搜索。",
    )

    roles = [finding.role for finding in report.findings]
    assert AgentRole.DATA in roles
    assert AgentRole.PORTFOLIO in roles
    assert AgentRole.RISK in roles
    assert AgentRole.SUPERVISOR in roles
    assert "#needs_verification" in report.to_prompt_context()
    assert report.warnings


def test_multi_agent_can_use_specialist_runner() -> None:
    config = StockConfig(focus_areas=["科技"])
    state = AnalysisState()
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供股票。")
    state.add_user_message("请分析 000001 示例科技股A。")
    seen_roles: list[AgentRole] = []

    def runner(task: AgentTask) -> str:
        seen_roles.append(task.spec.role)
        return f"{task.spec.name} 已完成公开 finding，证据来自 RAG 和行情。"

    report = MultiAgentOrchestrator(config).run(
        stage=AnalysisStage.FUNDAMENTAL,
        state=state,
        use_case="stock_deep_dive",
        query="000001 示例科技股A",
        knowledge_context="逐股文档 #official_or_exchange",
        web_context="行情上下文",
        specialist_runner=runner,
    )

    assert seen_roles == [finding.role for finding in report.findings]
    assert any(finding.model_backed for finding in report.findings)
    assert "LLM specialist" in report.to_prompt_context()


class MultiCallLLM:
    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        if "专业角色出发" in messages[-1].content:
            return AIMessage(content="关键发现：专业 Agent 已检查公开资料、行情和风险标签。")
        return AIMessage(
            content=(
                "## 分析流程\n"
                "1. 多Agent分工：DataAgent、QuantAgent、RiskAgent 和 SupervisorAgent 参与。\n\n"
                "## 结论\n"
                "进入观察池。仅供研究辅助，不构成投资建议。"
            )
        )


def test_langchain_harness_runs_specialist_agents_before_final_synthesis() -> None:
    llm = MultiCallLLM()
    config = StockConfig(focus_areas=["科技"])
    state = AnalysisState()
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供股票。")
    state.add_user_message("请分析 000001 示例科技股A，并看是否适合我的持仓。")
    harness = LangChainStockHarness(config, llm=llm)  # type: ignore[arg-type]

    result = harness.generate_result(AnalysisStage.FUNDAMENTAL, state)

    assert "进入观察池" in result.text
    assert len(llm.calls) > 1
    final_user_prompt = llm.calls[-1][-1].content
    assert "## 多Agent编排报告" in final_user_prompt
    assert "DataAgent" in final_user_prompt
    assert "SupervisorAgent" in final_user_prompt

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from stock_agent.core.config import AnalysisStage, StockConfig
from stock_agent.agent.harness import LangChainStockHarness
from stock_agent.agent.prompt_orchestrator import PromptUseCase, StockPromptOrchestrator
from stock_agent.agent.state import AnalysisState


class FakeContextProvider:
    def knowledge_context(self, query: str) -> str:
        return f"RAG_CONTEXT query={query} #needs_verification"

    def web_context(self, query: str) -> str:
        return f"WEB_CONTEXT query={query}"


class CapturingLLM:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.messages = messages
        return AIMessage(
            content=(
                "## 分析流程\n"
                "1. 识别目标：读取用户输入。\n"
                "2. 检索资料：使用 RAG。\n\n"
                "## 结论\n"
                "进入观察池。仅供研究辅助，不构成投资建议。"
            )
        )


def test_prompt_orchestrator_builds_stage_bundle_with_rag_and_output_contract() -> None:
    config = StockConfig(focus_areas=["科技"])
    state = AnalysisState()
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供要研究的股票。")
    state.add_user_message("请分析示例科技股A，结合我的持仓成本。")
    orchestrator = StockPromptOrchestrator(config)

    bundle = orchestrator.stage_bundle(AnalysisStage.FUNDAMENTAL, state, FakeContextProvider())

    assert bundle.use_case == PromptUseCase.PORTFOLIO_REVIEW
    assert "资料可信度规则" in bundle.system
    assert "输出契约" in bundle.system
    assert "## 分析流程" in bundle.user
    assert "RAG_CONTEXT" in bundle.user
    assert "#needs_verification" in bundle.user


def test_prompt_orchestrator_skips_rag_for_initial_intake() -> None:
    orchestrator = StockPromptOrchestrator(StockConfig(focus_areas=["科技"]))

    bundle = orchestrator.stage_bundle(
        AnalysisStage.INTAKE,
        AnalysisState(),
        FakeContextProvider(),
    )

    assert "信息收集阶段暂不强制检索知识库" in bundle.user
    assert "RAG_CONTEXT" not in bundle.user


def test_langchain_harness_invokes_llm_with_orchestrated_prompt() -> None:
    llm = CapturingLLM()
    config = StockConfig(focus_areas=["科技"])
    state = AnalysisState()
    state.add_analyst_message(AnalysisStage.INTAKE, "请提供股票。")
    state.add_user_message("请分析 000001 示例科技股A。")
    harness = LangChainStockHarness(config, llm=llm)  # type: ignore[arg-type]

    result = harness.generate_result(AnalysisStage.FUNDAMENTAL, state)

    assert "进入观察池" in result.text
    assert llm.messages
    assert "资料可信度规则" in llm.messages[0].content
    assert "RAG_CONTEXT" not in llm.messages[1].content
    assert "未配置知识库" in llm.messages[1].content
    assert "## 分析流程" in llm.messages[1].content

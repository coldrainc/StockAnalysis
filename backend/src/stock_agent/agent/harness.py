from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from stock_agent.core.config import AnalysisStage, StockConfig
from stock_agent.agent.guardrails import HarnessGuardrails
from stock_agent.agent.harness_result import HarnessResult
from stock_agent.agent.multi_agent import AgentTask, MultiAgentOrchestrator
from stock_agent.rag.knowledge_base import MarkdownKnowledgeBase
from stock_agent.agent.prompt_orchestrator import StockPromptOrchestrator
from stock_agent.agent.state import AnalysisState
from stock_agent.services.web_search import WebSearchClient


class StockHarness(Protocol):
    guardrails: HarnessGuardrails

    def generate(self, stage: AnalysisStage, state: AnalysisState) -> str:
        ...

    def generate_result(self, stage: AnalysisStage, state: AnalysisState) -> HarnessResult:
        ...

    def respond_to_candidate_question(self, question: str, state: AnalysisState) -> str:
        ...

    def respond_to_candidate_question_result(
        self, question: str, state: AnalysisState
    ) -> HarnessResult:
        ...


class BaseStockHarness(ABC):
    def __init__(
        self,
        config: StockConfig,
        guardrails: HarnessGuardrails | None = None,
    ) -> None:
        self.config = config
        self.guardrails = guardrails or HarnessGuardrails()

    @abstractmethod
    def generate_result(self, stage: AnalysisStage, state: AnalysisState) -> HarnessResult:
        raise NotImplementedError

    def generate(self, stage: AnalysisStage, state: AnalysisState) -> str:
        return self.generate_result(stage, state).text

    @abstractmethod
    def respond_to_candidate_question_result(
        self, question: str, state: AnalysisState
    ) -> HarnessResult:
        raise NotImplementedError

    def respond_to_candidate_question(self, question: str, state: AnalysisState) -> str:
        return self.respond_to_candidate_question_result(question, state).text


class LangChainStockHarness(BaseStockHarness):
    """LangChain-backed harness that turns stock analysis state into model calls."""

    def __init__(
        self,
        config: StockConfig,
        llm: BaseChatModel | None = None,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        web_search: WebSearchClient | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        wire_api: str | None = None,
        temperature: float = 0.4,
        guardrails: HarnessGuardrails | None = None,
    ) -> None:
        super().__init__(config, guardrails=guardrails)
        llm_kwargs = {"model": model, "temperature": temperature}
        if base_url:
            llm_kwargs["base_url"] = base_url
        if api_key:
            llm_kwargs["api_key"] = api_key
        if wire_api == "responses":
            llm_kwargs["use_responses_api"] = True
        self.llm = llm or ChatOpenAI(**llm_kwargs)
        self.knowledge_base = knowledge_base
        self.web_search = web_search
        self.prompt_orchestrator = StockPromptOrchestrator(config)
        self.multi_agent_orchestrator = MultiAgentOrchestrator(config)

    def generate_result(self, stage: AnalysisStage, state: AnalysisState) -> HarnessResult:
        prompt_bundle = self.prompt_orchestrator.stage_bundle(stage, state, self)
        multi_agent_context = ""
        if self._should_run_multi_agent(stage, state):
            report = self.multi_agent_orchestrator.run(
                stage=stage,
                state=state,
                use_case=prompt_bundle.use_case.value,
                query=prompt_bundle.query,
                knowledge_context=prompt_bundle.knowledge_context,
                web_context=prompt_bundle.web_context,
                specialist_runner=self._invoke_specialist_agent,
            )
            multi_agent_context = report.to_prompt_context()
            prompt_bundle = self.prompt_orchestrator.stage_bundle(
                stage,
                state,
                self,
                multi_agent_context=multi_agent_context,
                knowledge_context=prompt_bundle.knowledge_context,
                web_context=prompt_bundle.web_context,
            )
        messages = prompt_bundle.to_messages()
        return self._safe_invoke(messages, fallback=self._fallback_message(stage, state))

    def respond_to_candidate_question_result(
        self, question: str, state: AnalysisState
    ) -> HarnessResult:
        prompt_bundle = self.prompt_orchestrator.follow_up_bundle(question, state, self)
        report = self.multi_agent_orchestrator.run(
            stage=state.stage,
            state=state,
            use_case=prompt_bundle.use_case.value,
            query=prompt_bundle.query,
            knowledge_context=prompt_bundle.knowledge_context,
            web_context=prompt_bundle.web_context,
            specialist_runner=self._invoke_specialist_agent,
        )
        prompt_bundle = self.prompt_orchestrator.follow_up_bundle(
            question,
            state,
            self,
            multi_agent_context=report.to_prompt_context(),
            knowledge_context=prompt_bundle.knowledge_context,
            web_context=prompt_bundle.web_context,
        )
        messages = prompt_bundle.to_messages()
        active_question = state.turns[-1].analyst if state.turns else "当前问题"
        fallback = (
            f"这个问题需要结合标的、周期和风险偏好判断。"
            f"请继续补充当前分析任务所需信息：{active_question}"
        )
        return self._safe_invoke(messages, fallback=fallback)

    def _current_focus(self, state: AnalysisState) -> str:
        return self.prompt_orchestrator.current_focus(state)

    def _knowledge_context(self, query: str) -> str:
        if self.knowledge_base is None:
            return "未配置知识库。"

        return self.knowledge_base.context_for(query)

    def _web_context(self, query: str) -> str:
        if self.web_search is None:
            return "未启用联网搜索。"
        try:
            return self.web_search.context_for(query)
        except Exception:
            return "联网搜索暂时不可用。"

    def knowledge_context(self, query: str) -> str:
        return self._knowledge_context(query)

    def web_context(self, query: str) -> str:
        return self._web_context(query)

    def _should_run_multi_agent(self, stage: AnalysisStage, state: AnalysisState) -> bool:
        if stage == AnalysisStage.INTAKE and not any(turn.user for turn in state.turns):
            return False
        return stage != AnalysisStage.COMPLETE

    def _invoke_specialist_agent(self, task: AgentTask) -> str:
        messages = [
            SystemMessage(content=task.system),
            HumanMessage(content=task.user),
        ]
        response = self.llm.invoke(messages)
        return self._content_to_text(response.content)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts).strip()
        return str(content).strip()

    def _safe_invoke(self, messages: list[Any], fallback: str) -> HarnessResult:
        try:
            response = self.llm.invoke(messages)
            raw_text = self._content_to_text(response.content)
            checked = self.guardrails.check_model_output(raw_text)
            return HarnessResult(text=checked.text, findings=checked.findings)
        except Exception:
            checked = self.guardrails.check_model_output(fallback)
            return HarnessResult(text=checked.text, findings=checked.findings, fallback_used=True)

    def _fallback_message(self, stage: AnalysisStage, state: AnalysisState) -> str:
        focus = self._current_focus(state)
        if stage == AnalysisStage.RECOMMENDATION:
            return "当前模型暂时不可用。我会先给出保守提示：请等待公告、财务数据和行情刷新后再形成完整推荐结论；存疑或未刷新资料需核验。仅供研究辅助，不构成投资建议。"
        if stage == AnalysisStage.RISK:
            return "我先追问风险约束：你的投资周期、可承受回撤和是否已有相关持仓是什么？"
        return f"我们继续围绕{focus}。请提供股票代码、公司名、市场、持有周期或你关心的核心问题。"


class ScriptedStockHarness(BaseStockHarness):
    """Deterministic harness for tests and offline demos."""

    def __init__(
        self,
        config: StockConfig,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        guardrails: HarnessGuardrails | None = None,
    ) -> None:
        super().__init__(config, guardrails=guardrails)
        self.knowledge_base = knowledge_base

    def generate_result(self, stage: AnalysisStage, state: AnalysisState) -> HarnessResult:
        focus = self.config.focus_areas[
            min(state.current_focus_index, len(self.config.focus_areas) - 1)
        ]
        if stage == AnalysisStage.INTAKE:
            text = (
                f"{self.config.investor.name}你好，我是股票量化研究和持仓分析助手。"
                f"我可以围绕{focus}生成每日候选观察池、做逐股基本面/催化剂分析，"
                "也可以结合你的持仓成本和仓位诊断风险。请给我股票代码、公司名、行业方向或持仓文件。"
            )
        elif stage == AnalysisStage.FUNDAMENTAL:
            text = f"基本面先看{focus}的收入质量、利润率、现金流和估值位置。你希望分析哪只股票？"
        elif stage == AnalysisStage.CATALYST:
            text = "催化剂可以从业绩拐点、产品价格、政策变化、订单和分红回购入手。"
        elif stage == AnalysisStage.RISK:
            text = "风险侧需要看估值、周期、财务质量、政策和组合集中度。"
        elif stage == AnalysisStage.RECOMMENDATION:
            text = "研究结论：可先列入量化观察池，等待公告、财务数据、行情和向量库刷新后再决定仓位。资料核验提醒：静态、第三方或过期资料需复核。仅供研究辅助，不构成投资建议。"
        else:
            text = "本轮股票分析结束。"
        checked = self.guardrails.check_model_output(text)
        return HarnessResult(text=checked.text, findings=checked.findings)

    def respond_to_candidate_question_result(
        self, question: str, state: AnalysisState
    ) -> HarnessResult:
        active_question = state.turns[-1].analyst if state.turns else "当前问题"
        text = (
            f"简短说明：这个问题需要结合标的、市场和投资周期判断。"
            f"请继续补充当前分析信息：{active_question}"
        )
        checked = self.guardrails.check_model_output(text)
        return HarnessResult(text=checked.text, findings=checked.findings)


InterviewHarness = StockHarness
BaseInterviewHarness = BaseStockHarness
LangChainInterviewHarness = LangChainStockHarness
ScriptedInterviewHarness = ScriptedStockHarness

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from stock_agent.core.config import AnalysisStage, StockConfig
from stock_agent.agent.state import AnalysisState


class PromptUseCase(str, Enum):
    STAGE_RESEARCH = "stage_research"
    FOLLOW_UP = "follow_up"
    STOCK_DEEP_DIVE = "stock_deep_dive"
    PORTFOLIO_REVIEW = "portfolio_review"
    OPPORTUNITY_POOL = "opportunity_pool"
    SOURCE_AUDIT = "source_audit"


class ContextProvider(Protocol):
    def knowledge_context(self, query: str) -> str:
        ...

    def web_context(self, query: str) -> str:
        ...


@dataclass(frozen=True)
class PromptBundle:
    use_case: PromptUseCase
    query: str
    system: str
    user: str
    knowledge_context: str = ""
    web_context: str = ""
    multi_agent_context: str = ""

    def to_messages(self) -> list[object]:
        return [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user),
        ]


class StockPromptOrchestrator:
    """Builds LLM prompts for stock research use cases from shared policy blocks."""

    def __init__(self, config: StockConfig) -> None:
        self.config = config

    def stage_bundle(
        self,
        stage: AnalysisStage,
        state: AnalysisState,
        context_provider: ContextProvider,
        multi_agent_context: str = "",
        knowledge_context: str | None = None,
        web_context: str | None = None,
    ) -> PromptBundle:
        focus = self.current_focus(state)
        query = self.context_query(stage, state, focus)
        if knowledge_context is None or web_context is None:
            knowledge_context, web_context = self._contexts(
                stage=stage,
                state=state,
                query=query,
                context_provider=context_provider,
            )
        intent = self._stage_instruction(stage, focus)
        return PromptBundle(
            use_case=self._use_case_for_stage(stage, state),
            query=query,
            system=self.system_prompt(),
            user=self._research_user_prompt(
                stage=stage.value,
                focus=focus,
                transcript=state.transcript() or "No prior turns.",
                knowledge_context=knowledge_context,
                web_context=web_context,
                multi_agent_context=multi_agent_context,
                intent=intent,
            ),
            knowledge_context=knowledge_context,
            web_context=web_context,
            multi_agent_context=multi_agent_context,
        )

    def follow_up_bundle(
        self,
        question: str,
        state: AnalysisState,
        context_provider: ContextProvider,
        multi_agent_context: str = "",
        knowledge_context: str | None = None,
        web_context: str | None = None,
    ) -> PromptBundle:
        focus = self.current_focus(state)
        query = self.context_query(state.stage, state, focus, extra=question)
        if knowledge_context is None:
            knowledge_context = context_provider.knowledge_context(query)
        if web_context is None:
            web_context = context_provider.web_context(query)
        active_task = state.turns[-1].analyst if state.turns else "暂无当前任务。"
        intent = self._follow_up_instruction()
        return PromptBundle(
            use_case=self._follow_up_use_case(question),
            query=query,
            system=self.system_prompt(),
            user=self._follow_up_user_prompt(
                active_task=active_task,
                question=question,
                knowledge_context=knowledge_context,
                web_context=web_context,
                multi_agent_context=multi_agent_context,
                intent=intent,
            ),
            knowledge_context=knowledge_context,
            web_context=web_context,
            multi_agent_context=multi_agent_context,
        )

    def system_prompt(self) -> str:
        context = self.config.to_prompt_context()
        return f"""你是一位严谨的股票量化研究、机会挖掘和持仓分析 Agent，面向中文用户工作。

{self._investor_block(context)}

覆盖行业：{context["focus_areas"]}

分析框架：
{context["analysis_framework"]}

{self._source_reliability_policy()}

{self._research_policy()}

{self._output_contract(context["disclaimer"])}"""

    def current_focus(self, state: AnalysisState) -> str:
        if not self.config.focus_areas:
            return "股票研究"
        index = min(state.current_focus_index, len(self.config.focus_areas) - 1)
        return self.config.focus_areas[index]

    def context_query(
        self,
        stage: AnalysisStage,
        state: AnalysisState,
        focus: str,
        extra: str = "",
    ) -> str:
        last_answer = ""
        if state.turns and state.turns[-1].user:
            last_answer = state.turns[-1].user
        return f"{stage.value} {focus} {last_answer} {extra}".strip()

    def _contexts(
        self,
        stage: AnalysisStage,
        state: AnalysisState,
        query: str,
        context_provider: ContextProvider,
    ) -> tuple[str, str]:
        if not self._should_retrieve_context(stage, state):
            return "信息收集阶段暂不强制检索知识库。", "信息收集阶段暂不联网搜索。"
        return context_provider.knowledge_context(query), context_provider.web_context(query)

    def _should_retrieve_context(self, stage: AnalysisStage, state: AnalysisState) -> bool:
        return stage != AnalysisStage.INTAKE or any(turn.user for turn in state.turns)

    def _use_case_for_stage(
        self,
        stage: AnalysisStage,
        state: AnalysisState,
    ) -> PromptUseCase:
        last_user = state.turns[-1].user if state.turns else ""
        text = (last_user or "").lower()
        if stage == AnalysisStage.RECOMMENDATION or any(
            keyword in text for keyword in ("推荐", "可能要涨", "今日关注", "强关注", "top")
        ):
            return PromptUseCase.OPPORTUNITY_POOL
        if any(keyword in text for keyword in ("持仓", "仓位", "成本价")):
            return PromptUseCase.PORTFOLIO_REVIEW
        if any(keyword in text for keyword in ("核验", "资料", "真假", "存疑", "needs_verification")):
            return PromptUseCase.SOURCE_AUDIT
        if any(char.isdigit() for char in text) or "分析" in text:
            return PromptUseCase.STOCK_DEEP_DIVE
        return PromptUseCase.STAGE_RESEARCH

    def _follow_up_use_case(self, question: str) -> PromptUseCase:
        text = question.lower()
        if any(keyword in text for keyword in ("持仓", "仓位", "成本")):
            return PromptUseCase.PORTFOLIO_REVIEW
        if any(keyword in text for keyword in ("核验", "真假", "来源", "tag", "标签")):
            return PromptUseCase.SOURCE_AUDIT
        return PromptUseCase.FOLLOW_UP

    def _stage_instruction(self, stage: AnalysisStage, focus: str) -> str:
        instructions = {
            AnalysisStage.INTAKE: "用中文简短说明你能做每日量化候选、推荐观察池、逐股分析和持仓诊断，然后询问用户想分析的股票、行业、持仓文件或投资目标。",
            AnalysisStage.FUNDAMENTAL: f"围绕{focus}或用户给定标的，做基本面分析，并指出仍需补充的数据。",
            AnalysisStage.CATALYST: "基于用户输入和知识库，分析行业/公司催化剂、价格变量、政策变量和估值驱动。",
            AnalysisStage.RISK: "聚焦风险：估值、财务、周期、政策、流动性、竞争和组合适配风险。",
            AnalysisStage.RECOMMENDATION: "给出结构化研究结论和原则性推荐分层，必须包含量化排序依据、组合适配、风险提示、资料核验提醒和免责声明。",
        }
        return instructions.get(stage, "用中文结束本轮分析。")

    def _follow_up_instruction(self) -> str:
        return (
            "用中文回答用户的问题，并服务于当前股票研究：\n"
            "1. 先说明你能确认的信息和不能确认的信息。\n"
            "2. 如果涉及具体股票，给出需要补充的数据口径，如财报期、市场、持仓周期、风险偏好。\n"
            "3. 从知识库和联网搜索中提炼相关行业、公司、风险或催化剂。\n"
            "4. 最后用一句话把用户带回当前分析任务。\n"
            "不要承诺收益，不要给出绝对买卖指令。"
        )

    def _research_user_prompt(
        self,
        stage: str,
        focus: str,
        transcript: str,
        knowledge_context: str,
        web_context: str,
        multi_agent_context: str,
        intent: str,
    ) -> str:
        return f"""当前阶段：{stage}
当前重点：{focus}

对话记录：
{transcript}

{self._context_pack(knowledge_context, web_context)}

{self._multi_agent_pack(multi_agent_context)}

指令：
{intent}

输出结构：
{self._answer_scaffold()}"""

    def _follow_up_user_prompt(
        self,
        active_task: str,
        question: str,
        knowledge_context: str,
        web_context: str,
        multi_agent_context: str,
        intent: str,
    ) -> str:
        return f"""用户在股票研究过程中提出了澄清问题或追加约束。

当前分析任务：
{active_task}

用户的问题：
{question}

{self._context_pack(knowledge_context, web_context)}

{self._multi_agent_pack(multi_agent_context)}

指令：
{intent}

输出结构：
{self._answer_scaffold()}"""

    def _context_pack(self, knowledge_context: str, web_context: str) -> str:
        return f"""知识库上下文：
{knowledge_context}

联网搜索上下文：
{web_context}"""

    def _multi_agent_pack(self, multi_agent_context: str) -> str:
        if not multi_agent_context:
            return "多Agent协作上下文：\n本轮尚未运行专业 Agent；请按基础研究框架回答。"
        return f"""多Agent协作上下文：
{multi_agent_context}

请在“## 分析流程”中用公开摘要说明哪些 Agent 参与、各自检查了什么证据和哪些资料标签需要提醒。"""

    def _investor_block(self, context: dict[str, str]) -> str:
        return f"""投资者画像：
- 姓名：{context["investor_name"]}
- 关注市场：{context["market"]}
- 风险偏好：{context["risk_appetite"]}
- 投资周期：{context["investment_horizon"]}
- 资金/持仓备注：{context["capital_notes"]}"""

    def _source_reliability_policy(self) -> str:
        return """资料可信度规则：
- 必须区分事实、推断和不确定性；不确定时要求补充数据或提示需要联网刷新。
- 必须读取知识库上下文中的“资料标签/资料可信度/核验提示”。如果出现 #needs_verification、#needs_refresh、#third_party_dataset 或 #user_supplied，回答中必须单独列出“资料核验提醒”，说明哪些信息存疑或需要刷新。
- 对“公司公告、财务数据、行情价格”只把带 #official_or_exchange 的内容当作较高可信资料；第三方行情或用户手动资料只能作为线索。
- 当资料超过当日或出现 #needs_refresh 时，必须提醒先运行数据刷新和向量库同步，再给出较强推荐。"""

    def _research_policy(self) -> str:
        return """研究编排规则：
- 默认围绕每日量化候选池、逐股基本面、公告催化、行情量价、估值约束、组合适配和风险控制组织答案。
- 当用户问“推荐/可能要涨/今天关注什么”时，优先引用 RAG 中的 daily-picks、动态行情和逐股文档；必须说明评分、入选逻辑、触发条件、风险标签和是否适合现有持仓。
- 当用户提供持仓时，必须分析成本价、浮盈浮亏、仓位集中度、同主题暴露、候选池匹配度和减仓/加仓前置条件。
- 每轮最多提出 1-3 个关键澄清问题，避免泛泛而谈。
- 不输出“稳赚”“必涨”“无风险”等承诺。"""

    def _output_contract(self, disclaimer: str) -> str:
        return f"""输出契约：
- 每次回答必须包含公开的“## 分析流程”小节，用 3-6 条概括本轮如何读取输入、多Agent分工、检索 RAG/行情、核验资料标签、交叉分析和形成结论；这是给用户看的可审计流程摘要，不要输出模型内部隐藏推理或逐字思维链。
- 推荐结论必须包含：候选标的/方向、量化评分或排序依据、核心逻辑、组合适配、主要风险、触发/失效条件、观察指标、仓位/止损纪律的原则性提示。
- 结尾保留免责声明：{disclaimer}"""

    def _answer_scaffold(self) -> str:
        return """## 分析流程
1. 识别目标：说明本轮用户问题、市场/持仓/周期约束。
2. 多Agent分工：说明 DataAgent、QuantAgent、FundamentalAgent、CatalystAgent、PortfolioAgent、RiskAgent、SupervisorAgent 中哪些参与了本轮。
3. 检索资料：说明使用了 RAG、日报、逐股文档、行情或联网信息中的哪些类型。
4. 核验标签：说明是否存在 #needs_verification、#needs_refresh、#third_party_dataset、#user_supplied。
5. 交叉判断：说明量价、基本面、催化剂、风险和组合适配如何被放到同一框架。
6. 输出结论：说明候选分层、触发条件、失效条件和风险纪律。

## 结论
先给一句可执行摘要，再展开依据。

## 资料核验提醒
只要上下文出现存疑标签就必须列出。

## 风险与纪律
说明触发/失效条件、观察指标和原则性仓位纪律。"""

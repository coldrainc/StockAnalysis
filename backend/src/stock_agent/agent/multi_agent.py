from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import re

from stock_agent.core.config import AnalysisStage, StockConfig
from stock_agent.agent.state import AnalysisState


class AgentRole(str, Enum):
    DATA = "data"
    FUNDAMENTAL = "fundamental"
    QUANT = "quant"
    CATALYST = "catalyst"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    SUPERVISOR = "supervisor"


@dataclass(frozen=True)
class AgentSpec:
    role: AgentRole
    name: str
    mission: str
    evidence_focus: tuple[str, ...]


@dataclass(frozen=True)
class AgentTask:
    spec: AgentSpec
    system: str
    user: str


@dataclass(frozen=True)
class AgentFinding:
    role: AgentRole
    agent_name: str
    title: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: str = "medium"
    next_action: str = ""
    model_backed: bool = False

    def to_markdown(self) -> str:
        evidence = "；".join(self.evidence) if self.evidence else "暂无可引用证据摘要。"
        tags = " ".join(self.tags) if self.tags else "#no_special_tag"
        model_source = "LLM specialist" if self.model_backed else "rule fallback"
        lines = [
            f"#### {self.agent_name}",
            f"- 角色：{self.role.value}",
            f"- 关键发现：{self.summary}",
            f"- 证据焦点：{evidence}",
            f"- 可信度：{self.confidence}",
            f"- 标签：{tags}",
            f"- 下一步：{self.next_action or '进入 SupervisorAgent 综合。'}",
            f"- 生成方式：{model_source}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class MultiAgentReport:
    stage: AnalysisStage
    use_case: str
    query: str
    findings: list[AgentFinding]
    warnings: list[str] = field(default_factory=list)

    def participating_agents(self) -> list[str]:
        return [finding.agent_name for finding in self.findings]

    def to_prompt_context(self) -> str:
        if not self.findings:
            return "多Agent编排报告：本轮未触发专业 Agent。"
        lines = [
            "## 多Agent编排报告",
            f"- 阶段：{self.stage.value}",
            f"- 用例：{self.use_case}",
            f"- 查询：{self.query or '未提供明确查询'}",
            f"- 参与 Agent：{'、'.join(self.participating_agents())}",
            "",
            "### 专业 Agent 公开发现",
        ]
        lines.extend(finding.to_markdown() for finding in self.findings)
        if self.warnings:
            lines.append("")
            lines.append("### 编排提醒")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n\n".join(lines)


SpecialistRunner = Callable[[AgentTask], str]


class MultiAgentOrchestrator:
    """Coordinates specialist stock-research agents before final synthesis."""

    def __init__(self, config: StockConfig, max_context_chars: int = 2200) -> None:
        self.config = config
        self.max_context_chars = max_context_chars
        self._specs = {
            AgentRole.DATA: AgentSpec(
                role=AgentRole.DATA,
                name="DataAgent",
                mission="核验 RAG、公告、行情、联网资料和资料标签，明确哪些信息可用、过期或存疑。",
                evidence_focus=("RAG 逐股文档", "公告/交易所", "行情刷新", "资料可信度标签"),
            ),
            AgentRole.FUNDAMENTAL: AgentSpec(
                role=AgentRole.FUNDAMENTAL,
                name="FundamentalAgent",
                mission="分析公司业务、收入质量、利润率、现金流、资产负债表和估值约束。",
                evidence_focus=("财报", "业务结构", "盈利质量", "估值区间"),
            ),
            AgentRole.QUANT: AgentSpec(
                role=AgentRole.QUANT,
                name="QuantAgent",
                mission="分析量价、成交额、换手、趋势、波动和候选池排序信号。",
                evidence_focus=("当日行情", "成交额/换手", "趋势动量", "量化评分"),
            ),
            AgentRole.CATALYST: AgentSpec(
                role=AgentRole.CATALYST,
                name="CatalystAgent",
                mission="识别业绩、政策、订单、产品价格、产业链和主题事件催化剂。",
                evidence_focus=("公告事件", "政策变量", "产品价格", "产业链景气度"),
            ),
            AgentRole.PORTFOLIO: AgentSpec(
                role=AgentRole.PORTFOLIO,
                name="PortfolioAgent",
                mission="结合用户持仓、成本、仓位集中度、主题暴露和组合适配形成持仓诊断。",
                evidence_focus=("持仓金额", "成本价", "同主题暴露", "仓位纪律"),
            ),
            AgentRole.RISK: AgentSpec(
                role=AgentRole.RISK,
                name="RiskAgent",
                mission="审查估值、财务、周期、政策、流动性和资料真实性风险，给出失效条件。",
                evidence_focus=("风险标签", "估值风险", "回撤约束", "资料核验"),
            ),
            AgentRole.SUPERVISOR: AgentSpec(
                role=AgentRole.SUPERVISOR,
                name="SupervisorAgent",
                mission="汇总各专业 Agent 公开发现，组织最终回答、候选分层和风险纪律。",
                evidence_focus=("专业 Agent 发现", "冲突信息", "结论分层", "下一步动作"),
            ),
        }

    def run(
        self,
        *,
        stage: AnalysisStage,
        state: AnalysisState,
        use_case: str,
        query: str,
        knowledge_context: str,
        web_context: str,
        specialist_runner: SpecialistRunner | None = None,
    ) -> MultiAgentReport:
        tags = self._source_tags(knowledge_context, web_context)
        warnings = self._warnings_for_tags(tags)
        transcript = state.transcript() or "No prior turns."
        roles = self._select_roles(stage, use_case, transcript)
        findings: list[AgentFinding] = []

        for role in roles:
            spec = self._specs[role]
            task = self._build_task(
                spec=spec,
                stage=stage,
                use_case=use_case,
                query=query,
                transcript=transcript,
                knowledge_context=knowledge_context,
                web_context=web_context,
                source_tags=tags,
                prior_findings=findings,
            )
            finding = self._run_agent_task(
                task=task,
                specialist_runner=specialist_runner,
                tags=tags,
                knowledge_context=knowledge_context,
                web_context=web_context,
                prior_findings=findings,
            )
            findings.append(finding)

        return MultiAgentReport(
            stage=stage,
            use_case=use_case,
            query=query,
            findings=findings,
            warnings=warnings,
        )

    def _select_roles(
        self,
        stage: AnalysisStage,
        use_case: str,
        transcript: str,
    ) -> list[AgentRole]:
        portfolio_requested = self._looks_like_portfolio_request(transcript)
        opportunity_requested = use_case in {"opportunity_pool", "stock_deep_dive"}

        if stage == AnalysisStage.INTAKE:
            roles = [AgentRole.DATA, AgentRole.SUPERVISOR]
        elif stage == AnalysisStage.FUNDAMENTAL:
            roles = [AgentRole.DATA, AgentRole.FUNDAMENTAL, AgentRole.RISK]
            if opportunity_requested:
                roles.insert(2, AgentRole.QUANT)
        elif stage == AnalysisStage.CATALYST:
            roles = [AgentRole.DATA, AgentRole.CATALYST, AgentRole.QUANT, AgentRole.RISK]
        elif stage == AnalysisStage.RISK:
            roles = [AgentRole.DATA, AgentRole.RISK, AgentRole.QUANT]
        elif stage == AnalysisStage.RECOMMENDATION:
            roles = [
                AgentRole.DATA,
                AgentRole.QUANT,
                AgentRole.FUNDAMENTAL,
                AgentRole.CATALYST,
                AgentRole.RISK,
            ]
        else:
            roles = [AgentRole.DATA, AgentRole.RISK]

        if portfolio_requested and AgentRole.PORTFOLIO not in roles:
            insert_at = max(1, len(roles) - 1)
            roles.insert(insert_at, AgentRole.PORTFOLIO)
        if AgentRole.SUPERVISOR not in roles:
            roles.append(AgentRole.SUPERVISOR)
        return roles

    def _build_task(
        self,
        *,
        spec: AgentSpec,
        stage: AnalysisStage,
        use_case: str,
        query: str,
        transcript: str,
        knowledge_context: str,
        web_context: str,
        source_tags: list[str],
        prior_findings: list[AgentFinding],
    ) -> AgentTask:
        prior = "\n".join(
            f"- {finding.agent_name}: {finding.summary}" for finding in prior_findings[-4:]
        )
        if not prior:
            prior = "暂无前序 Agent 发现。"

        system = f"""你是股票量化研究系统中的 {spec.name}。

职责：{spec.mission}

要求：
- 只输出给用户可见的公开研究摘要，不输出隐藏思维链或逐字推理过程。
- 必须区分事实、推断和需要核验的信息。
- 涉及推荐时只能使用“观察、候选、触发、失效、风险纪律”等表述，不承诺收益。
- 输出 4 行以内，包含：关键发现、证据、存疑标签、下一步。"""

        user = f"""当前阶段：{stage.value}
当前用例：{use_case}
当前查询：{query or '未提供明确查询'}
关注市场：{self.config.investor.market}
风险偏好：{self.config.investor.risk_appetite}
投资周期：{self.config.investor.investment_horizon}

对话记录：
{self._clip(transcript, 1400)}

知识库上下文：
{self._clip(knowledge_context, self.max_context_chars)}

联网/行情上下文：
{self._clip(web_context, self.max_context_chars)}

已识别资料标签：{' '.join(source_tags) if source_tags else '#no_special_tag'}

前序 Agent 发现：
{prior}

请从你的专业角色出发给出公开 finding。"""
        return AgentTask(spec=spec, system=system, user=user)

    def _run_agent_task(
        self,
        *,
        task: AgentTask,
        specialist_runner: SpecialistRunner | None,
        tags: list[str],
        knowledge_context: str,
        web_context: str,
        prior_findings: list[AgentFinding],
    ) -> AgentFinding:
        if specialist_runner is None:
            return self._fallback_finding(task.spec, tags, knowledge_context, web_context)

        try:
            raw = specialist_runner(task)
        except Exception:
            return self._fallback_finding(task.spec, tags, knowledge_context, web_context)

        summary = self._compact_model_output(raw)
        if not summary:
            return self._fallback_finding(task.spec, tags, knowledge_context, web_context)

        confidence = self._confidence_for_tags(tags)
        if task.spec.role == AgentRole.SUPERVISOR and prior_findings:
            evidence = [f"{finding.agent_name}: {finding.title}" for finding in prior_findings[-4:]]
        else:
            evidence = list(task.spec.evidence_focus[:3])
        return AgentFinding(
            role=task.spec.role,
            agent_name=task.spec.name,
            title=task.spec.mission,
            summary=summary,
            evidence=evidence,
            tags=tags or ["#no_special_tag"],
            confidence=confidence,
            next_action=self._next_action(task.spec.role, tags),
            model_backed=True,
        )

    def _fallback_finding(
        self,
        spec: AgentSpec,
        tags: list[str],
        knowledge_context: str,
        web_context: str,
    ) -> AgentFinding:
        tag_text = " ".join(tags) if tags else "#no_special_tag"
        summaries = {
            AgentRole.DATA: (
                "已检查 RAG 与联网/行情上下文；需要优先引用公告、交易所和当日刷新数据，"
                f"当前识别标签：{tag_text}。"
            ),
            AgentRole.FUNDAMENTAL: "需要围绕收入质量、利润率、现金流、资产负债表和估值区间做逐股基本面交叉验证。",
            AgentRole.QUANT: "需要把当日涨跌幅、成交额、换手、趋势动量和风险惩罚纳入候选池排序。",
            AgentRole.CATALYST: "需要核验业绩拐点、订单、产品价格、政策和产业链景气度是否存在可跟踪催化。",
            AgentRole.PORTFOLIO: "需要结合用户持仓金额、成本价、同主题暴露和可承受回撤判断是否适合加减仓。",
            AgentRole.RISK: "需要把估值、财务质量、周期下行、流动性和资料真实性作为失效条件。",
            AgentRole.SUPERVISOR: "将各专业 Agent 的公开 finding 汇总成候选分层、触发条件、失效条件和资料核验提醒。",
        }
        evidence = self._fallback_evidence(spec, knowledge_context, web_context)
        return AgentFinding(
            role=spec.role,
            agent_name=spec.name,
            title=spec.mission,
            summary=summaries[spec.role],
            evidence=evidence,
            tags=tags or ["#no_special_tag"],
            confidence=self._confidence_for_tags(tags),
            next_action=self._next_action(spec.role, tags),
            model_backed=False,
        )

    def _fallback_evidence(
        self,
        spec: AgentSpec,
        knowledge_context: str,
        web_context: str,
    ) -> list[str]:
        evidence = list(spec.evidence_focus[:2])
        if knowledge_context and "未配置知识库" not in knowledge_context:
            evidence.append("RAG 上下文可用")
        if web_context and "未启用联网搜索" not in web_context:
            evidence.append("联网/行情上下文可用")
        return evidence[:4]

    def _source_tags(self, knowledge_context: str, web_context: str) -> list[str]:
        text = f"{knowledge_context}\n{web_context}"
        tags = sorted(set(re.findall(r"#[A-Za-z][A-Za-z0-9_]+", text)))
        return tags

    def _warnings_for_tags(self, tags: list[str]) -> list[str]:
        warnings: list[str] = []
        if "#needs_verification" in tags:
            warnings.append("资料中存在 #needs_verification，最终回答必须提示用户先核验来源。")
        if "#needs_refresh" in tags:
            warnings.append("资料中存在 #needs_refresh，推荐强度需要等待行情/公告刷新后再提高。")
        if "#third_party_dataset" in tags:
            warnings.append("资料中存在 #third_party_dataset，第三方数据只可作为线索。")
        if "#user_supplied" in tags:
            warnings.append("资料中存在 #user_supplied，用户手动输入需与公告、财报或行情交叉校验。")
        return warnings

    def _confidence_for_tags(self, tags: list[str]) -> str:
        weak_tags = {"#needs_verification", "#needs_refresh", "#third_party_dataset"}
        if any(tag in weak_tags for tag in tags):
            return "low_to_medium"
        if "#official_or_exchange" in tags:
            return "medium_to_high"
        return "medium"

    def _next_action(self, role: AgentRole, tags: list[str]) -> str:
        if tags and any(tag in {"#needs_verification", "#needs_refresh"} for tag in tags):
            return "先标注资料核验提醒，再进入结论分层。"
        if role == AgentRole.SUPERVISOR:
            return "组织最终回答并保留免责声明。"
        return "交给 SupervisorAgent 做跨角色综合。"

    def _looks_like_portfolio_request(self, text: str) -> bool:
        keywords = ("持仓", "仓位", "成本", "浮盈", "浮亏")
        return any(keyword in text for keyword in keywords)

    def _clip(self, text: str, limit: int) -> str:
        value = (text or "").strip()
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "\n...[已截断给专业 Agent，完整资料仍保留在知识库/文档中]"

    def _compact_model_output(self, text: str, limit: int = 700) -> str:
        cleaned = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
        cleaned = re.sub(r"(?im)^#{1,3}\s*(分析流程|思考流程|推理过程)\s*$", "公开流程摘要：", cleaned)
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip() + "...[专业 Agent 摘要已压缩]"

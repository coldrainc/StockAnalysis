from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AnalysisStage(str, Enum):
    INTAKE = "intake"
    FUNDAMENTAL = "fundamental"
    CATALYST = "catalyst"
    RISK = "risk"
    RECOMMENDATION = "recommendation"
    COMPLETE = "complete"


class InvestorProfile(BaseModel):
    name: str = "投资者"
    market: str = "A股、港股、美股"
    risk_appetite: str = "中等"
    investment_horizon: str = "3-12个月"
    capital_notes: str = "未提供资金规模、现有持仓、成本价和仓位约束。"


class StockConfig(BaseModel):
    investor: InvestorProfile = Field(default_factory=InvestorProfile)
    focus_areas: list[str] = Field(
        default_factory=lambda: [
            "科技",
            "材料",
            "贵金属",
            "能源",
            "锂电池",
            "银行",
            "具身智能",
            "无人机",
            "机器人",
            "硬件",
        ]
    )
    max_turns: int = 6
    analysis_framework: dict[str, str] = Field(
        default_factory=lambda: {
            "fundamental": "收入质量、盈利能力、资产负债表、现金流和估值区间。",
            "industry": "行业景气度、供需格局、政策周期和竞争地位。",
            "catalyst": "业绩拐点、订单、价格、产能、分红回购、监管和宏观变量。",
            "quant": "每日量价、流动性、换手、估值约束、主题强度和风险惩罚，用于生成候选观察池。",
            "portfolio": "结合持仓成本、仓位集中度、浮盈浮亏、回撤纪律和候选池匹配度做持仓诊断。",
            "risk": "估值过高、财务质量、周期下行、流动性、汇率利率和黑天鹅。",
            "suitability": "匹配投资期限、风险偏好、组合分散度和止损纪律。",
        }
    )
    disclaimer: str = "仅供研究辅助，不构成投资建议或收益承诺。"

    @classmethod
    def from_json_file(cls, path: Path) -> "StockConfig":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def to_prompt_context(self) -> dict[str, Any]:
        return {
            "investor_name": self.investor.name,
            "market": self.investor.market,
            "risk_appetite": self.investor.risk_appetite,
            "investment_horizon": self.investor.investment_horizon,
            "capital_notes": self.investor.capital_notes,
            "focus_areas": "、".join(self.focus_areas),
            "analysis_framework": "\n".join(
                f"- {key}: {value}" for key, value in self.analysis_framework.items()
            ),
            "disclaimer": self.disclaimer,
        }


# Backward-compatible names keep the migrated harness/tests simple.
InterviewStage = AnalysisStage
CandidateProfile = InvestorProfile
InterviewConfig = StockConfig

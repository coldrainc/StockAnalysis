from __future__ import annotations

import csv
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from stock_agent.market.a_share_refresh import (
    AShareRefreshCompany,
    QuoteSnapshot,
    fetch_quote_snapshot,
    load_manifest_companies,
)
from stock_agent.market.a_share_tech import (
    NEEDS_REFRESH_TAG,
    NEEDS_VERIFICATION_TAG,
    SPECIAL_TREATMENT_TAG,
    THIRD_PARTY_TAG,
    format_money,
    format_number,
    safe_filename,
    utc_now,
)


QUANT_CANDIDATE_TAG = "#quant_candidate"
PORTFOLIO_TAG = "#portfolio_analysis"
MODEL_TAG = "#rule_based_quant_score"


@dataclass(frozen=True)
class PortfolioPosition:
    code: str
    name: str = ""
    market: str = "A股"
    shares: float | None = None
    cost_price: float | None = None
    market_value: float | None = None
    notes: str = ""

    @property
    def normalized_code(self) -> str:
        return normalize_code(self.code)


@dataclass(frozen=True)
class DailyPickCandidate:
    company: AShareRefreshCompany
    quote: QuoteSnapshot | None
    score: float
    rating: str
    reasons: tuple[str, ...]
    risk_flags: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioAnalysis:
    position: PortfolioPosition
    company: AShareRefreshCompany | None
    quote: QuoteSnapshot | None
    effective_shares: float | None
    market_value: float | None
    unrealized_pnl: float | None
    unrealized_pnl_pct: float | None
    in_top_picks: bool
    risk_flags: tuple[str, ...]


@dataclass(frozen=True)
class DailyPicksResult:
    output_dir: Path
    report_path: Path
    latest_path: Path
    generated_at: str
    universe_count: int
    scanned_count: int
    picked_count: int
    portfolio_count: int


def build_daily_picks(
    output_dir: Path,
    *,
    portfolio_path: Path | None = None,
    max_candidates: int | None = 800,
    top_k: int = 30,
    workers: int = 8,
    timeout: float = 8.0,
    categories: Iterable[str] = (),
) -> DailyPicksResult:
    generated_at = utc_now()
    companies = load_manifest_companies(output_dir / "manifest.json")
    filtered = filter_companies(companies, categories)
    selected = filtered[:max_candidates] if max_candidates is not None else filtered
    positions = load_portfolio(portfolio_path) if portfolio_path else []
    quote_targets = include_portfolio_quote_targets(selected, companies, positions)
    quotes = fetch_quotes(quote_targets, workers=workers, timeout=timeout)
    candidates = [score_company(company, quotes.get(company.record.code)) for company in selected]
    candidates.sort(key=lambda item: item.score, reverse=True)
    picks = candidates[: max(top_k, 0)]
    portfolio = analyze_portfolio(positions, companies, quotes, picks)

    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = daily_dir / f"{date_slug}_daily_picks.md"
    latest_path = daily_dir / "latest_daily_picks.md"
    content = render_daily_report(
        picks,
        portfolio,
        generated_at=generated_at,
        universe_count=len(companies),
        scanned_count=len(selected),
        max_candidates=max_candidates,
        top_k=top_k,
        categories=tuple(categories),
    )
    report_path.write_text(content, encoding="utf-8")
    shutil.copyfile(report_path, latest_path)
    return DailyPicksResult(
        output_dir=output_dir,
        report_path=report_path,
        latest_path=latest_path,
        generated_at=generated_at,
        universe_count=len(companies),
        scanned_count=len(selected),
        picked_count=len(picks),
        portfolio_count=len(portfolio),
    )


def filter_companies(
    companies: list[AShareRefreshCompany],
    categories: Iterable[str],
) -> list[AShareRefreshCompany]:
    category_terms = tuple(term.strip() for term in categories if term and term.strip())
    if not category_terms:
        return companies
    selected = []
    for company in companies:
        haystack = " ".join([company.record.name, company.record.industry, *company.categories])
        if any(term in haystack for term in category_terms):
            selected.append(company)
    return selected


def include_portfolio_quote_targets(
    selected: list[AShareRefreshCompany],
    companies: list[AShareRefreshCompany],
    positions: list[PortfolioPosition],
) -> list[AShareRefreshCompany]:
    if not positions:
        return selected
    by_code = {company.record.code: company for company in companies}
    selected_codes = {company.record.code for company in selected}
    targets = list(selected)
    for position in positions:
        code = position.normalized_code
        company = by_code.get(code)
        if company and code not in selected_codes:
            targets.append(company)
            selected_codes.add(code)
    return targets


def fetch_quotes(
    companies: list[AShareRefreshCompany],
    *,
    workers: int,
    timeout: float,
) -> dict[str, QuoteSnapshot]:
    quotes: dict[str, QuoteSnapshot] = {}
    if not companies:
        return quotes
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = {
            executor.submit(fetch_quote_snapshot, company.record, timeout=timeout): company
            for company in companies
        }
        for future in as_completed(futures):
            company = futures[future]
            try:
                quote = future.result()
            except Exception:
                quote = None
            if quote:
                quotes[company.record.code] = quote
    return quotes


def score_company(
    company: AShareRefreshCompany,
    quote: QuoteSnapshot | None,
) -> DailyPickCandidate:
    score = 18.0
    reasons: list[str] = []
    risks: list[str] = []
    category_bonus = category_score(company.categories)
    score += category_bonus
    if category_bonus:
        reasons.append(f"主题方向加分 {category_bonus:.1f}：{'、'.join(company.categories)}")
    if quote is None:
        risks.append("行情缺失，不能判断当日量价强弱")
        score -= 8.0
    else:
        momentum_score = score_momentum(quote.pct_change)
        liquidity_score = score_liquidity(quote.amount)
        turnover_score = score_turnover(quote.turnover_rate)
        valuation_score = score_valuation(quote.pe_dynamic, quote.pb)
        score += momentum_score + liquidity_score + turnover_score + valuation_score
        if quote.pct_change is not None:
            reasons.append(f"涨跌幅因子 {momentum_score:.1f}：{quote.pct_change:.2f}%")
        if quote.amount is not None:
            reasons.append(f"成交额因子 {liquidity_score:.1f}：{format_money(quote.amount)}")
        if quote.turnover_rate is not None:
            reasons.append(f"换手因子 {turnover_score:.1f}：{quote.turnover_rate:.2f}%")
        if quote.pe_dynamic is not None or quote.pb is not None:
            reasons.append(
                "估值约束 "
                f"{valuation_score:.1f}：PE={format_number(quote.pe_dynamic)}，PB={format_number(quote.pb)}"
            )
        if quote.pct_change is not None and quote.pct_change < -4:
            risks.append("当日跌幅较大，需确认是否有利空或破位")
        if quote.pct_change is not None and quote.pct_change > 9:
            risks.append("涨幅过高，追高风险较大")
        if quote.amount is not None and quote.amount < 50_000_000:
            risks.append("成交额偏低，流动性不足")
        if quote.pe_dynamic is not None and quote.pe_dynamic < 0:
            risks.append("PE 为负，盈利口径可能亏损")
    if SPECIAL_TREATMENT_TAG in company.record.tags:
        risks.append("ST/风险警示股票，优先降低权重")
        score -= 18.0
    if not reasons:
        reasons.append("仅有静态公司画像，需等待行情和公告刷新")
    score = max(0.0, min(100.0, score))
    return DailyPickCandidate(
        company=company,
        quote=quote,
        score=round(score, 2),
        rating=rating_for_score(score),
        reasons=tuple(reasons[:5]),
        risk_flags=tuple(dict.fromkeys(risks)),
    )


def category_score(categories: tuple[str, ...]) -> float:
    score = 0.0
    joined = "、".join(categories)
    if "AI算力" in joined:
        score += 8
    if "半导体" in joined:
        score += 7
    if "机器人" in joined or "无人机" in joined:
        score += 6
    if "新能源" in joined or "锂电" in joined:
        score += 5
    if "软件" in joined or "通信" in joined or "硬件" in joined:
        score += 4
    return min(score, 18.0)


def score_momentum(pct_change: float | None) -> float:
    if pct_change is None:
        return 0.0
    if pct_change < -6:
        return -8.0
    if pct_change < -2:
        return -3.0
    if 0 <= pct_change <= 5:
        return 8.0 + pct_change * 2.0
    if 5 < pct_change <= 9:
        return 14.0 - (pct_change - 5) * 1.2
    if pct_change > 9:
        return 5.0
    return 2.0


def score_liquidity(amount: float | None) -> float:
    if amount is None or amount <= 0:
        return 0.0
    return min(18.0, max(0.0, (math.log10(amount) - 7.0) * 6.0))


def score_turnover(turnover_rate: float | None) -> float:
    if turnover_rate is None:
        return 0.0
    if turnover_rate < 1:
        return 1.0
    if 1 <= turnover_rate <= 8:
        return 3.0 + turnover_rate
    if 8 < turnover_rate <= 18:
        return 10.0
    return 5.0


def score_valuation(pe_dynamic: float | None, pb: float | None) -> float:
    score = 0.0
    if pe_dynamic is not None:
        if 0 < pe_dynamic <= 35:
            score += 8.0
        elif 35 < pe_dynamic <= 80:
            score += 4.0
        elif pe_dynamic < 0:
            score -= 6.0
    if pb is not None:
        if 0 < pb <= 6:
            score += 6.0
        elif 6 < pb <= 12:
            score += 2.0
        elif pb > 20:
            score -= 3.0
    return score


def rating_for_score(score: float) -> str:
    if score >= 72:
        return "强关注"
    if score >= 58:
        return "观察"
    if score >= 45:
        return "轻仓跟踪"
    return "暂不优先"


def load_portfolio(path: Path | None) -> list[PortfolioPosition]:
    if path is None or not path.exists():
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("positions", payload) if isinstance(payload, dict) else payload
        return [position_from_mapping(row) for row in rows if isinstance(row, dict)]
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [position_from_mapping(row) for row in reader]


def position_from_mapping(row: dict) -> PortfolioPosition:
    code = str(row.get("code") or row.get("ticker") or row.get("股票代码") or "").strip()
    name = str(row.get("name") or row.get("公司") or row.get("股票名称") or "").strip()
    market = str(row.get("market") or row.get("市场") or "A股").strip() or "A股"
    notes = str(row.get("notes") or row.get("备注") or "").strip()
    return PortfolioPosition(
        code=code,
        name=name,
        market=market,
        shares=parse_float(row.get("shares") or row.get("持仓数量")),
        cost_price=parse_float(row.get("cost_price") or row.get("cost") or row.get("成本价")),
        market_value=parse_float(row.get("market_value") or row.get("value") or row.get("持仓市值")),
        notes=notes,
    )


def analyze_portfolio(
    positions: list[PortfolioPosition],
    companies: list[AShareRefreshCompany],
    quotes: dict[str, QuoteSnapshot],
    picks: list[DailyPickCandidate],
) -> list[PortfolioAnalysis]:
    by_code = {company.record.code: company for company in companies}
    picked_codes = {pick.company.record.code for pick in picks}
    analyses: list[PortfolioAnalysis] = []
    for position in positions:
        code = position.normalized_code
        company = by_code.get(code)
        quote = quotes.get(code)
        effective_shares = position.shares
        if effective_shares is None and position.market_value is not None:
            if quote and quote.price:
                effective_shares = position.market_value / quote.price
            elif position.cost_price:
                effective_shares = position.market_value / position.cost_price
        market_value = position.market_value
        pnl = None
        pnl_pct = None
        risks: list[str] = []
        if quote and quote.price is not None and effective_shares is not None:
            market_value = quote.price * effective_shares
        if quote and quote.price is not None and position.cost_price:
            pnl = (quote.price - position.cost_price) * (effective_shares or 0)
            pnl_pct = (quote.price - position.cost_price) / position.cost_price * 100
        if company is None:
            risks.append("持仓未匹配到 A股科技清单")
        if quote is None:
            risks.append("持仓行情缺失")
        if pnl_pct is not None and pnl_pct < -12:
            risks.append("浮亏超过 12%，需复核止损纪律")
        if pnl_pct is not None and pnl_pct > 25:
            risks.append("浮盈较高，需考虑分批止盈或回撤保护")
        analyses.append(
            PortfolioAnalysis(
                position=position,
                company=company,
                quote=quote,
                effective_shares=effective_shares,
                market_value=market_value,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                in_top_picks=code in picked_codes,
                risk_flags=tuple(risks),
            )
        )
    return analyses


def render_daily_report(
    picks: list[DailyPickCandidate],
    portfolio: list[PortfolioAnalysis],
    *,
    generated_at: str,
    universe_count: int,
    scanned_count: int,
    max_candidates: int | None,
    top_k: int,
    categories: tuple[str, ...],
) -> str:
    lines = [
        "# 每日量化推荐观察池",
        "",
        f"- 生成时间：{generated_at}",
        "- 市场：A股",
        f"- 股票清单总数：{universe_count}",
        f"- 本次扫描数量：{scanned_count}",
        f"- 输出 Top K：{top_k}",
        f"- 扫描模式：{'全量' if max_candidates is None else f'前 {max_candidates} 只'}",
        f"- 过滤分类：{'、'.join(categories) if categories else '全部 A股科技相关分类'}",
        f"- 资料标签：{QUANT_CANDIDATE_TAG} {MODEL_TAG} {THIRD_PARTY_TAG} {NEEDS_VERIFICATION_TAG} {NEEDS_REFRESH_TAG}",
        "- 定位：这是每日量化观察池，用于发现可能走强的候选股票；不是收益承诺，也不是无条件买入指令。",
        "- 核验提醒：行情、估值和公告线索来自第三方或自动刷新资料，必须结合交易所公告、公司披露、最新行情和你的持仓约束复核。",
        "",
        "## 评分方法",
        "",
        "- 主题方向：AI 算力、半导体、机器人、无人机、新能源/锂电、软件通信硬件等方向有主题权重。",
        "- 量价因子：涨跌幅、成交额、换手率用于识别短期资金关注度和流动性。",
        "- 估值约束：PE/PB 只做粗过滤，负 PE、高 PB 或缺失数据会降低可信度。",
        "- 风险惩罚：ST/风险警示、行情缺失、流动性偏低、涨幅过高会降低排序。",
        "",
        "## 今日候选 Top",
        "",
    ]
    if not picks:
        lines.extend([f"本次没有生成候选。tags: {NEEDS_REFRESH_TAG} {NEEDS_VERIFICATION_TAG}", ""])
    for index, pick in enumerate(picks, start=1):
        record = pick.company.record
        quote = pick.quote
        lines.extend(
            [
                f"### {index}. {record.name}（{record.secucode}）",
                "",
                f"- 评分：{pick.score:.2f}",
                f"- 等级：{pick.rating}",
                f"- 分类：{'、'.join(pick.company.categories) or '待核验'}",
                f"- 行业：{record.industry or '未披露'}",
                f"- 最新价：{format_number(quote.price, suffix=' 元') if quote else '待刷新'}",
                f"- 涨跌幅：{format_number(quote.pct_change, suffix='%') if quote else '待刷新'}",
                f"- 成交额：{format_money(quote.amount) if quote else '待刷新'}",
                f"- 换手率：{format_number(quote.turnover_rate, suffix='%') if quote else '待刷新'}",
                f"- 估值：PE={format_number(quote.pe_dynamic) if quote else '待刷新'}，PB={format_number(quote.pb) if quote else '待刷新'}",
                f"- 推荐逻辑：{'；'.join(pick.reasons)}",
                f"- 风险提示：{'；'.join(pick.risk_flags) if pick.risk_flags else '未触发明显规则风险，但仍需核验公告和行情'}",
                f"- tags: {QUANT_CANDIDATE_TAG} {MODEL_TAG} {THIRD_PARTY_TAG} {NEEDS_VERIFICATION_TAG} {NEEDS_REFRESH_TAG}",
                "",
            ]
        )
    lines.extend(["## 持仓分析", ""])
    if not portfolio:
        lines.extend(["未提供持仓文件。可以用 CSV/JSON 提供 code、name、shares、cost_price、notes 字段。", ""])
    for item in portfolio:
        position = item.position
        display_name = position.name or (item.company.record.name if item.company else position.code)
        lines.extend(
            [
                f"### {display_name}（{position.code}）",
                "",
                f"- 持仓数量：{format_number(position.shares) if position.shares is not None else '未提供'}",
                f"- 估算数量：{format_number(item.effective_shares) if position.shares is None and item.effective_shares is not None else '不适用'}",
                f"- 成本价：{format_number(position.cost_price, suffix=' 元') if position.cost_price is not None else '未提供'}",
                f"- 名义持仓：{format_money(position.market_value) if position.market_value is not None else '未提供'}",
                f"- 最新价：{format_number(item.quote.price, suffix=' 元') if item.quote else '待刷新'}",
                f"- 市值：{format_money(item.market_value) if item.market_value is not None else '待核验'}",
                f"- 浮动盈亏：{format_money(item.unrealized_pnl) if item.unrealized_pnl is not None else '待核验'}",
                f"- 浮动盈亏率：{format_number(item.unrealized_pnl_pct, suffix='%') if item.unrealized_pnl_pct is not None else '待核验'}",
                f"- 是否进入今日候选：{'是' if item.in_top_picks else '否'}",
                f"- 持仓风险：{'；'.join(item.risk_flags) if item.risk_flags else '未触发规则风险'}",
                f"- 备注：{position.notes or '无'}",
                f"- tags: {PORTFOLIO_TAG} {NEEDS_VERIFICATION_TAG} {NEEDS_REFRESH_TAG}",
                "",
            ]
        )
    lines.extend(
        [
            "## 使用建议",
            "",
            "- 强关注：适合加入次日重点观察池，继续核验公告、财报、行业催化和盘中量价。",
            "- 观察：等待更明确的成交额、趋势或公告催化。",
            "- 轻仓跟踪：只适合作为备选，不建议忽略风险直接加仓。",
            "- 暂不优先：数据缺失、风险较高或量价不支持，先放低优先级。",
            "- 所有推荐都必须叠加你的仓位、回撤承受能力、交易纪律和最新信息。",
            "",
        ]
    )
    return "\n".join(lines)


def normalize_code(value: str) -> str:
    cleaned = value.strip().upper()
    if "." in cleaned:
        cleaned = cleaned.split(".", 1)[0]
    return cleaned.zfill(6) if cleaned.isdigit() and len(cleaned) < 6 else cleaned


def parse_float(value) -> float | None:
    if value in {None, "", "-", "N/D"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

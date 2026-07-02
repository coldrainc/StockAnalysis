from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from stock_agent.market.a_share_tech import (
    AShareTechRecord,
    AnnouncementItem,
    DELISTING_RISK_TAG,
    EASTMONEY_ANNOUNCEMENT_URL,
    NEEDS_REFRESH_TAG,
    NEEDS_VERIFICATION_TAG,
    OFFICIAL_TAG,
    SPECIAL_TREATMENT_TAG,
    THIRD_PARTY_TAG,
    exchange_from_cninfo_org_id,
    fetch_announcements,
    fetch_json,
    format_money,
    format_number,
    is_special_treatment_name,
    parse_float,
    parse_int,
    safe_filename,
    secid_market_from_exchange,
    short_date,
    text_value,
    utc_now,
)


EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
EASTMONEY_ULIST_QUOTE_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"


@dataclass(frozen=True)
class AShareRefreshCompany:
    record: AShareTechRecord
    categories: tuple[str, ...]
    document_path: str


@dataclass(frozen=True)
class QuoteSnapshot:
    price: float | None = None
    pct_change: float | None = None
    change: float | None = None
    volume: float | None = None
    amount: float | None = None
    amplitude: float | None = None
    high: float | None = None
    low: float | None = None
    open_price: float | None = None
    prev_close: float | None = None
    turnover_rate: float | None = None
    pe_dynamic: float | None = None
    pb: float | None = None
    total_market_cap: float | None = None
    float_market_cap: float | None = None
    quote_time: str = ""
    source: str = EASTMONEY_QUOTE_URL
    tags: tuple[str, ...] = (THIRD_PARTY_TAG, NEEDS_VERIFICATION_TAG, NEEDS_REFRESH_TAG)

    @property
    def tag_text(self) -> str:
        return " ".join(self.tags)


@dataclass(frozen=True)
class FinancialSummary:
    title: str
    content: str
    as_of: str
    source: str
    reliability: str = "configured_financials_api"
    tags: tuple[str, ...] = (OFFICIAL_TAG, NEEDS_REFRESH_TAG)

    @property
    def tag_text(self) -> str:
        return " ".join(self.tags)


@dataclass(frozen=True)
class RefreshedCompany:
    company: AShareRefreshCompany
    quote: QuoteSnapshot | None
    announcements: tuple[AnnouncementItem, ...]
    financial: FinancialSummary | None
    generated_at: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class AShareTechRefreshResult:
    output_dir: Path
    dynamic_dir: Path
    generated_at: str
    total_manifest_count: int
    refreshed_count: int
    quote_count: int
    announcement_count: int
    financial_count: int
    offset: int
    max_companies: int | None


def refresh_a_share_tech_dynamic(
    output_dir: Path,
    *,
    max_companies: int | None = 200,
    offset: int = 0,
    include_quotes: bool = True,
    include_announcements: bool = True,
    include_financials: bool = True,
    announcement_limit: int = 5,
    workers: int = 8,
    timeout: float = 8.0,
) -> AShareTechRefreshResult:
    companies = load_manifest_companies(output_dir / "manifest.json")
    start = max(offset, 0)
    end = None if max_companies is None else start + max(max_companies, 0)
    selected = companies[start:end]
    generated_at = utc_now()

    refreshed: list[RefreshedCompany] = []
    if selected:
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
            futures = {
                executor.submit(
                    refresh_single_company,
                    company,
                    include_quotes=include_quotes,
                    include_announcements=include_announcements,
                    include_financials=include_financials,
                    announcement_limit=announcement_limit,
                    timeout=timeout,
                    generated_at=generated_at,
                ): company
                for company in selected
            }
            for future in as_completed(futures):
                try:
                    refreshed.append(future.result())
                except Exception as exc:
                    company = futures[future]
                    refreshed.append(
                        RefreshedCompany(
                            company=company,
                            quote=None,
                            announcements=(),
                            financial=None,
                            generated_at=generated_at,
                            errors=(f"{exc.__class__.__name__}: {exc}",),
                        )
                    )

    refreshed.sort(key=lambda item: item.company.record.code)
    dynamic_dir = output_dir / "dynamic"
    write_refresh_markdown(
        dynamic_dir,
        refreshed,
        generated_at=generated_at,
        total_manifest_count=len(companies),
        offset=start,
        max_companies=max_companies,
        include_quotes=include_quotes,
        include_announcements=include_announcements,
        include_financials=include_financials,
    )
    return AShareTechRefreshResult(
        output_dir=output_dir,
        dynamic_dir=dynamic_dir,
        generated_at=generated_at,
        total_manifest_count=len(companies),
        refreshed_count=len(refreshed),
        quote_count=sum(1 for item in refreshed if item.quote is not None),
        announcement_count=sum(len(item.announcements) for item in refreshed),
        financial_count=sum(1 for item in refreshed if item.financial is not None),
        offset=start,
        max_companies=max_companies,
    )


def load_manifest_companies(manifest_path: Path) -> list[AShareRefreshCompany]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"A股科技逐股清单不存在：{manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    companies: list[AShareRefreshCompany] = []
    for item in payload.get("documents", []):
        code = text_value(item.get("code"))
        name = text_value(item.get("name"))
        secucode = text_value(item.get("secucode"))
        if not code or not name:
            continue
        exchange = secucode.split(".")[-1] if "." in secucode else exchange_from_cninfo_org_id("", code)
        tags = [THIRD_PARTY_TAG, NEEDS_VERIFICATION_TAG, NEEDS_REFRESH_TAG]
        if is_special_treatment_name(name):
            tags.extend([SPECIAL_TREATMENT_TAG, DELISTING_RISK_TAG])
        companies.append(
            AShareRefreshCompany(
                record=AShareTechRecord(
                    code=code,
                    name=name,
                    exchange=exchange,
                    secid_market=secid_market_from_exchange(exchange),
                    industry=text_value(item.get("industry")),
                    region="",
                    concepts=tuple(item.get("categories") or ()),
                    match_reasons=tuple(item.get("match_reasons") or ()),
                    tags=tuple(dict.fromkeys(tags)),
                    source=str(manifest_path),
                ),
                categories=tuple(item.get("categories") or ()),
                document_path=text_value(item.get("path")),
            )
        )
    return sorted(companies, key=lambda item: item.record.code)


def refresh_single_company(
    company: AShareRefreshCompany,
    *,
    include_quotes: bool,
    include_announcements: bool,
    include_financials: bool,
    announcement_limit: int,
    timeout: float,
    generated_at: str,
) -> RefreshedCompany:
    errors: list[str] = []
    quote = None
    announcements: tuple[AnnouncementItem, ...] = ()
    financial = None

    if include_quotes:
        try:
            quote = fetch_quote_snapshot(company.record, timeout=timeout)
        except Exception as exc:
            errors.append(f"行情刷新失败：{exc.__class__.__name__}: {exc}")
    if include_announcements:
        try:
            announcements = tuple(
                fetch_announcements(
                    company.record,
                    limit=announcement_limit,
                    timeout=timeout,
                )
            )
        except Exception as exc:
            errors.append(f"公告刷新失败：{exc.__class__.__name__}: {exc}")
    if include_financials:
        try:
            financial = fetch_financial_summary(company.record, timeout=timeout)
        except Exception as exc:
            errors.append(f"财务刷新失败：{exc.__class__.__name__}: {exc}")

    return RefreshedCompany(
        company=company,
        quote=quote,
        announcements=announcements,
        financial=financial,
        generated_at=generated_at,
        errors=tuple(errors),
    )


def fetch_quote_snapshot(record: AShareTechRecord, timeout: float = 8.0) -> QuoteSnapshot | None:
    params = {
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "secid": record.secid,
        "fields": (
            "f43,f44,f45,f46,f47,f48,f50,f57,f58,f59,f60,f86,"
            "f116,f117,f162,f167,f168,f169,f170,f171"
        ),
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    try:
        payload = fetch_json(
            EASTMONEY_QUOTE_URL,
            params=params,
            timeout=timeout,
            headers=headers,
        )
        data = payload.get("data") or {}
    except Exception:
        data = {}
    if not data:
        return fetch_quote_snapshot_from_ulist(record, timeout=timeout, headers=headers)
    return quote_snapshot_from_stock_get(data)


def quote_snapshot_from_stock_get(data: dict) -> QuoteSnapshot | None:
    if not data:
        return None
    precision = parse_int(data.get("f59"), default=2)
    return QuoteSnapshot(
        price=scaled_eastmoney_number(data.get("f43"), precision),
        high=scaled_eastmoney_number(data.get("f44"), precision),
        low=scaled_eastmoney_number(data.get("f45"), precision),
        open_price=scaled_eastmoney_number(data.get("f46"), precision),
        volume=parse_float(data.get("f47")),
        amount=parse_float(data.get("f48")),
        amplitude=scaled_eastmoney_number(data.get("f171"), precision),
        prev_close=scaled_eastmoney_number(data.get("f60"), precision),
        total_market_cap=parse_float(data.get("f116")),
        float_market_cap=parse_float(data.get("f117")),
        pe_dynamic=scaled_eastmoney_number(data.get("f162"), precision),
        pb=scaled_eastmoney_number(data.get("f167"), precision),
        turnover_rate=scaled_eastmoney_number(data.get("f168"), precision),
        change=scaled_eastmoney_number(data.get("f169"), precision),
        pct_change=scaled_eastmoney_number(data.get("f170"), precision),
        quote_time=format_quote_time(data.get("f86")),
    )


def fetch_quote_snapshot_from_ulist(
    record: AShareTechRecord,
    *,
    timeout: float = 8.0,
    headers: dict[str, str] | None = None,
) -> QuoteSnapshot | None:
    payload = fetch_json(
        EASTMONEY_ULIST_QUOTE_URL,
        params={
            "fltt": 2,
            "secids": record.secid,
            "fields": "f12,f14,f2,f3,f4,f5,f6,f7,f8,f9,f15,f16,f17,f18,f20,f21,f23,f124",
        },
        timeout=timeout,
        headers=headers,
    )
    rows = (payload.get("data") or {}).get("diff") or []
    if not rows:
        return None
    return quote_snapshot_from_ulist_row(rows[0])


def quote_snapshot_from_ulist_row(row: dict) -> QuoteSnapshot:
    return QuoteSnapshot(
        price=parse_float(row.get("f2")),
        pct_change=parse_float(row.get("f3")),
        change=parse_float(row.get("f4")),
        volume=parse_float(row.get("f5")),
        amount=parse_float(row.get("f6")),
        amplitude=parse_float(row.get("f7")),
        high=parse_float(row.get("f15")),
        low=parse_float(row.get("f16")),
        open_price=parse_float(row.get("f17")),
        prev_close=parse_float(row.get("f18")),
        turnover_rate=parse_float(row.get("f8")),
        pe_dynamic=parse_float(row.get("f9")),
        pb=parse_float(row.get("f23")),
        total_market_cap=parse_float(row.get("f20")),
        float_market_cap=parse_float(row.get("f21")),
        quote_time=format_quote_time(row.get("f124")),
        source=EASTMONEY_ULIST_QUOTE_URL,
    )


def fetch_financial_summary(
    record: AShareTechRecord,
    *,
    timeout: float = 8.0,
) -> FinancialSummary | None:
    api_url = os.getenv("STOCK_FINANCIALS_API_URL", "").strip()
    if not api_url:
        return None
    try:
        response = requests.get(
            api_url,
            params={
                "ticker": record.secucode,
                "code": record.code,
                "name": record.name,
                "market": "A股",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None

    content = text_value(payload.get("summary") or payload.get("content"))
    if not content:
        return None
    return FinancialSummary(
        title=text_value(payload.get("title")) or f"{record.name} 财务数据摘要",
        content=content,
        as_of=text_value(payload.get("as_of") or payload.get("published_at")) or utc_now(),
        source=text_value(payload.get("url")) or api_url,
        reliability=text_value(payload.get("reliability")) or "configured_financials_api",
        tags=tuple(payload.get("tags") or (OFFICIAL_TAG, NEEDS_REFRESH_TAG)),
    )


def write_refresh_markdown(
    dynamic_dir: Path,
    refreshed: list[RefreshedCompany],
    *,
    generated_at: str,
    total_manifest_count: int,
    offset: int,
    max_companies: int | None,
    include_quotes: bool,
    include_announcements: bool,
    include_financials: bool,
) -> None:
    companies_dir = dynamic_dir / "companies"
    companies_dir.mkdir(parents=True, exist_ok=True)
    for item in refreshed:
        record = item.company.record
        path = companies_dir / f"{record.code}_{safe_filename(record.name)}_动态刷新.md"
        path.write_text(render_refreshed_company(item), encoding="utf-8")

    status = render_refresh_status(
        refreshed,
        generated_at=generated_at,
        total_manifest_count=total_manifest_count,
        offset=offset,
        max_companies=max_companies,
        include_quotes=include_quotes,
        include_announcements=include_announcements,
        include_financials=include_financials,
    )
    (dynamic_dir / "refresh_status.md").write_text(status, encoding="utf-8")


def render_refresh_status(
    refreshed: list[RefreshedCompany],
    *,
    generated_at: str,
    total_manifest_count: int,
    offset: int,
    max_companies: int | None,
    include_quotes: bool,
    include_announcements: bool,
    include_financials: bool,
) -> str:
    quote_count = sum(1 for item in refreshed if item.quote is not None)
    announcement_count = sum(len(item.announcements) for item in refreshed)
    financial_count = sum(1 for item in refreshed if item.financial is not None)
    mode = "全量" if max_companies is None else f"批次 offset={offset}, max={max_companies}"
    lines = [
        "# A股科技逐股动态刷新状态",
        "",
        f"- 生成时间：{generated_at}",
        f"- 清单总数：{total_manifest_count}",
        f"- 本次刷新模式：{mode}",
        f"- 本次覆盖股票数：{len(refreshed)}",
        f"- 行情快照写入数：{quote_count}",
        f"- 公告标题写入数：{announcement_count}",
        f"- 财务摘要写入数：{financial_count}",
        f"- 刷新开关：行情={include_quotes}；公告={include_announcements}；财务={include_financials}",
        f"- 资料标签：{THIRD_PARTY_TAG} {OFFICIAL_TAG} {NEEDS_VERIFICATION_TAG} {NEEDS_REFRESH_TAG}",
        "- 核验提醒：第三方行情和公告聚合只作为线索；回答时若引用动态刷新资料，仍需提示核验交易所公告原文、公司披露和最新行情。",
        "",
        "## 本次刷新股票",
        "",
    ]
    if not refreshed:
        lines.extend(
            [
                "本次没有刷新任何股票。",
                "",
                f"tags: {NEEDS_REFRESH_TAG} {NEEDS_VERIFICATION_TAG}",
                "",
            ]
        )
        return "\n".join(lines)
    for item in refreshed:
        record = item.company.record
        status = []
        status.append("行情OK" if item.quote else "行情缺失")
        status.append(f"公告{len(item.announcements)}条" if item.announcements else "公告缺失")
        status.append("财务OK" if item.financial else "财务缺失")
        if item.errors:
            status.append("有错误")
        lines.append(f"- {record.name}（{record.secucode}）：{'；'.join(status)}")
    lines.append("")
    return "\n".join(lines)


def render_refreshed_company(item: RefreshedCompany) -> str:
    record = item.company.record
    return "\n".join(
        [
            f"# {record.name}（{record.secucode}）A股动态刷新资料",
            "",
            "## RAG 元数据",
            "",
            f"- 公司：{record.name}",
            f"- 股票代码：{record.secucode}",
            "- 市场分类：A股",
            f"- 行业：{record.industry or '未披露'}",
            f"- 科技细分分类：{'、'.join(item.company.categories) or '待核验'}",
            f"- 原始逐股文档：{item.company.document_path or '待核验'}",
            f"- 刷新时间：{item.generated_at}",
            f"- 资料标签：{record.tag_text} {THIRD_PARTY_TAG} {NEEDS_VERIFICATION_TAG} {NEEDS_REFRESH_TAG}",
            "- 回答提醒：本文件为动态刷新线索，若行情、公告或财务字段缺失，回答时必须明确说明缺失并提示核验。",
            "",
            "## 行情与估值刷新快照",
            "",
            render_quote(item.quote),
            "",
            "## 最新公告线索",
            "",
            render_refresh_announcements(item.announcements),
            "",
            "## 财务数据刷新摘要",
            "",
            render_financial(item.financial),
            "",
            "## 刷新异常与存疑标签",
            "",
            render_errors(item.errors),
            "",
        ]
    )


def render_quote(quote: QuoteSnapshot | None) -> str:
    if quote is None:
        return "\n".join(
            [
                f"- 当前未取得行情快照。tags: {NEEDS_REFRESH_TAG} {NEEDS_VERIFICATION_TAG}",
                "- 分析价格、估值、市值、换手和成交额时必须重新刷新或使用权威行情源。",
            ]
        )
    return "\n".join(
        [
            f"- 数据时间：{quote.quote_time or '待核验'}",
            f"- 数据来源：{quote.source}",
            f"- 资料标签：{quote.tag_text}",
            f"- 最新价：{format_number(quote.price, suffix=' 元')}",
            f"- 涨跌幅：{format_number(quote.pct_change, suffix='%')}",
            f"- 涨跌额：{format_number(quote.change, suffix=' 元')}",
            f"- 今开/最高/最低/昨收：{format_number(quote.open_price)} / {format_number(quote.high)} / {format_number(quote.low)} / {format_number(quote.prev_close)}",
            f"- 成交量：{format_number(quote.volume, divisor=10000, suffix=' 万手/股口径待核验')}",
            f"- 成交额：{format_money(quote.amount)}",
            f"- 振幅：{format_number(quote.amplitude, suffix='%')}",
            f"- 换手率：{format_number(quote.turnover_rate, suffix='%')}",
            f"- 动态市盈率：{format_number(quote.pe_dynamic)}",
            f"- 市净率：{format_number(quote.pb)}",
            f"- 总市值：{format_money(quote.total_market_cap)}",
            f"- 流通市值：{format_money(quote.float_market_cap)}",
            "- 口径提示：东方财富行情字段为第三方快照，盘中、复权和单位口径需核验。",
        ]
    )


def render_refresh_announcements(announcements: tuple[AnnouncementItem, ...]) -> str:
    if not announcements:
        return "\n".join(
            [
                f"- 当前未取得公告标题。tags: {NEEDS_REFRESH_TAG} {NEEDS_VERIFICATION_TAG}",
                "- 回答催化剂、重大事项或风险警示时必须重新查询交易所公告原文。",
            ]
        )
    lines = [
        f"- 数据来源：{EASTMONEY_ANNOUNCEMENT_URL}",
        "- 以下为公告聚合标题，只作为进入交易所公告原文核验的线索：",
    ]
    for announcement in announcements:
        columns = "、".join(announcement.columns) if announcement.columns else "未分类"
        lines.append(
            f"- {short_date(announcement.notice_date) or announcement.notice_date or '日期待核验'}："
            f"{announcement.title}（栏目：{columns}，art_code：{announcement.art_code or '待核验'}，"
            f"tags: {' '.join(announcement.tags)}）"
        )
    return "\n".join(lines)


def render_financial(financial: FinancialSummary | None) -> str:
    if financial is None:
        return "\n".join(
            [
                f"- 当前未取得财务摘要。tags: {NEEDS_REFRESH_TAG} {NEEDS_VERIFICATION_TAG}",
                "- 回答收入、利润、现金流、资产负债、估值倍数时必须结合最新财报或配置权威财务数据 API。",
            ]
        )
    return "\n".join(
        [
            f"- 标题：{financial.title}",
            f"- 数据日期：{financial.as_of}",
            f"- 数据来源：{financial.source}",
            f"- 资料可信度：{financial.reliability}",
            f"- 资料标签：{financial.tag_text}",
            "",
            financial.content.strip(),
        ]
    )


def render_errors(errors: tuple[str, ...]) -> str:
    if not errors:
        return f"- 本次刷新没有捕获到异常。tags: {NEEDS_REFRESH_TAG}"
    return "\n".join(f"- {error} [tags: {NEEDS_REFRESH_TAG} {NEEDS_VERIFICATION_TAG}]" for error in errors)


def scaled_eastmoney_number(value, precision: int) -> float | None:
    number = parse_float(value)
    if number is None:
        return None
    raw = text_value(value)
    if "." in raw:
        return number
    if abs(number) < 100:
        return number
    divisor = 10 ** max(precision, 0)
    return number / divisor


def format_quote_time(value) -> str:
    raw = text_value(value)
    if not raw:
        return ""
    if len(raw) == 14 and raw.isdigit():
        return (
            f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]} "
            f"{raw[8:10]}:{raw[10:12]}:{raw[12:14]}"
        )
    try:
        timestamp = int(float(raw))
    except ValueError:
        return raw
    if timestamp <= 0:
        return raw
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

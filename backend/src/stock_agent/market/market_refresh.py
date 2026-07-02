from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

from stock_agent.market.stock_universe import (
    NEEDS_REFRESH_TAG,
    NEEDS_VERIFICATION_TAG,
    OFFICIAL_TAG,
    THIRD_PARTY_TAG,
    USER_SUPPLIED_TAG,
    CompanyRecord,
)


@dataclass(frozen=True)
class KnowledgeItem:
    title: str
    category: str
    company: str
    ticker: str
    market: str
    content: str
    source: str
    published_at: str
    reliability: str
    tags: tuple[str, ...]

    @property
    def tag_text(self) -> str:
        return " ".join(self.tags)


def refresh_dynamic_knowledge(
    output_dir: Path,
    records: Iterable[CompanyRecord],
    max_companies: int = 20,
    include_quotes: bool = True,
    include_announcements: bool = True,
    include_financials: bool = True,
    timeout: float = 8.0,
) -> list[KnowledgeItem]:
    dynamic_dir = output_dir / "dynamic"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    selected = list(records)[:max_companies]
    items: list[KnowledgeItem] = []
    if include_quotes:
        items.extend(fetch_quote_items(selected, timeout=timeout))
    if include_announcements:
        items.extend(fetch_announcement_items(selected, timeout=timeout))
    if include_financials:
        items.extend(fetch_financial_items(selected, timeout=timeout))
    write_dynamic_markdown(dynamic_dir, items, selected)
    return items


def fetch_quote_items(records: list[CompanyRecord], timeout: float = 8.0) -> list[KnowledgeItem]:
    items: list[KnowledgeItem] = []
    for record in records:
        if record.ticker == "PRIVATE":
            continue
        item = fetch_stooq_quote(record, timeout=timeout)
        if item:
            items.append(item)
    return items


def fetch_stooq_quote(record: CompanyRecord, timeout: float = 8.0) -> KnowledgeItem | None:
    symbol = stooq_symbol(record)
    if not symbol:
        return None
    url = f"https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        values = lines[1].split(",")
        row = dict(zip(header, values))
        if row.get("Close") in {None, "N/D"}:
            return None
    except requests.RequestException:
        return None
    published_at = " ".join(part for part in [row.get("Date", ""), row.get("Time", "")] if part)
    content = (
        f"{record.name} 最新行情快照：开盘 {row.get('Open')}，最高 {row.get('High')}，"
        f"最低 {row.get('Low')}，收盘 {row.get('Close')}，成交量 {row.get('Volume')}。"
    )
    return KnowledgeItem(
        title=f"{record.name} 行情快照",
        category="行情",
        company=record.name,
        ticker=record.ticker,
        market=record.market,
        content=content,
        source=url,
        published_at=published_at or utc_now(),
        reliability="third_party_market_data",
        tags=(THIRD_PARTY_TAG, NEEDS_VERIFICATION_TAG),
    )


def fetch_announcement_items(
    records: list[CompanyRecord], timeout: float = 8.0
) -> list[KnowledgeItem]:
    api_url = os.getenv("STOCK_ANNOUNCEMENT_API_URL", "").strip()
    if not api_url:
        return []
    items: list[KnowledgeItem] = []
    for record in records:
        try:
            response = requests.get(
                api_url,
                params={"ticker": record.ticker, "name": record.name},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, json.JSONDecodeError):
            continue
        for entry in payload.get("items", [])[:5]:
            items.append(
                KnowledgeItem(
                    title=str(entry.get("title", f"{record.name} 公告")),
                    category="公告",
                    company=record.name,
                    ticker=record.ticker,
                    market=record.market,
                    content=str(entry.get("summary") or entry.get("content") or ""),
                    source=str(entry.get("url") or api_url),
                    published_at=str(entry.get("published_at") or utc_now()),
                    reliability="configured_announcement_api",
                    tags=(OFFICIAL_TAG, NEEDS_REFRESH_TAG),
                )
            )
    return [item for item in items if item.content.strip()]


def fetch_financial_items(records: list[CompanyRecord], timeout: float = 8.0) -> list[KnowledgeItem]:
    api_url = os.getenv("STOCK_FINANCIALS_API_URL", "").strip()
    if not api_url:
        return []
    items: list[KnowledgeItem] = []
    for record in records:
        try:
            response = requests.get(
                api_url,
                params={"ticker": record.ticker, "name": record.name},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, json.JSONDecodeError):
            continue
        summary = payload.get("summary") or payload.get("content")
        if not summary:
            continue
        items.append(
            KnowledgeItem(
                title=f"{record.name} 财务数据摘要",
                category="财务数据",
                company=record.name,
                ticker=record.ticker,
                market=record.market,
                content=str(summary),
                source=str(payload.get("url") or api_url),
                published_at=str(payload.get("as_of") or utc_now()),
                reliability="configured_financials_api",
                tags=(OFFICIAL_TAG, NEEDS_REFRESH_TAG),
            )
        )
    return items


def write_manual_item(output_dir: Path, payload: dict) -> Path:
    item = KnowledgeItem(
        title=str(payload.get("title", "手动资料")),
        category=str(payload.get("category", "手动资料")),
        company=str(payload.get("company", "UNKNOWN")),
        ticker=str(payload.get("ticker", "UNKNOWN")),
        market=str(payload.get("market", "UNKNOWN")),
        content=str(payload.get("content", "")),
        source=str(payload.get("source", "manual")),
        published_at=str(payload.get("published_at", utc_now())),
        reliability=str(payload.get("reliability", "user_supplied")),
        tags=tuple(payload.get("tags", [USER_SUPPLIED_TAG, NEEDS_VERIFICATION_TAG])),
    )
    dynamic_dir = output_dir / "dynamic"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    path = dynamic_dir / safe_filename(f"manual_{item.company}_{item.title}.md")
    path.write_text(render_item(item), encoding="utf-8")
    return path


def write_dynamic_markdown(
    dynamic_dir: Path,
    items: list[KnowledgeItem],
    records: list[CompanyRecord],
) -> None:
    generated_at = utc_now()
    status_lines = [
        "# 动态数据刷新状态",
        "",
        f"- 生成时间：{generated_at}",
        f"- 覆盖公司数：{len(records)}",
        f"- 写入资料数：{len(items)}",
        "- 资料类型：公告、行情、财务数据",
        "- 标签说明：#official_or_exchange 表示官方或交易所来源；#third_party_dataset 表示第三方数据；#user_supplied 表示用户提供；#needs_refresh 表示需定时刷新；#needs_verification 表示资料真实性或口径需要核验。",
        "",
    ]
    if not items:
        status_lines.extend(
            [
                "## 刷新结果",
                "",
                "本次没有拉取到动态资料。",
                "",
                f"资料标签：{NEEDS_REFRESH_TAG} {NEEDS_VERIFICATION_TAG}",
                "提醒：回答时应说明当前动态数据缺失或未刷新，不能把静态公司库当作最新公告/行情/财报。",
                "",
            ]
        )
    (dynamic_dir / "refresh_status.md").write_text("\n".join(status_lines), encoding="utf-8")

    for item in items:
        filename = safe_filename(f"{item.category}_{item.company}_{item.published_at}_{item.title}.md")
        (dynamic_dir / filename).write_text(render_item(item), encoding="utf-8")


def render_item(item: KnowledgeItem) -> str:
    return "\n".join(
        [
            f"# {item.title}",
            "",
            f"- 公司：{item.company}",
            f"- 股票代码：{item.ticker}",
            f"- 市场：{item.market}",
            f"- 类别：{item.category}",
            f"- 发布时间/数据时间：{item.published_at}",
            f"- 数据来源：{item.source}",
            f"- 资料可信度：{item.reliability}",
            f"- 资料标签：{item.tag_text}",
            "",
            "## 内容",
            "",
            item.content.strip(),
            "",
            "## 回答提醒",
            "",
            "如果资料标签包含 #needs_verification 或 #needs_refresh，回答时必须说明该资料需要进一步核验或刷新，不能当作确定事实。",
            "",
        ]
    )


def stooq_symbol(record: CompanyRecord) -> str | None:
    ticker = record.ticker.lower()
    if ticker == "private":
        return None
    if ticker.endswith(".sh"):
        return ticker.replace(".sh", ".cn")
    if ticker.endswith(".sz"):
        return ticker.replace(".sz", ".cn")
    if ticker.endswith(".hk"):
        return ticker
    if "." in ticker:
        return None
    if record.market.upper().startswith("US"):
        return ticker
    return None


def safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return f"{safe[:160].strip('_') or 'item'}.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

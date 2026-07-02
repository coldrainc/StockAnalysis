from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import requests

from stock_agent.market.stock_universe import (
NEEDS_REFRESH_TAG,
    NEEDS_VERIFICATION_TAG,
    OFFICIAL_TAG,
    THIRD_PARTY_TAG,
)


EASTMONEY_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_COMPANY_SURVEY_URL = (
    "https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax"
)
EASTMONEY_ANNOUNCEMENT_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
CNINFO_STOCK_LIST_URL = "http://www.cninfo.com.cn/new/data/szse_stock.json"
SPECIAL_TREATMENT_TAG = "#special_treatment"
DELISTING_RISK_TAG = "#delisting_risk"

DEFAULT_TECH_INDUSTRY_KEYWORDS = (
    "半导体",
    "元件",
    "电子",
    "光学光电子",
    "消费电子",
    "电子化学品",
    "通信设备",
    "通信服务",
    "软件",
    "互联网服务",
    "计算机设备",
    "IT服务",
    "游戏",
    "数字媒体",
    "自动化设备",
    "仪器仪表",
    "电池",
    "光伏设备",
    "风电设备",
)

DEFAULT_TECH_CONCEPT_KEYWORDS = (
    "人工智能",
    "AI",
    "AIGC",
    "算力",
    "数据中心",
    "云计算",
    "大数据",
    "数据要素",
    "边缘计算",
    "机器人",
    "具身智能",
    "人形机器人",
    "减速器",
    "伺服",
    "无人机",
    "低空经济",
    "eVTOL",
    "芯片",
    "半导体",
    "集成电路",
    "存储芯片",
    "第三代半导体",
    "光刻",
    "EDA",
    "先进封装",
    "PCB",
    "CPO",
    "光模块",
    "6G",
    "5G",
    "卫星导航",
    "量子",
    "信创",
    "国产软件",
    "网络安全",
    "鸿蒙",
    "华为",
    "小米",
    "苹果",
    "消费电子",
    "MR",
    "虚拟现实",
    "增强现实",
    "OLED",
    "MiniLED",
    "电子纸",
    "智能穿戴",
    "智能驾驶",
    "无人驾驶",
    "汽车电子",
    "传感器",
    "物联网",
    "工业互联网",
    "智能制造",
    "锂电池",
    "固态电池",
    "储能",
    "光伏",
)

NAME_TECH_KEYWORDS = (
    "科技",
    "信息",
    "软件",
    "智能",
    "电子",
    "通信",
    "芯片",
    "微电",
    "数码",
)

WEAK_BUSINESS_TERMS = (
    "电子",
    "软件",
    "电池",
    "光伏",
    "AI",
    "仪器仪表",
)

CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("半导体与电子", ("半导体", "芯片", "集成电路", "光刻", "EDA", "先进封装", "元件", "电子", "PCB")),
    ("软件与信息服务", ("软件", "互联网服务", "IT服务", "信创", "国产软件", "网络安全", "数据要素")),
    ("AI算力与数据中心", ("人工智能", "AI", "AIGC", "算力", "数据中心", "云计算", "大数据", "边缘计算", "CPO", "光模块")),
    ("通信与网络", ("通信", "5G", "6G", "卫星导航", "物联网", "工业互联网")),
    ("机器人与自动化", ("机器人", "具身智能", "人形机器人", "减速器", "伺服", "自动化", "仪器仪表", "智能制造")),
    ("无人机与低空经济", ("无人机", "低空经济", "eVTOL")),
    ("智能汽车与硬件", ("智能驾驶", "无人驾驶", "汽车电子", "传感器", "消费电子", "智能穿戴", "MR", "虚拟现实", "增强现实")),
    ("新能源科技与锂电", ("锂电池", "固态电池", "储能", "光伏", "电池", "风电")),
)


@dataclass(frozen=True)
class AShareTechRecord:
    code: str
    name: str
    exchange: str
    secid_market: int
    industry: str
    region: str
    concepts: tuple[str, ...]
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
    match_reasons: tuple[str, ...] = ()
    tags: tuple[str, ...] = (THIRD_PARTY_TAG, NEEDS_VERIFICATION_TAG, NEEDS_REFRESH_TAG)
    source: str = EASTMONEY_CLIST_URL

    @property
    def secucode(self) -> str:
        return f"{self.code}.{self.exchange}"

    @property
    def f10_code(self) -> str:
        return f"{self.exchange}{self.code}"

    @property
    def secid(self) -> str:
        return f"{self.secid_market}.{self.code}"

    @property
    def tag_text(self) -> str:
        return " ".join(self.tags)


@dataclass(frozen=True)
class CompanyProfile:
    org_name: str = ""
    org_name_en: str = ""
    security_type: str = ""
    trade_market: str = ""
    em_industry: str = ""
    csrc_industry: str = ""
    chairman: str = ""
    president: str = ""
    legal_person: str = ""
    secretary: str = ""
    website: str = ""
    telephone: str = ""
    email: str = ""
    address: str = ""
    province: str = ""
    employees: str = ""
    register_capital: str = ""
    org_profile: str = ""
    business_scope: str = ""
    found_date: str = ""
    listing_date: str = ""
    issue_price: str = ""
    source: str = EASTMONEY_COMPANY_SURVEY_URL
    tags: tuple[str, ...] = (THIRD_PARTY_TAG, NEEDS_VERIFICATION_TAG, NEEDS_REFRESH_TAG)


@dataclass(frozen=True)
class AnnouncementItem:
    title: str
    notice_date: str
    columns: tuple[str, ...]
    art_code: str
    source: str = EASTMONEY_ANNOUNCEMENT_URL
    tags: tuple[str, ...] = (OFFICIAL_TAG, NEEDS_VERIFICATION_TAG, NEEDS_REFRESH_TAG)


@dataclass(frozen=True)
class StockAnalysisDocument:
    record: AShareTechRecord
    profile: CompanyProfile
    announcements: tuple[AnnouncementItem, ...]
    categories: tuple[str, ...]
    generated_at: str
    source_urls: tuple[str, ...]


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    generated_at: str
    total_a_share_count: int
    matched_count: int
    document_count: int
    category_counts: dict[str, int]
    data_sources: tuple[str, ...]
    tags: tuple[str, ...]
    source_mode: str = "eastmoney"
    fallback_reason: str = ""


def build_a_share_tech_knowledge_base(
    output_dir: Path,
    *,
    industry_keywords: Iterable[str] = DEFAULT_TECH_INDUSTRY_KEYWORDS,
    concept_keywords: Iterable[str] = DEFAULT_TECH_CONCEPT_KEYWORDS,
    max_companies: int | None = None,
    include_profiles: bool = True,
    include_announcements: bool = True,
    announcement_limit: int = 5,
    workers: int = 8,
    timeout: float = 8.0,
    source_mode: str = "auto",
) -> BuildResult:
    """Fetch A-share technology-related companies and write one Markdown file per stock."""

    generated_at = utc_now()
    all_records, resolved_source_mode, fallback_reason = fetch_a_share_records_with_fallback(
        timeout=timeout,
        source_mode=source_mode,
    )
    industry_terms = normalize_terms(industry_keywords)
    concept_terms = normalize_terms(concept_keywords)
    needs_profile_selection = resolved_source_mode == "cninfo"
    profile_cache: dict[str, CompanyProfile] = {}
    matched = (
        select_tech_records_by_profile(
            all_records,
            industry_terms=industry_terms,
            concept_terms=concept_terms,
            workers=workers,
            timeout=timeout,
            max_matches=max_companies,
            profile_cache=profile_cache,
        )
        if needs_profile_selection
        else select_tech_records(
            all_records,
            industry_keywords=industry_terms,
            concept_keywords=concept_terms,
        )
    )
    if max_companies is not None:
        matched = matched[: max(max_companies, 0)]

    documents = enrich_documents(
        matched,
        include_profiles=include_profiles,
        include_announcements=include_announcements,
        announcement_limit=announcement_limit,
        workers=workers,
        timeout=timeout,
        generated_at=generated_at,
        profile_cache=profile_cache,
    )
    write_analysis_documents(output_dir, documents, total_a_share_count=len(all_records))
    category_counts: dict[str, int] = {}
    for doc in documents:
        for category in doc.categories:
            category_counts[category] = category_counts.get(category, 0) + 1

    result = BuildResult(
        output_dir=output_dir,
        generated_at=generated_at,
        total_a_share_count=len(all_records),
        matched_count=len(matched),
        document_count=len(documents),
        category_counts=dict(sorted(category_counts.items())),
        data_sources=(
            EASTMONEY_CLIST_URL,
            CNINFO_STOCK_LIST_URL,
            EASTMONEY_COMPANY_SURVEY_URL,
            EASTMONEY_ANNOUNCEMENT_URL,
        ),
        tags=(THIRD_PARTY_TAG, OFFICIAL_TAG, NEEDS_VERIFICATION_TAG, NEEDS_REFRESH_TAG),
        source_mode=resolved_source_mode,
        fallback_reason=fallback_reason,
    )
    write_manifest(output_dir, result, documents)
    return result


def fetch_a_share_records_with_fallback(
    *,
    timeout: float = 8.0,
    source_mode: str = "auto",
) -> tuple[list[AShareTechRecord], str, str]:
    normalized = source_mode.lower()
    if normalized not in {"auto", "eastmoney", "cninfo"}:
        raise ValueError("source_mode must be auto, eastmoney or cninfo")
    if normalized in {"auto", "eastmoney"}:
        try:
            records = fetch_a_share_records(timeout=timeout)
            if records:
                return records, "eastmoney", ""
        except Exception as exc:
            if normalized == "eastmoney":
                raise
            reason = f"{exc.__class__.__name__}: {exc}"
        else:
            reason = "eastmoney returned no records"
    else:
        reason = ""
    return fetch_cninfo_a_share_records(timeout=timeout), "cninfo", reason


def fetch_a_share_records(timeout: float = 8.0, page_size: int = 500) -> list[AShareTechRecord]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            ),
            "Referer": "https://quote.eastmoney.com/",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
    )
    records: list[AShareTechRecord] = []
    page = 1
    total = None
    while True:
        payload = fetch_a_share_page(session, page=page, page_size=page_size, timeout=timeout)
        data = payload.get("data") or {}
        total = int(data.get("total") or total or 0)
        rows = list(data.get("diff") or [])
        if not rows:
            break
        records.extend(parse_a_share_row(row) for row in rows)
        if len(records) >= total:
            break
        page += 1
    return records


def fetch_cninfo_a_share_records(timeout: float = 8.0) -> list[AShareTechRecord]:
    payload = fetch_json(
        CNINFO_STOCK_LIST_URL,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "http://www.cninfo.com.cn/",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    records: list[AShareTechRecord] = []
    for row in payload.get("stockList", []):
        if row.get("category") != "A股":
            continue
        code = text_value(row.get("code"))
        name = text_value(row.get("zwjc"))
        if not code or not name:
            continue
        if is_delisted_name(name):
            continue
        exchange = exchange_from_cninfo_org_id(text_value(row.get("orgId")), code)
        tags = [THIRD_PARTY_TAG, NEEDS_VERIFICATION_TAG, NEEDS_REFRESH_TAG]
        if is_special_treatment_name(name):
            tags.extend([SPECIAL_TREATMENT_TAG, DELISTING_RISK_TAG])
        records.append(
            AShareTechRecord(
                code=code,
                name=name,
                exchange=exchange,
                secid_market=secid_market_from_exchange(exchange),
                industry="",
                region="",
                concepts=(),
                tags=tuple(dict.fromkeys(tags)),
                source=CNINFO_STOCK_LIST_URL,
            )
        )
    return deduplicate_records(records)


def fetch_a_share_page(
    session: requests.Session,
    *,
    page: int,
    page_size: int,
    timeout: float,
) -> dict:
    params = {
        "pn": page,
        "pz": page_size,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81,m:1+t:81",
        "fields": (
            "f12,f13,f14,f2,f3,f4,f5,f6,f7,f15,f16,f17,f18,"
            "f8,f9,f23,f20,f21,f100,f102,f103"
        ),
    }
    return fetch_json(
        EASTMONEY_CLIST_URL,
        params=params,
        timeout=timeout,
        session=session,
        headers={
            "User-Agent": session.headers["User-Agent"],
            "Referer": "https://quote.eastmoney.com/",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        },
    )


def parse_a_share_row(row: dict) -> AShareTechRecord:
    code = str(row.get("f12") or "").strip()
    secid_market = parse_int(row.get("f13"), default=0)
    exchange = exchange_from_market(secid_market, code)
    concepts = tuple(split_concepts(str(row.get("f103") or "")))
    return AShareTechRecord(
        code=code,
        name=str(row.get("f14") or "").strip(),
        exchange=exchange,
        secid_market=secid_market,
        industry=str(row.get("f100") or "").strip(),
        region=str(row.get("f102") or "").strip(),
        concepts=concepts,
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
    )


def select_tech_records(
    records: Iterable[AShareTechRecord],
    *,
    industry_keywords: Iterable[str] = DEFAULT_TECH_INDUSTRY_KEYWORDS,
    concept_keywords: Iterable[str] = DEFAULT_TECH_CONCEPT_KEYWORDS,
) -> list[AShareTechRecord]:
    industry_terms = normalize_terms(industry_keywords)
    concept_terms = normalize_terms(concept_keywords)
    selected: list[AShareTechRecord] = []
    for record in records:
        reasons = match_reasons(record, industry_terms, concept_terms)
        if not reasons:
            continue
        selected.append(
            AShareTechRecord(
                **{
                    **asdict(record),
                    "concepts": record.concepts,
                    "match_reasons": tuple(reasons),
                    "tags": record.tags,
                }
            )
        )
    return sorted(selected, key=lambda item: (item.industry, item.code, item.name))


def select_tech_records_by_profile(
    records: list[AShareTechRecord],
    *,
    industry_terms: tuple[str, ...],
    concept_terms: tuple[str, ...],
    workers: int,
    timeout: float,
    max_matches: int | None = None,
    profile_cache: dict[str, CompanyProfile] | None = None,
) -> list[AShareTechRecord]:
    worker_count = max(1, workers)
    selected_by_code: dict[str, AShareTechRecord] = {}
    batch_size = max(worker_count * 4, 8)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(profile_selected_record, record, industry_terms, concept_terms, timeout): record
                for record in batch
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    selected, profile = future.result()
                except Exception:
                    selected, profile = None, None
                if profile and profile_cache is not None:
                    profile_cache[record.code] = profile
                if selected:
                    selected_by_code[record.code] = selected
        if max_matches is not None and len(selected_by_code) >= max_matches:
            break
    return sorted(selected_by_code.values(), key=lambda item: (item.industry, item.code, item.name))


def profile_selected_record(
    record: AShareTechRecord,
    industry_terms: tuple[str, ...],
    concept_terms: tuple[str, ...],
    timeout: float,
) -> tuple[AShareTechRecord | None, CompanyProfile | None]:
    if is_delisted_name(record.name):
        return None, None
    profile = fetch_company_profile(record, timeout=timeout)
    industry = profile.em_industry or profile.csrc_industry or record.industry
    concepts = tuple(
        item
        for item in (
            profile.em_industry,
            profile.csrc_industry,
            extract_keyword_tags(profile.business_scope),
            extract_keyword_tags(profile.org_profile),
        )
        if item
    )
    enriched = AShareTechRecord(
        **{
            **asdict(record),
            "industry": industry,
            "concepts": concepts,
            "tags": record.tags,
        }
    )
    direct_reasons = match_reasons(enriched, industry_terms, concept_terms)
    business_reasons = match_business_reasons(profile, industry_terms + concept_terms)
    reasons = tuple(dict.fromkeys([*direct_reasons, *business_reasons]))
    if not is_strong_profile_match(direct_reasons, business_reasons):
        return None, profile
    selected = AShareTechRecord(
        **{
            **asdict(enriched),
            "concepts": concepts,
            "match_reasons": reasons,
            "tags": record.tags,
        }
    )
    return selected, profile


def match_reasons(
    record: AShareTechRecord,
    industry_terms: tuple[str, ...],
    concept_terms: tuple[str, ...],
) -> list[str]:
    reasons: list[str] = []
    industry_text = normalize_text(record.industry)
    concept_text = normalize_text("、".join(record.concepts))
    name_text = normalize_text(record.name)

    for term in industry_terms:
        normalized = normalize_text(term)
        if normalized and normalized in industry_text:
            reasons.append(f"行业命中：{term}")
    for term in concept_terms:
        normalized = normalize_text(term)
        if normalized and normalized in concept_text:
            reasons.append(f"概念命中：{term}")
    if not reasons:
        for term in NAME_TECH_KEYWORDS:
            normalized = normalize_text(term)
            if normalized and normalized in name_text:
                reasons.append(f"名称命中：{term}")
                break
    return tuple(dict.fromkeys(reasons))


def match_business_reasons(profile: CompanyProfile, terms: tuple[str, ...]) -> tuple[str, ...]:
    text = normalize_text(f"{profile.business_scope} {profile.org_profile} {profile.em_industry} {profile.csrc_industry}")
    reasons: list[str] = []
    for term in terms:
        normalized = normalize_text(term)
        if normalized and normalized in text:
            reasons.append(f"业务/F10命中：{term}")
    return tuple(dict.fromkeys(reasons))


def is_strong_profile_match(
    direct_reasons: Iterable[str],
    business_reasons: Iterable[str],
) -> bool:
    direct = tuple(direct_reasons)
    business = tuple(business_reasons)
    if any(reason.startswith(("行业命中", "名称命中")) for reason in direct):
        return True
    if any(reason.startswith("概念命中") for reason in direct):
        return True
    strong_business = [
        reason
        for reason in business
        if extract_reason_term(reason) not in WEAK_BUSINESS_TERMS
    ]
    if strong_business:
        return True
    return len(set(business)) >= 2


def extract_reason_term(reason: str) -> str:
    return reason.split("：", 1)[1] if "：" in reason else reason


def enrich_documents(
    records: list[AShareTechRecord],
    *,
    include_profiles: bool,
    include_announcements: bool,
    announcement_limit: int,
    workers: int,
    timeout: float,
    generated_at: str,
    profile_cache: dict[str, CompanyProfile] | None = None,
) -> list[StockAnalysisDocument]:
    if not records:
        return []

    worker_count = max(1, workers)
    documents_by_code: dict[str, StockAnalysisDocument] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                enrich_single_document,
                record,
                include_profiles=include_profiles,
                include_announcements=include_announcements,
                announcement_limit=announcement_limit,
                timeout=timeout,
                generated_at=generated_at,
                profile=profile_cache.get(record.code) if profile_cache else None,
            ): record
            for record in records
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                documents_by_code[record.code] = future.result()
            except Exception:
                documents_by_code[record.code] = StockAnalysisDocument(
                    record=record,
                    profile=CompanyProfile(),
                    announcements=(),
                    categories=classify_categories(record),
                    generated_at=generated_at,
                    source_urls=(EASTMONEY_CLIST_URL,),
                )
    return [documents_by_code[record.code] for record in records if record.code in documents_by_code]


def enrich_single_document(
    record: AShareTechRecord,
    *,
    include_profiles: bool,
    include_announcements: bool,
    announcement_limit: int,
    timeout: float,
    generated_at: str,
    profile: CompanyProfile | None = None,
) -> StockAnalysisDocument:
    profile = profile or (
        fetch_company_profile(record, timeout=timeout) if include_profiles else CompanyProfile()
    )
    announcements = (
        tuple(fetch_announcements(record, limit=announcement_limit, timeout=timeout))
        if include_announcements
        else ()
    )
    source_urls = [record.source or EASTMONEY_CLIST_URL]
    if include_profiles:
        source_urls.append(company_profile_url(record))
    if include_announcements:
        source_urls.append(EASTMONEY_ANNOUNCEMENT_URL)
    return StockAnalysisDocument(
        record=record,
        profile=profile,
        announcements=announcements,
        categories=classify_categories(record),
        generated_at=generated_at,
        source_urls=tuple(source_urls),
    )


def fetch_company_profile(record: AShareTechRecord, timeout: float = 8.0) -> CompanyProfile:
    payload = fetch_json(
        company_profile_url(record),
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://emweb.securities.eastmoney.com/PC_HSF10/pages/index.html?type=web&code={record.f10_code}",
        },
    )
    basics = first_dict(payload.get("jbzl"))
    issue = first_dict(payload.get("fxxg"))
    return CompanyProfile(
        org_name=text_value(basics.get("ORG_NAME")),
        org_name_en=text_value(basics.get("ORG_NAME_EN")),
        security_type=text_value(basics.get("SECURITY_TYPE")),
        trade_market=text_value(basics.get("TRADE_MARKET")),
        em_industry=text_value(basics.get("EM2016")),
        csrc_industry=text_value(basics.get("INDUSTRYCSRC1")),
        chairman=text_value(basics.get("CHAIRMAN")),
        president=text_value(basics.get("PRESIDENT")),
        legal_person=text_value(basics.get("LEGAL_PERSON")),
        secretary=text_value(basics.get("SECRETARY")),
        website=text_value(basics.get("ORG_WEB")),
        telephone=text_value(basics.get("ORG_TEL")),
        email=text_value(basics.get("ORG_EMAIL")),
        address=text_value(basics.get("ADDRESS") or basics.get("REG_ADDRESS")),
        province=text_value(basics.get("PROVINCE")),
        employees=text_value(basics.get("EMP_NUM")),
        register_capital=text_value(basics.get("REG_CAPITAL")),
        org_profile=clean_paragraph(text_value(basics.get("ORG_PROFILE"))),
        business_scope=clean_paragraph(text_value(basics.get("BUSINESS_SCOPE"))),
        found_date=text_value(issue.get("FOUND_DATE")),
        listing_date=text_value(issue.get("LISTING_DATE")),
        issue_price=text_value(issue.get("ISSUE_PRICE")),
        source=company_profile_url(record),
    )


def fetch_announcements(
    record: AShareTechRecord,
    *,
    limit: int = 5,
    timeout: float = 8.0,
) -> list[AnnouncementItem]:
    params = {
        "sr": -1,
        "page_size": max(limit, 1),
        "page_index": 1,
        "ann_type": "A",
        "client_source": "web",
        "stock_list": record.code,
    }
    payload = fetch_json(
        EASTMONEY_ANNOUNCEMENT_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/notice/"},
    )
    rows = ((payload.get("data") or {}).get("list") or [])[:limit]
    items: list[AnnouncementItem] = []
    for row in rows:
        columns = tuple(
            text_value(column.get("column_name"))
            for column in row.get("columns", [])
            if text_value(column.get("column_name"))
        )
        items.append(
            AnnouncementItem(
                title=text_value(row.get("title_ch") or row.get("title")),
                notice_date=text_value(row.get("notice_date") or row.get("display_time")),
                columns=columns,
                art_code=text_value(row.get("art_code")),
            )
        )
    return [item for item in items if item.title]


def write_analysis_documents(
    output_dir: Path,
    documents: list[StockAnalysisDocument],
    *,
    total_a_share_count: int,
) -> None:
    docs_dir = output_dir / "companies"
    category_dir = output_dir / "categories"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    if category_dir.exists():
        shutil.rmtree(category_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    category_dir.mkdir(parents=True, exist_ok=True)

    for document in documents:
        path = docs_dir / f"{document.record.code}_{safe_filename(document.record.name)}.md"
        path.write_text(render_analysis_document(document), encoding="utf-8")

    write_overview(output_dir, documents, total_a_share_count=total_a_share_count)
    write_category_indexes(category_dir, documents)


def write_overview(
    output_dir: Path,
    documents: list[StockAnalysisDocument],
    *,
    total_a_share_count: int,
) -> None:
    generated_at = documents[0].generated_at if documents else utc_now()
    lines = [
        "# A股科技相关股票完整分析库",
        "",
        f"- 生成时间：{generated_at}",
        f"- A股候选总数：{total_a_share_count}",
        f"- 科技相关命中数：{len(documents)}",
        "- 文档粒度：每只股票一个 Markdown 文件，适合直接进入 RAG/向量数据库。",
        "- 数据来源：东方财富公开行情列表、巨潮资讯全 A 股列表、东方财富 F10 公司概况、公告聚合接口。",
        f"- 资料标签：{THIRD_PARTY_TAG} {OFFICIAL_TAG} {NEEDS_VERIFICATION_TAG} {NEEDS_REFRESH_TAG}",
        "- 核验提醒：第三方行情/F10/公告聚合只作为研究线索；回答时如果引用带 #needs_verification 或 #needs_refresh 的内容，必须提示需要核验原公告、交易所披露与最新行情。",
        "",
        "## 分类统计",
        "",
    ]
    category_counts: dict[str, int] = {}
    for document in documents:
        for category in document.categories:
            category_counts[category] = category_counts.get(category, 0) + 1
    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}：{count} 家")
    lines.extend(["", "## 股票清单", ""])
    for document in documents:
        record = document.record
        categories = "、".join(document.categories)
        reasons = "；".join(record.match_reasons)
        lines.append(
            f"- {record.name}（{record.secucode}）：{record.industry}；分类：{categories}；命中：{reasons}"
        )
    (output_dir / "overview.md").write_text("\n".join(lines), encoding="utf-8")


def write_category_indexes(category_dir: Path, documents: list[StockAnalysisDocument]) -> None:
    by_category: dict[str, list[StockAnalysisDocument]] = {}
    for document in documents:
        for category in document.categories:
            by_category.setdefault(category, []).append(document)
    for category, items in sorted(by_category.items()):
        lines = [
            f"# {category}",
            "",
            f"- 覆盖股票数：{len(items)}",
            f"- 资料标签：{THIRD_PARTY_TAG} {NEEDS_VERIFICATION_TAG} {NEEDS_REFRESH_TAG}",
            "",
        ]
        for document in sorted(items, key=lambda item: item.record.code):
            record = document.record
            lines.append(
                f"- {record.name}（{record.secucode}）：{record.industry}；"
                f"概念：{'、'.join(record.concepts[:8]) or '未披露'}"
            )
        (category_dir / f"{safe_filename(category)}.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


def write_manifest(
    output_dir: Path,
    result: BuildResult,
    documents: list[StockAnalysisDocument],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": result.generated_at,
        "total_a_share_count": result.total_a_share_count,
        "matched_count": result.matched_count,
        "document_count": result.document_count,
        "category_counts": result.category_counts,
        "data_sources": list(result.data_sources),
        "tags": list(result.tags),
        "source_mode": result.source_mode,
        "fallback_reason": result.fallback_reason,
        "documents": [
            {
                "code": document.record.code,
                "name": document.record.name,
                "secucode": document.record.secucode,
                "industry": document.record.industry,
                "categories": list(document.categories),
                "match_reasons": list(document.record.match_reasons),
                "path": f"companies/{document.record.code}_{safe_filename(document.record.name)}.md",
            }
            for document in documents
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_analysis_document(document: StockAnalysisDocument) -> str:
    record = document.record
    profile = document.profile
    categories = "、".join(document.categories)
    concepts = "、".join(record.concepts) if record.concepts else "未披露"
    announcements = render_announcements(document.announcements)
    return "\n".join(
        [
            f"# {record.name}（{record.secucode}）A股科技相关股票完整分析",
            "",
            "## RAG 元数据",
            "",
            f"- 公司：{record.name}",
            f"- 股票代码：{record.secucode}",
            "- 市场分类：A股",
            f"- 交易所：{record.exchange}",
            f"- 东方财富 secid：{record.secid}",
            f"- 行业：{record.industry or '未披露'}",
            f"- 地域板块：{record.region or '未披露'}",
            f"- 科技细分分类：{categories}",
            f"- 关联概念：{concepts}",
            f"- 命中原因：{'；'.join(record.match_reasons) or '待核验'}",
            f"- 生成时间：{document.generated_at}",
            f"- 数据来源：{'；'.join(document.source_urls)}",
            f"- 资料标签：{record.tag_text}",
            "- 核验提醒：本文件由第三方公开数据自动生成，回答中引用时必须提示需核验交易所公告、公司原文披露、最新行情和财报。",
            "",
            "## 一句话结论",
            "",
            render_one_line_view(record, profile, document.categories),
            "",
            "## 公司与业务画像",
            "",
            f"- 法定公司名称：{profile.org_name or '待核验'}",
            f"- 英文名称：{profile.org_name_en or '待核验'}",
            f"- 证券类型：{profile.security_type or '待核验'}",
            f"- 交易市场：{profile.trade_market or '待核验'}",
            f"- 东方财富行业：{profile.em_industry or record.industry or '待核验'}",
            f"- 证监会行业：{profile.csrc_industry or '待核验'}",
            f"- 上市日期：{short_date(profile.listing_date) or '待核验'}",
            f"- 成立日期：{short_date(profile.found_date) or '待核验'}",
            f"- 注册资本：{profile.register_capital or '待核验'}",
            f"- 员工人数：{profile.employees or '待核验'}",
            f"- 董事长/总经理：{profile.chairman or '待核验'} / {profile.president or '待核验'}",
            f"- 董秘/电话/邮箱：{profile.secretary or '待核验'} / {profile.telephone or '待核验'} / {profile.email or '待核验'}",
            f"- 官网：{profile.website or '待核验'}",
            f"- 地址：{profile.address or profile.province or '待核验'}",
            "",
            "### 公司简介",
            "",
            profile.org_profile or "公司简介未从当前数据源取得，需通过年报、招股书或交易所公告补充核验。",
            "",
            "### 经营范围",
            "",
            profile.business_scope or "经营范围未从当前数据源取得，需通过工商信息、年报或交易所公告补充核验。",
            "",
            "## 科技相关性分析",
            "",
            render_tech_relevance(record, document.categories),
            "",
            "## 行情与估值快照",
            "",
            render_market_snapshot(record),
            "",
            "## 基本面分析框架",
            "",
            render_fundamental_framework(record, profile, document.categories),
            "",
            "## 催化剂与公告线索",
            "",
            announcements,
            "",
            "## 风险清单",
            "",
            render_risk_list(record, document.categories),
            "",
            "## 后续跟踪指标",
            "",
            render_tracking_metrics(record, document.categories),
            "",
            "## 推荐使用方式",
            "",
            "- 本文档可作为 RAG 检索底稿，用于回答公司画像、科技分类、估值快照、催化剂和风险问题。",
            "- 不直接给出买入、卖出或收益承诺；需要结合用户风险偏好、持仓周期、最新公告、最新行情和组合约束。",
            "- 若回答引用了本文件，必须同步提示资料标签中的 #needs_verification / #needs_refresh。",
            "",
        ]
    )


def render_one_line_view(
    record: AShareTechRecord,
    profile: CompanyProfile,
    categories: tuple[str, ...],
) -> str:
    category_text = "、".join(categories)
    business = profile.business_scope or profile.org_profile
    if business:
        business = truncate(clean_paragraph(business), 120)
    else:
        business = f"当前公开列表将其归入 {record.industry or '未披露行业'}，关联概念包括 {', '.join(record.concepts[:6]) or '待核验'}。"
    return (
        f"{record.name} 属于 A 股科技相关公司，主要落在 {category_text} 方向；"
        f"当前研究重点应围绕业务真实性、订单/收入兑现、估值消化和公告催化展开。{business}"
    )


def render_tech_relevance(record: AShareTechRecord, categories: tuple[str, ...]) -> str:
    lines = [
        f"- 细分方向：{'、'.join(categories)}。",
        f"- 行业标签：{record.industry or '未披露'}。",
        f"- 概念标签：{'、'.join(record.concepts[:20]) or '未披露'}。",
        f"- 自动命中逻辑：{'；'.join(record.match_reasons) or '待核验'}。",
        "- 研究判断：概念标签不等于主营收入贡献，需继续核对年报主营构成、投资者关系纪要和交易所公告。",
    ]
    return "\n".join(lines)


def render_market_snapshot(record: AShareTechRecord) -> str:
    lines = [
        f"- 最新价：{format_number(record.price, suffix=' 元')}",
        f"- 涨跌幅：{format_number(record.pct_change, suffix='%')}",
        f"- 涨跌额：{format_number(record.change, suffix=' 元')}",
        f"- 今开/最高/最低/昨收：{format_number(record.open_price)} / {format_number(record.high)} / {format_number(record.low)} / {format_number(record.prev_close)}",
        f"- 成交量：{format_number(record.volume, divisor=10000, suffix=' 万手/股口径待核验')}",
        f"- 成交额：{format_money(record.amount)}",
        f"- 换手率：{format_number(record.turnover_rate, suffix='%')}",
        f"- 动态市盈率：{format_number(record.pe_dynamic)}",
        f"- 市净率：{format_number(record.pb)}",
        f"- 总市值：{format_money(record.total_market_cap)}",
        f"- 流通市值：{format_money(record.float_market_cap)}",
        "- 估值提示：行情字段来自第三方快照，盘中和复权口径可能变化；正式分析需以交易所行情、公司财报和专业终端复核。",
    ]
    return "\n".join(lines)


def render_fundamental_framework(
    record: AShareTechRecord,
    profile: CompanyProfile,
    categories: tuple[str, ...],
) -> str:
    business_basis = profile.business_scope or profile.org_profile or record.industry
    lines = [
        f"- 主营核验：先用公司经营范围和年报主营构成确认收入是否真正来自 {record.industry or '相关行业'}，再判断概念含金量。",
        "- 成长性：重点跟踪收入增速、毛利率、研发费用率、在手订单、客户集中度和新产品放量节奏。",
        "- 盈利质量：关注经营现金流、应收账款、存货跌价、资本开支和政府补助占比。",
        "- 竞争格局：比较同赛道公司在客户、技术壁垒、产能、成本曲线和渠道上的差异。",
        "- 估值位置：结合 PE/PB/PS、历史分位、行业景气周期和盈利可见度评估估值是否透支。",
        f"- 当前底稿依据：{truncate(clean_paragraph(business_basis), 180) if business_basis else '当前数据源未提供主营详情，需补充年报或公告。'}",
    ]
    if "机器人与自动化" in categories:
        lines.append("- 机器人链条额外跟踪：减速器、伺服、电机、控制器、整机出货、客户验证和量产节奏。")
    if "AI算力与数据中心" in categories:
        lines.append("- AI 算力链条额外跟踪：GPU/服务器/光模块/液冷/IDC 订单、资本开支和供货约束。")
    if "半导体与电子" in categories:
        lines.append("- 半导体链条额外跟踪：库存周期、国产替代进度、晶圆产能、良率、价格和客户认证。")
    if "新能源科技与锂电" in categories:
        lines.append("- 新能源科技链条额外跟踪：电池装机、材料价格、海外客户、产能利用率和技术路线变化。")
    return "\n".join(lines)


def render_announcements(announcements: tuple[AnnouncementItem, ...]) -> str:
    if not announcements:
        return "\n".join(
            [
                f"- 当前未抓取到公告列表。资料标签：{NEEDS_REFRESH_TAG} {NEEDS_VERIFICATION_TAG}",
                "- 回答时必须说明公告数据缺失或刷新失败，不能把本文档视为最新公告结论。",
            ]
        )
    lines = [
        "- 以下为公告聚合接口返回的近期公告标题，需点开交易所公告原文核验：",
    ]
    for item in announcements:
        columns = "、".join(item.columns) if item.columns else "未分类"
        lines.append(
            f"- {short_date(item.notice_date) or item.notice_date or '日期待核验'}："
            f"{item.title}（栏目：{columns}，art_code：{item.art_code or '待核验'}，"
            f"tags: {' '.join(item.tags)}）"
        )
    return "\n".join(lines)


def render_risk_list(record: AShareTechRecord, categories: tuple[str, ...]) -> str:
    risks = [
        "- 资料风险：第三方概念和行情字段可能滞后、缺失或口径不同，需核验公告原文和交易所数据。",
        "- 估值风险：科技主题交易容易提前反映乐观预期，盈利兑现不及预期时估值回撤可能较快。",
        "- 业绩风险：研发投入、价格竞争、客户验收、库存周期和应收账款会影响利润质量。",
        "- 概念风险：概念命中不代表主营收入占比高，需核对主营构成和订单披露。",
        "- 流动性风险：小市值或高换手公司可能波动较大，策略上应设置仓位和止损纪律。",
    ]
    if record.pe_dynamic is not None and record.pe_dynamic < 0:
        risks.append("- 盈利风险：动态市盈率为负，说明当前盈利口径可能亏损或异常，需要重点核验。")
    if SPECIAL_TREATMENT_TAG in record.tags or is_special_treatment_name(record.name):
        risks.append("- 特别处理风险：该股票名称或标签显示存在 ST/退市风险特征，需优先核验交易所风险警示、持续经营和退市指标。")
    if "半导体与电子" in categories:
        risks.append("- 半导体周期风险：库存、价格、产能利用率和海外限制政策可能导致业绩波动。")
    if "AI算力与数据中心" in categories:
        risks.append("- AI 资本开支风险：下游算力投入节奏、芯片供给和订单真实性需要持续验证。")
    if "机器人与自动化" in categories:
        risks.append("- 机器人产业化风险：样机展示和量产交付之间存在较大不确定性。")
    if "新能源科技与锂电" in categories:
        risks.append("- 新能源价格风险：锂价、组件价格、产能过剩和技术路线切换可能压缩盈利。")
    return "\n".join(risks)


def render_tracking_metrics(record: AShareTechRecord, categories: tuple[str, ...]) -> str:
    metrics = [
        "- 最新公告：重大合同、定增、并购、减持、业绩预告、问询函和回购计划。",
        "- 财务指标：收入增速、净利润增速、毛利率、经营现金流、研发费用率、存货和应收账款。",
        "- 行情指标：成交额、换手率、相对行业指数强弱、估值分位和机构持仓变化。",
        "- 产业指标：所在赛道订单、价格、产能利用率、政策变化和核心客户资本开支。",
    ]
    if "AI算力与数据中心" in categories:
        metrics.append("- AI 专项：服务器/光模块/IDC/液冷订单、GPU 供给、云厂商资本开支。")
    if "机器人与自动化" in categories:
        metrics.append("- 机器人专项：人形机器人样机、核心零部件量产、客户验证、出货量。")
    if "无人机与低空经济" in categories:
        metrics.append("- 低空经济专项：空域政策、适航认证、订单交付、地方试点和运营牌照。")
    if "半导体与电子" in categories:
        metrics.append("- 半导体专项：晶圆价格、库存天数、客户导入、国产替代招标和良率。")
    return "\n".join(metrics)


def classify_categories(record: AShareTechRecord) -> tuple[str, ...]:
    haystack = normalize_text(
        " ".join([record.name, record.industry, " ".join(record.concepts), " ".join(record.match_reasons)])
    )
    categories: list[str] = []
    for category, keywords in CATEGORY_RULES:
        if any(normalize_text(keyword) in haystack for keyword in keywords):
            categories.append(category)
    if not categories:
        categories.append("其他科技相关")
    return tuple(dict.fromkeys(categories))


def company_profile_url(record: AShareTechRecord) -> str:
    return f"{EASTMONEY_COMPANY_SURVEY_URL}?code={record.f10_code}"


def fetch_json(
    url: str,
    *,
    params: dict | None = None,
    timeout: float = 8.0,
    headers: dict[str, str] | None = None,
    session: requests.Session | None = None,
) -> dict:
    request_headers = headers or {"User-Agent": "Mozilla/5.0"}
    client = session or requests
    try:
        response = client.get(url, params=params, timeout=timeout, headers=request_headers)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError):
        return fetch_json_with_curl(url, params=params, timeout=timeout, headers=request_headers)


def fetch_json_with_curl(
    url: str,
    *,
    params: dict | None = None,
    timeout: float = 8.0,
    headers: dict[str, str] | None = None,
) -> dict:
    full_url = url
    if params:
        separator = "&" if "?" in url else "?"
        full_url = f"{url}{separator}{urlencode(params)}"
    cmd = ["curl", "-L", "--compressed", "--silent", "--show-error", "--max-time", str(timeout)]
    for key, value in (headers or {}).items():
        cmd.extend(["-H", f"{key}: {value}"])
    cmd.append(full_url)
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout + 3,
    )
    return json.loads(completed.stdout)


def exchange_from_market(secid_market: int, code: str) -> str:
    if secid_market == 1 or code.startswith("6"):
        return "SH"
    if secid_market == 2 or code.startswith(("8", "4")):
        return "BJ"
    return "SZ"


def exchange_from_cninfo_org_id(org_id: str, code: str) -> str:
    normalized = org_id.lower()
    if code.startswith(("6", "7")):
        return "SH"
    if code.startswith(("0", "2", "3")):
        return "SZ"
    if code.startswith(("8", "4", "9")):
        return "BJ"
    if normalized.startswith("gssh"):
        return "SH"
    if normalized.startswith("gfbj"):
        return "BJ"
    return "SZ"


def is_delisted_name(name: str) -> bool:
    normalized = name.upper()
    return normalized.endswith("退") or "退市" in normalized


def is_special_treatment_name(name: str) -> bool:
    normalized = name.upper()
    return "ST" in normalized or "*ST" in normalized


def secid_market_from_exchange(exchange: str) -> int:
    if exchange == "SH":
        return 1
    if exchange == "BJ":
        return 2
    return 0


def deduplicate_records(records: Iterable[AShareTechRecord]) -> list[AShareTechRecord]:
    seen: set[str] = set()
    unique: list[AShareTechRecord] = []
    for record in records:
        if record.code in seen:
            continue
        seen.add(record.code)
        unique.append(record)
    return sorted(unique, key=lambda item: item.code)


def extract_keyword_tags(value: str) -> str:
    normalized = normalize_text(value)
    matches = [
        term
        for term in (*DEFAULT_TECH_INDUSTRY_KEYWORDS, *DEFAULT_TECH_CONCEPT_KEYWORDS)
        if normalize_text(term) in normalized
    ]
    return "、".join(dict.fromkeys(matches))


def split_concepts(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,，、]", value) if part.strip()]


def normalize_terms(terms: Iterable[str]) -> tuple[str, ...]:
    return tuple(term.strip() for term in terms if term and term.strip())


def normalize_text(value: str) -> str:
    return value.lower().replace(" ", "")


def first_dict(value) -> dict:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    return {}


def parse_float(value) -> float | None:
    if value in {None, "", "-", "N/D"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def text_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_paragraph(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def short_date(value: str) -> str:
    if not value:
        return ""
    match = re.match(r"(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else value


def truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 1].rstrip() + "..."


def format_number(value: float | None, *, divisor: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "待核验"
    number = value / divisor
    return f"{number:.2f}{suffix}"


def format_money(value: float | None) -> str:
    if value is None:
        return "待核验"
    abs_value = abs(value)
    if abs_value >= 100_000_000:
        return f"{value / 100_000_000:.2f} 亿元"
    if abs_value >= 10_000:
        return f"{value / 10_000:.2f} 万元"
    return f"{value:.2f} 元"


def safe_filename(value: str) -> str:
    safe = re.sub(r"[\\/:*?\"<>|\s]+", "_", value).strip("_")
    return safe[:80] or "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

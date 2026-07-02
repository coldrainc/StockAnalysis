from pathlib import Path

from stock_agent.market.a_share_tech import (
    AShareTechRecord,
    AnnouncementItem,
    CompanyProfile,
    StockAnalysisDocument,
    exchange_from_cninfo_org_id,
    is_strong_profile_match,
    render_analysis_document,
    select_tech_records,
    write_analysis_documents,
)
from stock_agent.market.a_share_refresh import (
    EASTMONEY_ULIST_QUOTE_URL,
    FinancialSummary,
    QuoteSnapshot,
    fetch_quote_snapshot,
    format_quote_time,
    load_manifest_companies,
    refresh_a_share_tech_dynamic,
)
from stock_agent.market.market_scope import prepare_market_rag_source


def test_select_tech_records_matches_industry_and_concept() -> None:
    records = [
        AShareTechRecord(
            code="300124",
            name="汇川技术",
            exchange="SZ",
            secid_market=0,
            industry="自动化设备",
            region="广东板块",
            concepts=("机器人概念", "工业互联网"),
        ),
        AShareTechRecord(
            code="600519",
            name="贵州茅台",
            exchange="SH",
            secid_market=1,
            industry="白酒",
            region="贵州板块",
            concepts=("白酒",),
        ),
    ]

    selected = select_tech_records(records)

    assert [record.code for record in selected] == ["300124"]
    assert "行业命中：自动化设备" in selected[0].match_reasons
    assert "概念命中：机器人" in selected[0].match_reasons


def test_exchange_from_cninfo_org_id_supports_sh_sz_bj() -> None:
    assert exchange_from_cninfo_org_id("gssz0000021", "000021") == "SZ"
    assert exchange_from_cninfo_org_id("gfbj0001259", "001259") == "SZ"
    assert exchange_from_cninfo_org_id("gssh600000", "600000") == "SH"
    assert exchange_from_cninfo_org_id("gfbj0831726", "831726") == "BJ"


def test_profile_match_requires_stronger_evidence_for_weak_business_terms() -> None:
    assert is_strong_profile_match((), ("业务/F10命中：软件",)) is False
    assert is_strong_profile_match((), ("业务/F10命中：软件", "业务/F10命中：电子")) is True
    assert is_strong_profile_match((), ("业务/F10命中：半导体",)) is True
    assert is_strong_profile_match(("行业命中：软件",), ()) is True


def test_render_analysis_document_contains_full_sections_and_tags() -> None:
    document = StockAnalysisDocument(
        record=AShareTechRecord(
            code="300124",
            name="汇川技术",
            exchange="SZ",
            secid_market=0,
            industry="自动化设备",
            region="广东板块",
            concepts=("机器人概念", "工业互联网"),
            pe_dynamic=30.5,
            total_market_cap=100_000_000_000,
            match_reasons=("行业命中：自动化设备", "概念命中：机器人"),
        ),
        profile=CompanyProfile(
            org_name="深圳市汇川技术股份有限公司",
            business_scope="工业自动化控制产品的研发、生产和销售。",
            org_profile="公司聚焦工业自动化、新能源汽车和智慧电梯等业务。",
        ),
        announcements=(
            AnnouncementItem(
                title="汇川技术:年度报告",
                notice_date="2026-04-20 00:00:00",
                columns=("年度报告",),
                art_code="AN1",
            ),
        ),
        categories=("机器人与自动化",),
        generated_at="2026-07-01 00:00:00Z",
        source_urls=("https://example.test/list",),
    )

    content = render_analysis_document(document)

    assert "## 一句话结论" in content
    assert "## 科技相关性分析" in content
    assert "## 基本面分析框架" in content
    assert "## 催化剂与公告线索" in content
    assert "## 风险清单" in content
    assert "#needs_verification" in content
    assert "汇川技术:年度报告" in content


def test_write_analysis_documents_and_market_scope_include_company_docs(tmp_path: Path) -> None:
    stock_universe = tmp_path / "stock_universe"
    markets = stock_universe / "markets"
    markets.mkdir(parents=True)
    (markets / "A股.md").write_text("# A股\n\n## 科技\n\n- 汇川技术", encoding="utf-8")
    (markets / "market_universe.md").write_text("# 市场总览", encoding="utf-8")
    (stock_universe / "dynamic").mkdir()
    (stock_universe / "dynamic" / "refresh_status.md").write_text("# refresh", encoding="utf-8")

    output = tmp_path / "a_share_technology"
    write_analysis_documents(
        output,
        [
            StockAnalysisDocument(
                record=AShareTechRecord(
                    code="300124",
                    name="汇川技术",
                    exchange="SZ",
                    secid_market=0,
                    industry="自动化设备",
                    region="广东板块",
                    concepts=("机器人概念",),
                    match_reasons=("行业命中：自动化设备",),
                ),
                profile=CompanyProfile(),
                announcements=(),
                categories=("机器人与自动化",),
                generated_at="2026-07-01 00:00:00Z",
                source_urls=("https://example.test/list",),
            )
        ],
        total_a_share_count=1,
    )

    source_dir = tmp_path / "A股_rag"
    files = prepare_market_rag_source(stock_universe, source_dir, "A股")

    assert any(path.name == "300124_汇川技术.md" for path in files)
    assert (source_dir / "a_share_technology" / "companies" / "300124_汇川技术.md").exists()


def test_market_scope_removes_stale_a_share_docs(tmp_path: Path) -> None:
    stock_universe = tmp_path / "stock_universe"
    markets = stock_universe / "markets"
    markets.mkdir(parents=True)
    (markets / "A股.md").write_text("# A股", encoding="utf-8")

    source_docs = tmp_path / "a_share_technology" / "companies"
    source_docs.mkdir(parents=True)
    (source_docs / "300124_汇川技术.md").write_text("# 汇川技术", encoding="utf-8")

    output = tmp_path / "A股_rag"
    stale_dir = output / "a_share_technology" / "companies"
    stale_dir.mkdir(parents=True)
    (stale_dir / "000004_国华退.md").write_text("# stale", encoding="utf-8")

    prepare_market_rag_source(stock_universe, output, "A股")

    assert not (stale_dir / "000004_国华退.md").exists()
    assert (stale_dir / "300124_汇川技术.md").exists()


def test_load_manifest_companies_and_refresh_dynamic_docs(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "a_share_technology"
    output.mkdir()
    (output / "manifest.json").write_text(
        """
        {
          "documents": [
            {
              "code": "300750",
              "name": "宁德时代",
              "secucode": "300750.SZ",
              "industry": "电池",
              "categories": ["新能源科技与锂电"],
              "match_reasons": ["概念命中：锂电池"],
              "path": "companies/300750_宁德时代.md"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    def fake_quote(record, timeout=8.0):
        return QuoteSnapshot(price=200.0, pct_change=1.2, quote_time="2026-07-01 10:00:00")

    def fake_announcements(record, limit=5, timeout=8.0):
        return [
            AnnouncementItem(
                title="宁德时代:年度报告",
                notice_date="2026-04-20 00:00:00",
                columns=("年度报告",),
                art_code="AN1",
            )
        ]

    def fake_financial(record, timeout=8.0):
        return FinancialSummary(
            title="宁德时代 财务摘要",
            content="收入与利润摘要。",
            as_of="2026-03-31",
            source="https://example.test/financials",
        )

    monkeypatch.setattr("stock_agent.market.a_share_refresh.fetch_quote_snapshot", fake_quote)
    monkeypatch.setattr("stock_agent.market.a_share_refresh.fetch_announcements", fake_announcements)
    monkeypatch.setattr("stock_agent.market.a_share_refresh.fetch_financial_summary", fake_financial)

    companies = load_manifest_companies(output / "manifest.json")
    result = refresh_a_share_tech_dynamic(output, max_companies=None, workers=1)

    content = (
        output / "dynamic" / "companies" / "300750_宁德时代_动态刷新.md"
    ).read_text(encoding="utf-8")
    status = (output / "dynamic" / "refresh_status.md").read_text(encoding="utf-8")

    assert companies[0].record.secucode == "300750.SZ"
    assert result.refreshed_count == 1
    assert "行情与估值刷新快照" in content
    assert "宁德时代:年度报告" in content
    assert "收入与利润摘要" in content
    assert "#needs_verification" in content
    assert "本次覆盖股票数：1" in status


def test_fetch_quote_snapshot_parses_eastmoney_scaled_fields(monkeypatch) -> None:
    def fake_fetch_json(url, *, params=None, timeout=8.0, headers=None, session=None):
        return {
            "data": {
                "f43": 20123,
                "f44": 20300,
                "f45": 19900,
                "f46": 20000,
                "f47": 123456,
                "f48": 987654321,
                "f59": 2,
                "f60": 19888,
                "f86": 1782892800,
                "f116": 900000000000,
                "f117": 800000000000,
                "f162": 2150,
                "f167": 500,
                "f168": 120,
                "f169": 235,
                "f170": 118,
                "f171": 300,
            }
        }

    monkeypatch.setattr("stock_agent.market.a_share_refresh.fetch_json", fake_fetch_json)

    quote = fetch_quote_snapshot(
        AShareTechRecord(
            code="300750",
            name="宁德时代",
            exchange="SZ",
            secid_market=0,
            industry="电池",
            region="",
            concepts=(),
        )
    )

    assert quote is not None
    assert quote.price == 201.23
    assert quote.pct_change == 1.18
    assert quote.pe_dynamic == 21.5


def test_fetch_quote_snapshot_falls_back_to_ulist(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch_json(url, *, params=None, timeout=8.0, headers=None, session=None):
        calls.append(url)
        if url != EASTMONEY_ULIST_QUOTE_URL:
            raise RuntimeError("empty reply")
        return {
            "data": {
                "diff": [
                    {
                        "f2": 106.64,
                        "f3": 2.98,
                        "f4": 3.09,
                        "f5": 2459790,
                        "f6": 26327311096.0,
                        "f8": 13.75,
                        "f9": 164.35,
                        "f12": "600584",
                        "f14": "长电科技",
                        "f17": 105.5,
                        "f18": 103.55,
                        "f20": 190823169745,
                        "f21": 190823169745,
                        "f124": 1782893506,
                    }
                ]
            }
        }

    monkeypatch.setattr("stock_agent.market.a_share_refresh.fetch_json", fake_fetch_json)

    quote = fetch_quote_snapshot(
        AShareTechRecord(
            code="600584",
            name="长电科技",
            exchange="SH",
            secid_market=1,
            industry="半导体",
            region="",
            concepts=(),
        )
    )

    assert calls[-1] == EASTMONEY_ULIST_QUOTE_URL
    assert quote is not None
    assert quote.price == 106.64
    assert quote.pct_change == 2.98
    assert quote.source == EASTMONEY_ULIST_QUOTE_URL


def test_format_quote_time_supports_eastmoney_timestamp_shapes() -> None:
    assert format_quote_time("20260701150203") == "2026-07-01 15:02:03"
    assert format_quote_time("1782892800") == "2026-07-01 08:00:00Z"

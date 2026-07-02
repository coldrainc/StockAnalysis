from pathlib import Path

from stock_agent.market.a_share_refresh import QuoteSnapshot
from stock_agent.market.daily_picks import build_daily_picks, load_portfolio


def write_manifest(output: Path) -> None:
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
            },
            {
              "code": "300124",
              "name": "汇川技术",
              "secucode": "300124.SZ",
              "industry": "自动化设备",
              "categories": ["机器人与自动化"],
              "match_reasons": ["行业命中：自动化设备"],
              "path": "companies/300124_汇川技术.md"
            },
            {
              "code": "688981",
              "name": "中芯国际",
              "secucode": "688981.SH",
              "industry": "半导体",
              "categories": ["半导体与电子"],
              "match_reasons": ["行业命中：半导体"],
              "path": "companies/688981_中芯国际.md"
            }
          ]
        }
        """,
        encoding="utf-8",
    )


def test_build_daily_picks_generates_report_and_portfolio_section(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "a_share_technology"
    write_manifest(output)
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text(
        "code,name,shares,cost_price,notes\n300750,宁德时代,100,180,核心持仓\n",
        encoding="utf-8",
    )

    quotes = {
        "300750": QuoteSnapshot(
            price=200.0,
            pct_change=2.5,
            amount=900_000_000,
            turnover_rate=2.0,
            pe_dynamic=28.0,
            pb=4.8,
        ),
        "300124": QuoteSnapshot(
            price=65.0,
            pct_change=4.2,
            amount=1_500_000_000,
            turnover_rate=3.5,
            pe_dynamic=32.0,
            pb=5.2,
        ),
        "688981": QuoteSnapshot(
            price=52.0,
            pct_change=-1.0,
            amount=700_000_000,
            turnover_rate=1.2,
            pe_dynamic=80.0,
            pb=3.1,
        ),
    }

    def fake_quote(record, timeout=8.0):
        return quotes[record.code]

    monkeypatch.setattr("stock_agent.market.daily_picks.fetch_quote_snapshot", fake_quote)

    result = build_daily_picks(
        output,
        portfolio_path=portfolio,
        max_candidates=None,
        top_k=2,
        workers=1,
    )

    content = result.report_path.read_text(encoding="utf-8")

    assert result.universe_count == 3
    assert result.scanned_count == 3
    assert result.picked_count == 2
    assert result.portfolio_count == 1
    assert result.latest_path.exists()
    assert "每日量化推荐观察池" in content
    assert "## 今日候选 Top" in content
    assert "## 持仓分析" in content
    assert "宁德时代" in content
    assert "核心持仓" in content
    assert "评分：" in content
    assert "#quant_candidate" in content
    assert "#portfolio_analysis" in content
    assert "#needs_verification" in content


def test_load_portfolio_supports_chinese_columns(tmp_path: Path) -> None:
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text(
        "股票代码,股票名称,持仓数量,成本价,持仓市值,备注\n300750,宁德时代,100,180,18000,核心持仓\n",
        encoding="utf-8",
    )

    positions = load_portfolio(portfolio)

    assert positions[0].normalized_code == "300750"
    assert positions[0].name == "宁德时代"
    assert positions[0].shares == 100
    assert positions[0].cost_price == 180
    assert positions[0].market_value == 18000
    assert positions[0].notes == "核心持仓"

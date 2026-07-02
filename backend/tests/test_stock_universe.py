from pathlib import Path

from stock_agent.market.stock_universe import classify_market, collect_company_universe


def test_collect_universe_includes_new_hardware_robotics_sectors(tmp_path: Path) -> None:
    records = collect_company_universe(
        tmp_path,
        sectors=["具身智能", "无人机", "机器人", "硬件"],
        include_remote=False,
    )

    sectors = {record.sector for record in records}
    assert {"具身智能", "无人机", "机器人", "硬件"} <= sectors

    overview = (tmp_path / "company_universe.md").read_text(encoding="utf-8")
    assert "优必选" in overview
    assert "中无人机" in overview
    assert "绿的谐波" in overview
    assert "立讯精密" in overview
    assert "#needs_refresh" in overview
    assert "市场分类摘要" in overview

    assert (tmp_path / "markets" / "A股.md").exists()
    assert (tmp_path / "markets" / "美股.md").exists()
    assert (tmp_path / "markets" / "港股.md").exists()
    assert "立讯精密" in (tmp_path / "markets" / "A股.md").read_text(encoding="utf-8")
    assert "Intuitive Surgical" in (tmp_path / "markets" / "美股.md").read_text(
        encoding="utf-8"
    )
    assert "优必选" in (tmp_path / "markets" / "港股.md").read_text(encoding="utf-8")


def test_collect_universe_marks_user_csv_as_needs_verification(tmp_path: Path) -> None:
    csv_path = tmp_path / "extra.csv"
    csv_path.write_text(
        "sector,name,ticker,market,aliases,reason,reliability,tags\n"
        "硬件,测试硬件,TEST,CN,别名,测试原因,user_supplied,#user_supplied|#needs_verification\n",
        encoding="utf-8",
    )

    collect_company_universe(
        tmp_path / "kb",
        sectors=["硬件"],
        extra_csv=csv_path,
        include_remote=False,
    )

    content = (tmp_path / "kb" / "硬件.md").read_text(encoding="utf-8")
    assert "测试硬件" in content
    assert "资料可信度：user_supplied" in content
    assert "#needs_verification" in content


def test_classify_market_supports_a_us_hk() -> None:
    assert classify_market("CN", "300750.SZ") == "A股"
    assert classify_market("US", "NVDA") == "美股"
    assert classify_market("HK", "0700.HK") == "港股"

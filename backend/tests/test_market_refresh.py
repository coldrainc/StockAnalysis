from pathlib import Path

from stock_agent.market.market_refresh import refresh_dynamic_knowledge, write_manual_item
from stock_agent.market.stock_universe import CompanyRecord


def test_refresh_dynamic_knowledge_writes_status_when_no_provider(tmp_path: Path) -> None:
    records = [
        CompanyRecord("硬件", "测试公司", "PRIVATE", "CN", reason="测试"),
    ]

    items = refresh_dynamic_knowledge(
        tmp_path,
        records,
        include_quotes=True,
        include_announcements=True,
        include_financials=True,
    )

    assert items == []
    status = (tmp_path / "dynamic" / "refresh_status.md").read_text(encoding="utf-8")
    assert "#needs_refresh" in status
    assert "#needs_verification" in status


def test_write_manual_item_adds_verification_tags(tmp_path: Path) -> None:
    path = write_manual_item(
        tmp_path,
        {
            "title": "测试公告",
            "category": "公告",
            "company": "测试公司",
            "ticker": "TEST",
            "market": "CN",
            "content": "这是一条用户手动资料。",
            "tags": ["#user_supplied", "#needs_verification"],
        },
    )

    content = path.read_text(encoding="utf-8")
    assert "测试公告" in content
    assert "#user_supplied #needs_verification" in content

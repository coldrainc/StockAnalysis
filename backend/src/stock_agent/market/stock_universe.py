from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_SECTORS = (
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
)


OFFICIAL_TAG = "#official_or_exchange"
CURATED_TAG = "#curated_seed"
THIRD_PARTY_TAG = "#third_party_dataset"
USER_SUPPLIED_TAG = "#user_supplied"
NEEDS_REFRESH_TAG = "#needs_refresh"
NEEDS_VERIFICATION_TAG = "#needs_verification"

MARKET_ALIASES = {
    "A股": ("CN", "SH", "SZ", "A"),
    "美股": ("US", "NASDAQ", "NYSE", "AMEX"),
    "港股": ("HK", "HKG"),
}


@dataclass(frozen=True)
class CompanyRecord:
    sector: str
    name: str
    ticker: str
    market: str
    aliases: tuple[str, ...] = ()
    reason: str = ""
    source: str = "seed"
    reliability: str = "curated_seed"
    tags: tuple[str, ...] = (CURATED_TAG, NEEDS_REFRESH_TAG)

    def normalized_key(self) -> tuple[str, str, str]:
        return (self.market.lower(), self.ticker.lower(), self.name.lower())

    @property
    def tag_text(self) -> str:
        return " ".join(self.tags)

    @property
    def verification_note(self) -> str:
        if NEEDS_VERIFICATION_TAG in self.tags:
            return "第三方或用户提供资料，回答时需提醒存在核验风险。"
        if NEEDS_REFRESH_TAG in self.tags:
            return "静态种子资料，回答时需提醒需结合最新公告、行情和财报刷新。"
        return "来源可信度较高，但仍需结合最新披露确认。"

    @property
    def market_group(self) -> str:
        return classify_market(self.market, self.ticker)


SEED_COMPANIES: tuple[CompanyRecord, ...] = (
    CompanyRecord("科技", "Apple", "AAPL", "US", ("苹果",), "消费电子、芯片、服务生态"),
    CompanyRecord("科技", "Microsoft", "MSFT", "US", ("微软",), "云计算、AI、企业软件"),
    CompanyRecord("科技", "NVIDIA", "NVDA", "US", ("英伟达",), "AI GPU、数据中心、自动驾驶"),
    CompanyRecord("科技", "台积电", "TSM", "US/TW", ("TSMC", "2330.TW"), "先进制程、半导体代工"),
    CompanyRecord("科技", "腾讯控股", "0700.HK", "HK", ("Tencent",), "社交、游戏、云和金融科技"),
    CompanyRecord("科技", "阿里巴巴", "9988.HK", "HK", ("BABA", "Alibaba"), "电商、云计算、AI"),
    CompanyRecord("科技", "中芯国际", "688981.SH", "CN", ("SMIC", "0981.HK"), "半导体制造、国产替代"),
    CompanyRecord("科技", "工业富联", "601138.SH", "CN", ("FII",), "AI服务器、电子制造"),
    CompanyRecord("材料", "宝钢股份", "600019.SH", "CN", ("Baoshan Iron",), "钢铁、制造业材料"),
    CompanyRecord("材料", "中国铝业", "601600.SH", "CN", ("Chalco", "2600.HK"), "铝土矿、氧化铝、电解铝"),
    CompanyRecord("材料", "万华化学", "600309.SH", "CN", ("Wanhua",), "MDI、化工新材料"),
    CompanyRecord("材料", "东方雨虹", "002271.SZ", "CN", ("Yuhong",), "建筑防水材料"),
    CompanyRecord("材料", "紫金矿业", "601899.SH", "CN", ("2899.HK",), "铜、金、锂等矿产资源"),
    CompanyRecord("材料", "洛阳钼业", "603993.SH", "CN", ("CMOC", "3993.HK"), "钼、钴、铜、铌"),
    CompanyRecord("贵金属", "山东黄金", "600547.SH", "CN", ("1787.HK",), "黄金开采、冶炼"),
    CompanyRecord("贵金属", "中金黄金", "600489.SH", "CN", (), "黄金矿山、冶炼"),
    CompanyRecord("贵金属", "银泰黄金", "000975.SZ", "CN", (), "黄金和有色金属矿产"),
    CompanyRecord("贵金属", "Barrick Gold", "GOLD", "US", ("巴里克黄金",), "全球黄金矿业"),
    CompanyRecord("贵金属", "Newmont", "NEM", "US", ("纽蒙特",), "黄金、铜矿资产"),
    CompanyRecord("能源", "中国石油", "601857.SH", "CN", ("PetroChina", "0857.HK"), "油气勘探、炼化、销售"),
    CompanyRecord("能源", "中国石化", "600028.SH", "CN", ("Sinopec", "0386.HK"), "炼化、化工、成品油"),
    CompanyRecord("能源", "中国海油", "600938.SH", "CN", ("CNOOC", "0883.HK"), "海上油气开采"),
    CompanyRecord("能源", "陕西煤业", "601225.SH", "CN", (), "动力煤、煤炭分红资产"),
    CompanyRecord("能源", "长江电力", "600900.SH", "CN", (), "水电、公用事业现金流"),
    CompanyRecord("能源", "Exxon Mobil", "XOM", "US", ("埃克森美孚",), "综合油气"),
    CompanyRecord("能源", "Chevron", "CVX", "US", ("雪佛龙",), "综合油气"),
    CompanyRecord("锂电池", "宁德时代", "300750.SZ", "CN", ("CATL",), "动力电池、储能电池"),
    CompanyRecord("锂电池", "比亚迪", "002594.SZ", "CN", ("BYD", "1211.HK"), "新能源汽车、动力电池"),
    CompanyRecord("锂电池", "赣锋锂业", "002460.SZ", "CN", ("1772.HK",), "锂资源、锂盐"),
    CompanyRecord("锂电池", "天齐锂业", "002466.SZ", "CN", ("9696.HK",), "锂矿、锂化工"),
    CompanyRecord("锂电池", "亿纬锂能", "300014.SZ", "CN", ("EVE",), "消费、电动、储能电池"),
    CompanyRecord("锂电池", "天赐材料", "002709.SZ", "CN", (), "电解液、锂电材料"),
    CompanyRecord("锂电池", "Tesla", "TSLA", "US", ("特斯拉",), "新能源汽车、储能、供应链牵引"),
    CompanyRecord("银行", "工商银行", "601398.SH", "CN", ("ICBC", "1398.HK"), "大型国有银行"),
    CompanyRecord("银行", "建设银行", "601939.SH", "CN", ("CCB", "0939.HK"), "大型国有银行"),
    CompanyRecord("银行", "农业银行", "601288.SH", "CN", ("ABC", "1288.HK"), "大型国有银行"),
    CompanyRecord("银行", "中国银行", "601988.SH", "CN", ("BOC", "3988.HK"), "大型国有银行、跨境业务"),
    CompanyRecord("银行", "招商银行", "600036.SH", "CN", ("CMB", "3968.HK"), "零售银行、财富管理"),
    CompanyRecord("银行", "兴业银行", "601166.SH", "CN", (), "股份制银行、绿色金融"),
    CompanyRecord("银行", "JPMorgan Chase", "JPM", "US", ("摩根大通",), "美国综合银行"),
    CompanyRecord("银行", "Bank of America", "BAC", "US", ("美国银行",), "美国大型银行"),
    CompanyRecord("具身智能", "Tesla", "TSLA", "US", ("特斯拉",), "自动驾驶、机器人、储能和制造数据闭环"),
    CompanyRecord("具身智能", "NVIDIA", "NVDA", "US", ("英伟达",), "机器人训练、仿真、边缘计算和 GPU 平台"),
    CompanyRecord("具身智能", "优必选", "9880.HK", "HK", ("UBTech",), "人形机器人、服务机器人"),
    CompanyRecord("具身智能", "小鹏汽车", "9868.HK", "HK", ("XPEV",), "智能驾驶、机器人相关研发"),
    CompanyRecord("具身智能", "小米集团", "1810.HK", "HK", ("Xiaomi",), "智能硬件、汽车、机器人生态"),
    CompanyRecord("具身智能", "Figure AI", "PRIVATE", "US", ("Figure",), "人形机器人未上市关联公司"),
    CompanyRecord("具身智能", "智元机器人", "PRIVATE", "CN", ("AgiBot",), "人形机器人未上市关联公司"),
    CompanyRecord("无人机", "中无人机", "688297.SH", "CN", (), "大型固定翼无人机系统"),
    CompanyRecord("无人机", "航天彩虹", "002389.SZ", "CN", (), "无人机和导弹装备"),
    CompanyRecord("无人机", "纵横股份", "688070.SH", "CN", (), "工业无人机系统"),
    CompanyRecord("无人机", "亿航智能", "EH", "US", ("EHang",), "eVTOL、载人无人机"),
    CompanyRecord("无人机", "AeroVironment", "AVAV", "US", (), "军用小型无人机、巡飞弹"),
    CompanyRecord("无人机", "Joby Aviation", "JOBY", "US", ("Joby",), "eVTOL 空中出行"),
    CompanyRecord("无人机", "DJI", "PRIVATE", "CN", ("大疆",), "消费级和工业无人机未上市关联公司"),
    CompanyRecord("机器人", "机器人", "300024.SZ", "CN", ("新松机器人",), "工业机器人、自动化装备"),
    CompanyRecord("机器人", "埃斯顿", "002747.SZ", "CN", (), "工业机器人、运动控制"),
    CompanyRecord("机器人", "汇川技术", "300124.SZ", "CN", (), "工业自动化、伺服和控制系统"),
    CompanyRecord("机器人", "绿的谐波", "688017.SH", "CN", (), "谐波减速器、机器人核心零部件"),
    CompanyRecord("机器人", "拓斯达", "300607.SZ", "CN", (), "工业机器人、注塑自动化"),
    CompanyRecord("机器人", "Intuitive Surgical", "ISRG", "US", ("直觉外科",), "手术机器人"),
    CompanyRecord("机器人", "Teradyne", "TER", "US", (), "协作机器人和自动化测试"),
    CompanyRecord("机器人", "ABB", "ABBNY", "US/CH", (), "工业机器人和电气自动化"),
    CompanyRecord("硬件", "立讯精密", "002475.SZ", "CN", ("Luxshare",), "消费电子、连接器、汽车电子"),
    CompanyRecord("硬件", "歌尔股份", "002241.SZ", "CN", ("Goertek",), "声学、VR/AR、智能硬件"),
    CompanyRecord("硬件", "蓝思科技", "300433.SZ", "CN", (), "玻璃盖板、结构件、消费电子"),
    CompanyRecord("硬件", "领益智造", "002600.SZ", "CN", (), "精密功能件、消费电子"),
    CompanyRecord("硬件", "工业富联", "601138.SH", "CN", ("FII",), "AI服务器、电子制造"),
    CompanyRecord("硬件", "联想集团", "0992.HK", "HK", ("Lenovo",), "PC、服务器、AI硬件"),
    CompanyRecord("硬件", "Dell Technologies", "DELL", "US", ("戴尔",), "服务器、PC、AI基础设施"),
    CompanyRecord("硬件", "HP", "HPQ", "US", ("惠普",), "PC、打印和终端硬件"),
)


def collect_company_universe(
    output_dir: Path,
    sectors: Iterable[str] = DEFAULT_SECTORS,
    extra_csv: Path | None = None,
    include_remote: bool = True,
    timeout: float = 8.0,
) -> list[CompanyRecord]:
    sector_set = {sector.strip() for sector in sectors if sector.strip()}
    records = [record for record in SEED_COMPANIES if record.sector in sector_set]
    if extra_csv and extra_csv.exists():
        records.extend(load_extra_csv(extra_csv))
    if include_remote:
        records.extend(fetch_us_listed_examples(timeout=timeout))
    records = deduplicate(record for record in records if record.sector in sector_set)
    write_company_markdown(output_dir, records, sector_set)
    return records


def load_extra_csv(path: Path) -> list[CompanyRecord]:
    records: list[CompanyRecord] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            aliases = tuple(
                item.strip() for item in row.get("aliases", "").split("|") if item.strip()
            )
            records.append(
                CompanyRecord(
                    sector=row.get("sector", "").strip(),
                    name=row.get("name", "").strip(),
                    ticker=row.get("ticker", "").strip(),
                    market=row.get("market", "").strip() or "UNKNOWN",
                    aliases=aliases,
                    reason=row.get("reason", "").strip(),
                    source=str(path),
                    reliability=row.get("reliability", "user_supplied").strip()
                    or "user_supplied",
                    tags=tuple(
                        item.strip()
                        for item in (
                            row.get("tags", f"{USER_SUPPLIED_TAG}|{NEEDS_VERIFICATION_TAG}")
                        ).split("|")
                        if item.strip()
                    ),
                )
            )
    return [record for record in records if record.sector and record.name and record.ticker]


def fetch_us_listed_examples(timeout: float = 8.0) -> list[CompanyRecord]:
    """Best-effort public refresh. The seed universe remains useful when offline."""

    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.json"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return []

    sector_map = {
        "Information Technology": "科技",
        "Materials": "材料",
        "Energy": "能源",
        "Financials": "银行",
    }
    records: list[CompanyRecord] = []
    for item in payload:
        sector = sector_map.get(str(item.get("Sector", "")))
        if not sector:
            continue
        name = str(item.get("Name", "")).strip()
        ticker = str(item.get("Symbol", "")).strip()
        if not name or not ticker:
            continue
        records.append(
            CompanyRecord(
                sector=sector,
                name=name,
                ticker=ticker,
                market="US",
                reason=str(item.get("Sub-Sector", "")).strip(),
                source=url,
                reliability="third_party_dataset",
                tags=(THIRD_PARTY_TAG, NEEDS_VERIFICATION_TAG),
            )
        )
    return records


def deduplicate(records: Iterable[CompanyRecord]) -> list[CompanyRecord]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[CompanyRecord] = []
    for record in records:
        key = record.normalized_key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return sorted(unique, key=lambda item: (item.sector, item.market, item.ticker, item.name))


def write_company_markdown(
    output_dir: Path,
    records: list[CompanyRecord],
    sectors: set[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    by_sector = {sector: [] for sector in sorted(sectors)}
    by_market = {"A股": [], "美股": [], "港股": [], "其他": []}
    for record in records:
        by_sector.setdefault(record.sector, []).append(record)
        by_market.setdefault(record.market_group, []).append(record)

    overview = [
        "# 股票分析 RAG 公司库",
        "",
        f"- 生成时间：{generated_at}",
        f"- 覆盖方向：{'、'.join(sorted(sectors))}",
        "- 用途：为股票分析 Agent 提供行业公司别名、代码、市场和研究线索。",
        "- 分类方式：同时按行业和市场分类；市场分类覆盖 A 股、美股、港股。",
        "- 注意：公司列表用于研究召回，不代表推荐买入；资料标签会进入 RAG，上下文若出现 #needs_verification 或 #needs_refresh，回答时必须提醒。",
        "",
        "## 市场分类摘要",
        "",
    ]
    for market_name in ("A股", "美股", "港股", "其他"):
        overview.append(f"- {market_name}：{len(by_market.get(market_name, []))} 家")
    overview.append("")
    overview.extend(
        [
            "## 行业分类摘要",
            "",
        ]
    )
    for sector, items in by_sector.items():
        overview.append(f"- {sector}：{len(items)} 家")
    overview.append("")
    overview.extend(
        [
            "## 行业明细",
            "",
        ]
    )
    for sector, items in by_sector.items():
        overview.extend([f"## {sector}", ""])
        for item in items:
            overview.append(
                f"- {item.name}（{item.ticker}，{item.market}）：{item.reason} "
                f"[tags: {item.tag_text}]"
            )
        overview.append("")
    (output_dir / "company_universe.md").write_text("\n".join(overview), encoding="utf-8")

    for sector, items in by_sector.items():
        lines = [f"# {sector} 上市公司与关联公司", ""]
        for item in items:
            aliases = "、".join(item.aliases) if item.aliases else "无"
            lines.extend(
                [
                    f"## {item.name}",
                    "",
                    f"- 股票代码：{item.ticker}",
                    f"- 市场：{item.market}",
                    f"- 别名：{aliases}",
                    f"- 关联逻辑：{item.reason or '待补充'}",
                    f"- 数据来源：{item.source}",
                    f"- 资料可信度：{item.reliability}",
                    f"- 资料标签：{item.tag_text}",
                    f"- 核验提示：{item.verification_note}",
                    "",
                    "研究提示：分析时结合最新公告、财报、行情、价格、政策、估值和风险偏好；资料带 #needs_verification 或 #needs_refresh 时，回答必须提醒用户。",
                    "",
                ]
            )
        filename = sector.replace("/", "_").replace(" ", "_")
        (output_dir / f"{filename}.md").write_text("\n".join(lines), encoding="utf-8")

    write_market_markdown(output_dir / "markets", by_market)


def write_market_markdown(output_dir: Path, by_market: dict[str, list[CompanyRecord]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = [
        "# 股票市场分类总览",
        "",
        "- 覆盖市场：A 股、美股、港股。",
        "- 用途：按交易市场检索公司，配合行业文件形成双维分类。",
        "- 注意：跨市场 ADR/H 股/多地上市公司会保留原始市场字段，回答时需说明具体交易市场和代码。",
        "",
    ]
    for market_name in ("A股", "美股", "港股", "其他"):
        items = sorted(by_market.get(market_name, []), key=lambda item: (item.sector, item.ticker))
        overview.append(f"## {market_name}")
        overview.append("")
        for item in items:
            overview.append(
                f"- {item.name}（{item.ticker}，原始市场：{item.market}，行业：{item.sector}）："
                f"{item.reason} [tags: {item.tag_text}]"
            )
        overview.append("")
        if market_name != "其他":
            write_single_market_file(output_dir / f"{market_name}.md", market_name, items)
    (output_dir / "market_universe.md").write_text("\n".join(overview), encoding="utf-8")


def write_single_market_file(path: Path, market_name: str, items: list[CompanyRecord]) -> None:
    lines = [
        f"# {market_name} 上市公司分类",
        "",
        f"- 市场：{market_name}",
        "- 分类维度：行业、公司、代码、资料标签。",
        "- 回答提示：同一公司跨市场上市时，必须说明使用的是哪个市场代码。",
        "",
    ]
    current_sector = ""
    for item in items:
        if item.sector != current_sector:
            current_sector = item.sector
            lines.extend([f"## {current_sector}", ""])
        aliases = "、".join(item.aliases) if item.aliases else "无"
        lines.extend(
            [
                f"### {item.name}",
                "",
                f"- 股票代码：{item.ticker}",
                f"- 原始市场：{item.market}",
                f"- 市场分类：{market_name}",
                f"- 行业分类：{item.sector}",
                f"- 别名：{aliases}",
                f"- 关联逻辑：{item.reason or '待补充'}",
                f"- 资料可信度：{item.reliability}",
                f"- 资料标签：{item.tag_text}",
                f"- 核验提示：{item.verification_note}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def classify_market(market: str, ticker: str) -> str:
    normalized_market = market.upper()
    normalized_ticker = ticker.upper()
    if normalized_ticker.endswith((".SH", ".SZ")):
        return "A股"
    if normalized_ticker.endswith(".HK") or "HK" in normalized_market:
        return "港股"
    if normalized_ticker in {"PRIVATE"}:
        return "其他"
    if normalized_market.startswith("US") or normalized_market in MARKET_ALIASES["美股"]:
        return "美股"
    if normalized_market.startswith("CN") or normalized_market in MARKET_ALIASES["A股"]:
        return "A股"
    return "其他"

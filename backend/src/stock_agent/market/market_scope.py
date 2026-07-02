from __future__ import annotations

import shutil
from pathlib import Path


def prepare_market_rag_source(
    source_root: Path,
    output_root: Path,
    market: str,
    include_dynamic: bool = True,
) -> list[Path]:
    """Create a compact market-specific Markdown source folder for RAG indexing."""

    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    market_file = source_root / "markets" / f"{market}.md"
    if not market_file.exists():
        raise FileNotFoundError(f"市场分类文件不存在：{market_file}")

    target = output_root / market_file.name
    shutil.copyfile(market_file, target)
    written.append(target)

    overview = source_root / "markets" / "market_universe.md"
    if overview.exists():
        target = output_root / overview.name
        shutil.copyfile(overview, target)
        written.append(target)

    status = source_root / "dynamic" / "refresh_status.md"
    if include_dynamic and status.exists():
        target = output_root / "refresh_status.md"
        shutil.copyfile(status, target)
        written.append(target)

    if market == "A股":
        a_share_tech = source_root.parent / "a_share_technology"
        if a_share_tech.exists():
            target_dir = output_root / "a_share_technology"
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(a_share_tech, target_dir, dirs_exist_ok=True)
            written.extend(sorted(target_dir.rglob("*.md")))

    return written

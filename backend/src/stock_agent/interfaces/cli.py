from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from stock_agent.agent.agent_loop import AgentLoop
from stock_agent.core.config import StockConfig
from stock_agent.core.conversation_store import ConversationStore
from stock_agent.core.codex_config import load_codex_model_config
from stock_agent.rag.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_SERVICE_URL,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    EmbeddingConfig,
    create_embedding_client,
)
from stock_agent.rag.knowledge_base import MarkdownKnowledgeBase
from stock_agent.rag.rag_index import RagIndexer
from stock_agent.core.settings import load_settings
from stock_agent.interfaces.terminal import (
    TerminalCommandKind,
    help_text,
    parse_terminal_command,
    render_web_search_results,
    render_search_results,
)
from stock_agent.services.web_search import WebSearchClient
from stock_agent.rag.vector_store import VectorStoreConfig, create_vector_store

app = typer.Typer(help="Run a LangChain-powered stock analysis agent.")
console = Console()
settings = load_settings()


def load_config(config_path: Path | None) -> StockConfig:
    if config_path is None:
        return StockConfig()
    return StockConfig.from_json_file(config_path)


def default_index_path() -> Path:
    return settings.rag_index_path


def default_vector_path() -> Path:
    return settings.rag_vector_path


def default_memory_path() -> Path:
    return settings.memory_path


def default_vector_store_metadata_path() -> Path:
    return settings.vector_store_metadata_path


def market_index_path(market: str) -> Path:
    return Path(".stock_agent") / f"{market}_rag_index.json"


def market_vector_path(market: str) -> Path:
    return Path(".stock_agent") / f"{market}_rag_vectors.json"


def market_source_path(market: str) -> Path:
    return Path("knowledge_base") / f"{market}_rag"


def load_vector_metadata(vector_path: Path | None = None) -> dict:
    path = vector_path or default_vector_path()
    if not path.exists():
        return {}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "vector_store": payload.get("vector_store", "json"),
        "embedding_provider": payload.get("embedding_provider", "openai"),
        "embedding_model": payload.get("embedding_model"),
        "chunk_count": payload.get("chunk_count"),
    }


def save_vector_store_metadata(metadata: dict) -> None:
    import json

    path = default_vector_store_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vector_store_metadata() -> dict:
    import json

    path = default_vector_store_metadata_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_knowledge_base_defaults(
    path: Path | None,
    index_path: Path | None,
) -> tuple[Path, Path]:
    """Use the latest market-specific RAG source when vector metadata points to one."""

    if path is not None:
        return path, index_path or default_index_path()
    if index_path is not None:
        return settings.knowledge_base_path, index_path

    metadata = load_vector_store_metadata()
    source_dir_value = metadata.get("source_dir")
    index_path_value = metadata.get("index_path")
    source_dir = Path(str(source_dir_value)) if source_dir_value else None
    metadata_index_path = Path(str(index_path_value)) if index_path_value else None
    if source_dir and metadata_index_path and source_dir.exists() and metadata_index_path.exists():
        return source_dir, metadata_index_path
    return settings.knowledge_base_path, default_index_path()


def load_embedding_client_for_existing_vectors(vector_path: Path | None = None):
    path = vector_path or default_vector_path()
    metadata = load_vector_store_metadata() or load_vector_metadata(path)
    if not metadata:
        return None

    provider = metadata.get("embedding_provider", "openai")
    model = metadata.get("embedding_model") or (
        DEFAULT_LOCAL_EMBEDDING_MODEL if provider == "local" else DEFAULT_EMBEDDING_MODEL
    )
    try:
        if provider == "service":
            return create_embedding_client(
                EmbeddingConfig(
                    provider="service",
                    model=model,
                    batch_size=64,
                    service_url=metadata.get("embedding_service_url")
                    or os.getenv("EMBEDDING_SERVICE_URL")
                    or DEFAULT_EMBEDDING_SERVICE_URL,
                )
            )
        if provider == "local":
            return create_embedding_client(
                EmbeddingConfig(
                    provider="local",
                    model=model,
                    batch_size=64,
                    local_files_only=True,
                )
            )

        codex_model_config = load_codex_model_config(Path.cwd())
        if not codex_model_config.api_key:
            return None
        return create_embedding_client(
            EmbeddingConfig(
                provider="openai",
                model=model,
                base_url=codex_model_config.base_url,
                api_key=codex_model_config.api_key,
            )
        )
    except Exception as exc:
        console.print(
            "[yellow]向量检索模型加载失败，已回退为 BM25。"
            f"原因：{exc.__class__.__name__}: {exc}[/yellow]"
        )
        return None


def load_vector_store_for_run(vector_path: Path | None = None):
    metadata = load_vector_store_metadata()
    provider = metadata.get("vector_store") or "json"
    if provider == "qdrant":
        try:
            return create_vector_store(
                VectorStoreConfig(
                    provider="qdrant",
                    collection_name=metadata.get("collection_name", settings.qdrant_collection),
                    url=metadata.get("url"),
                    api_key=os.getenv("QDRANT_API_KEY"),
                )
            )
        except Exception as exc:
            console.print(
                "[yellow]Qdrant 向量库连接失败，已回退本地 JSON/BM25。"
                f"原因：{exc.__class__.__name__}: {exc}[/yellow]"
            )
    path = vector_path or default_vector_path()
    if not path.exists():
        return None
    return create_vector_store(VectorStoreConfig(provider="json", path=path))


def default_knowledge_roots(primary: Path) -> list[Path]:
    roots = [primary]
    return roots


def load_knowledge_base(
    path: Path | None,
    index_path: Path | None = None,
    vector_path: Path | None = None,
    embedding_client=None,
    vector_store=None,
) -> MarkdownKnowledgeBase | None:
    kb_path, resolved_index_path = resolve_knowledge_base_defaults(path, index_path)
    if not kb_path.exists():
        return None
    return MarkdownKnowledgeBase(
        kb_path,
        index_path=resolved_index_path,
        vector_path=vector_path or default_vector_path(),
        embedding_client=embedding_client,
        vector_store=vector_store,
    )


def print_result(result) -> None:
    console.print(Panel(result.message, title="股票分析师"))
    if result.guardrail_findings:
        messages = "；".join(finding.message for finding in result.guardrail_findings)
        console.print(f"[yellow]Harness 护栏：{messages}[/yellow]")
    if result.fallback_used:
        console.print("[yellow]模型调用失败，已使用 Harness 降级回复。[/yellow]")


def handle_terminal_command(
    command,
    loop: AgentLoop,
    kb: MarkdownKnowledgeBase | None,
    web_search: WebSearchClient | None,
):
    if command.kind == TerminalCommandKind.HELP:
        console.print(Panel(help_text(), title="帮助"))
        return None
    if command.kind == TerminalCommandKind.KB_SEARCH:
        with console.status("[bold cyan]正在检索本地知识库...[/bold cyan]", spinner="dots"):
            search_result = render_search_results(kb, command.payload)
        console.print(Panel(search_result, title="知识库搜索"))
        return None
    if command.kind == TerminalCommandKind.WEB_SEARCH:
        with console.status("[bold cyan]正在联网搜索...[/bold cyan]", spinner="dots"):
            context = (
                web_search.context_for(command.payload)
                if web_search
                else "未启用联网搜索，请使用 --web-search。"
            )
        console.print(Panel(render_web_search_results(context, command.payload), title="联网搜索"))
        return None
    if command.kind == TerminalCommandKind.TRANSCRIPT:
        transcript = loop.state.transcript() or "暂无研究记录。"
        console.print(Panel(transcript, title="研究记录"))
        return None
    if command.kind == TerminalCommandKind.QUIT:
        console.print("[green]已结束当前分析。[/green]")
        raise typer.Exit()
    with console.status("[bold cyan]正在分析输入、检索知识库并生成下一步研究...[/bold cyan]", spinner="dots"):
        return loop.step(command.payload)


def persist_result(
    store: ConversationStore | None,
    config: StockConfig,
    result,
    event_type: str,
) -> None:
    if store is None:
        return
    store.record_event(
        event_type,
        {
            "message": result.message,
            "advanced": result.advanced,
            "fallback_used": result.fallback_used,
            "stage": result.state.stage.value,
        },
    )
    store.save_state(config, result.state)


@app.command()
def run(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to stock JSON config."),
    knowledge_base: Path | None = typer.Option(
        None,
        "--knowledge-base",
        "-k",
        help="Path to a Markdown knowledge base directory.",
    ),
    offline: bool = typer.Option(False, "--offline", help="Use deterministic local responses."),
    model: str | None = typer.Option(None, "--model", help="OpenAI chat model for LangChain."),
    use_codex_config: bool = typer.Option(
        True,
        "--use-codex-config/--no-use-codex-config",
        help="Read model defaults from Codex config.toml when available.",
    ),
    web_search_enabled: bool = typer.Option(
        False,
        "--web-search/--no-web-search",
        help="Automatically add web search context to harness prompts.",
    ),
    save_conversation: bool = typer.Option(
        True,
        "--save-conversation/--no-save-conversation",
        help="Persist stock research transcript and reusable memory snippets.",
    ),
) -> None:
    """Start an interactive stock analysis session."""

    started_at = perf_counter()
    console.print("[bold cyan]正在启动股票分析 Agent...[/bold cyan]")
    with console.status("[bold cyan]加载环境变量和股票分析配置...[/bold cyan]", spinner="dots"):
        from stock_agent.agent.harness import LangChainStockHarness, ScriptedStockHarness

        load_dotenv()
        stock_config = load_config(config)
        conversation_store = ConversationStore() if save_conversation else None
        codex_model_config = load_codex_model_config(Path.cwd()) if use_codex_config else None

    with console.status("[bold cyan]连接 EmbeddingService / 向量库...[/bold cyan]", spinner="dots"):
        embedding_client = load_embedding_client_for_existing_vectors(default_vector_path())
        vector_store = load_vector_store_for_run(default_vector_path())

    with console.status("[bold cyan]加载知识库索引...[/bold cyan]", spinner="dots"):
        resolved_model = (
            model
            or (codex_model_config.model if codex_model_config else None)
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        kb = load_knowledge_base(
            knowledge_base,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        web_search = WebSearchClient() if web_search_enabled else None
    if kb:
        console.print(
            f"[green]已配置知识库：[/green] {kb.root} "
            f"({kb.estimated_file_count()} 个 Markdown 文件，检索模式：{kb.retrieval_mode})"
        )
    else:
        console.print("[yellow]未加载知识库。[/yellow]")

    api_key = codex_model_config.api_key if codex_model_config else os.getenv("OPENAI_API_KEY")
    if offline or not api_key:
        harness = ScriptedStockHarness(stock_config, knowledge_base=kb)
        console.print("[yellow]正在使用离线脚本 harness。[/yellow]")
    else:
        console.print(f"[green]使用模型：[/green] {resolved_model}")
        if codex_model_config and codex_model_config.provider:
            console.print(f"[green]使用 provider 配置：[/green] {codex_model_config.provider}")
        harness = LangChainStockHarness(
            stock_config,
            knowledge_base=kb,
            web_search=web_search,
            model=resolved_model,
            base_url=codex_model_config.base_url if codex_model_config else None,
            api_key=api_key,
            wire_api=codex_model_config.wire_api if codex_model_config else None,
        )

    loop = AgentLoop(stock_config, harness)
    console.print(f"[dim]启动准备完成，用时 {perf_counter() - started_at:.1f}s。[/dim]")
    with console.status("[bold cyan]正在生成开场研究问题...[/bold cyan]", spinner="dots"):
        result = loop.start()
    print_result(result)
    persist_result(conversation_store, stock_config, result, "start")
    if web_search_enabled:
        hint = "RAG 会自动使用本地知识库；联网搜索也会自动注入上下文。"
    else:
        hint = "RAG 会自动使用本地知识库；联网搜索默认关闭，可用 --web-search 开启。"
    console.print(f"[dim]提示：{hint} 输入 /help 查看命令。[/dim]")

    while not result.state.completed:
        response = typer.prompt("用户")
        command = parse_terminal_command(response)
        next_result = handle_terminal_command(command, loop, kb, web_search)
        if next_result is None:
            continue
        result = next_result
        print_result(result)
        persist_result(conversation_store, stock_config, result, "turn")

    console.print("[green]本轮股票分析完成。[/green]")
    if conversation_store:
        console.print(f"[green]研究记录已保存：[/green] {conversation_store.markdown_path}")
        console.print(f"[green]可检索记忆已保存：[/green] {conversation_store.memory_path}")


@app.command()
def demo() -> None:
    """Run a deterministic non-interactive demo."""

    from stock_agent.agent.harness import ScriptedStockHarness

    config = StockConfig()
    kb = load_knowledge_base(None)
    loop = AgentLoop(config, ScriptedStockHarness(config, knowledge_base=kb))
    result = loop.start()
    print_result(result)

    sample_answers = [
        "请分析宁德时代，周期 6 个月，风险偏好中等。",
        "我更关注锂价下行后的盈利弹性和海外订单。",
        "可以接受 12% 左右回撤，希望给出观察指标。",
    ]
    for answer in sample_answers:
        if result.state.completed:
            break
        console.print(Panel(answer, title="用户"))
        result = loop.step(answer)
        print_result(result)


@app.command()
def index(
    knowledge_base: Path | None = typer.Option(
        None,
        "--knowledge-base",
        "-k",
        help="Path to a Markdown knowledge base directory.",
    ),
    output: Path = typer.Option(
        default_index_path(),
        "--output",
        "-o",
        help="Path to write the persistent RAG index.",
    ),
    chunk_size: int = typer.Option(1800, "--chunk-size", help="Chunk size in characters."),
    embeddings: bool = typer.Option(
        False,
        "--embeddings/--no-embeddings",
        help="Build embedding vectors for hybrid retrieval.",
    ),
    embedding_model: str = typer.Option(
        settings.embedding_model,
        "--embedding-model",
        help="Embedding model for vector retrieval.",
    ),
    embedding_provider: str = typer.Option(
        settings.embedding_provider,
        "--embedding-provider",
        help="Embedding provider: local or openai.",
    ),
    embedding_device: str | None = typer.Option(
        None,
        "--embedding-device",
        help="Optional local embedding device, e.g. cpu, mps, cuda.",
    ),
    embedding_download: bool = typer.Option(
        False,
        "--embedding-download/--no-embedding-download",
        help="Allow local embedding model download during indexing.",
    ),
    vector_store_provider: str = typer.Option(
        settings.vector_store,
        "--vector-store",
        help="Vector store backend: json or qdrant.",
    ),
    qdrant_url: str = typer.Option(
        settings.qdrant_url,
        "--qdrant-url",
        help="Qdrant service URL.",
    ),
    qdrant_collection: str = typer.Option(
        settings.qdrant_collection,
        "--qdrant-collection",
        help="Qdrant collection name.",
    ),
    include_memory: bool = typer.Option(
        True,
        "--include-memory/--no-include-memory",
        help="Include saved stock research memory in the RAG index.",
    ),
) -> None:
    """Build a persistent local RAG index for faster, more stable retrieval."""

    kb_path = knowledge_base or settings.knowledge_base_path
    if not kb_path.exists():
        raise typer.BadParameter(f"知识库目录不存在：{kb_path}")

    roots = default_knowledge_roots(kb_path)
    memory_path = default_memory_path()
    if include_memory and memory_path.exists():
        roots.append(memory_path)

    started_at = perf_counter()
    console.print(f"[dim]正在构建 RAG 索引：{', '.join(str(root) for root in roots)}[/dim]")
    embedding_client = None
    vector_path = None
    vector_store = None
    if embeddings:
        try:
            if embedding_provider not in {"local", "service", "openai"}:
                raise typer.BadParameter("--embedding-provider 仅支持 local、service 或 openai。")
            if embedding_provider == "openai":
                codex_model_config = load_codex_model_config(Path.cwd())
                if not codex_model_config.api_key:
                    raise typer.BadParameter("OpenAI embedding 需要 GATEWAY_API_KEY 或 OPENAI_API_KEY。")
                embedding_client = create_embedding_client(
                    EmbeddingConfig(
                        provider="openai",
                        model=embedding_model,
                        base_url=codex_model_config.base_url,
                        api_key=codex_model_config.api_key,
                    )
                )
            elif embedding_provider == "service":
                embedding_client = create_embedding_client(
                    EmbeddingConfig(
                        provider="service",
                        model=embedding_model,
                        batch_size=64,
                        service_url=os.getenv("EMBEDDING_SERVICE_URL")
                        or settings.embedding_service_url,
                    )
                )
            else:
                embedding_client = create_embedding_client(
                    EmbeddingConfig(
                        provider="local",
                        model=embedding_model,
                        batch_size=64,
                        device=embedding_device,
                        local_files_only=not embedding_download,
                    )
                )
        except typer.BadParameter:
            raise
        except Exception as exc:
            console.print(
                "[yellow]向量模型初始化失败，已回退为仅构建 BM25 索引。"
                f"原因：{exc.__class__.__name__}: {exc}[/yellow]"
            )
        vector_path = default_vector_path()
        if embedding_client:
            if vector_store_provider not in {"json", "qdrant"}:
                raise typer.BadParameter("--vector-store 仅支持 json 或 qdrant。")
            vector_store = create_vector_store(
                VectorStoreConfig(
                    provider=vector_store_provider,
                    path=vector_path,
                    collection_name=qdrant_collection,
                    url=qdrant_url,
                    recreate_collection=True,
                )
            )
            console.print(
                f"[dim]将构建向量索引：{vector_store_provider} "
                f"({embedding_provider}:{embedding_model})[/dim]"
            )

    vector_built = False
    try:
        payload = RagIndexer(
            roots,
            output,
            chunk_size=chunk_size,
            vector_path=vector_path,
            embedding_client=embedding_client,
            vector_store=vector_store,
        ).build()
        vector_built = bool(embeddings and vector_store and vector_store.is_available())
    except Exception as exc:
        if not embeddings:
            raise
        console.print(
            "[yellow]向量索引构建失败，已回退为仅构建 BM25 索引。"
            f"原因：{exc.__class__.__name__}: {exc}[/yellow]"
        )
        payload = RagIndexer(roots, output, chunk_size=chunk_size).build()
    console.print(
        f"[green]索引构建完成：[/green] {output} "
        f"({payload['chunk_count']} chunks，用时 {perf_counter() - started_at:.1f}s)"
    )
    if vector_built and vector_store:
        metadata = {
            **vector_store.metadata,
            "embedding_provider": getattr(embedding_client.config, "provider", "openai"),
            "embedding_model": embedding_client.config.model,
        }
        if getattr(embedding_client.config, "provider", None) == "service":
            metadata["embedding_service_url"] = embedding_client.config.service_url
        save_vector_store_metadata(metadata)
        console.print(f"[green]向量索引构建完成：[/green] {metadata}")


@app.command("collect-universe")
def collect_universe(
    output: Path = typer.Option(
        settings.knowledge_base_path,
        "--output",
        "-o",
        help="Directory to write stock universe Markdown knowledge files.",
    ),
    sectors: str = typer.Option(
        "科技,材料,贵金属,能源,锂电池,银行,具身智能,无人机,机器人,硬件",
        "--sectors",
        help="Comma-separated sector names to collect.",
    ),
    extra_csv: Path | None = typer.Option(
        None,
        "--extra-csv",
        help="Optional CSV with columns: sector,name,ticker,market,aliases,reason,reliability,tags.",
    ),
    remote: bool = typer.Option(
        False,
        "--remote/--no-remote",
        help="Try to enrich with public remote datasets when network is available.",
    ),
) -> None:
    """Create Markdown RAG sources for listed and related companies."""

    from stock_agent.market.stock_universe import collect_company_universe

    sector_list = [item.strip() for item in sectors.split(",") if item.strip()]
    with console.status("[bold cyan]正在生成股票公司 RAG 知识库...[/bold cyan]", spinner="dots"):
        records = collect_company_universe(
            output,
            sectors=sector_list,
            extra_csv=extra_csv,
            include_remote=remote,
        )
    console.print(
        f"[green]公司知识库已生成：[/green] {output} "
        f"({len(records)} companies, sectors={','.join(sector_list)})"
    )
    console.print("[dim]下一步可运行：./stock index --embeddings[/dim]")


@app.command("refresh-kb")
def refresh_kb(
    output: Path = typer.Option(
        settings.knowledge_base_path,
        "--output",
        "-o",
        help="Directory containing stock universe Markdown knowledge files.",
    ),
    sectors: str = typer.Option(
        "科技,材料,贵金属,能源,锂电池,银行,具身智能,无人机,机器人,硬件",
        "--sectors",
        help="Comma-separated sector names to refresh.",
    ),
    max_companies: int = typer.Option(
        30,
        "--max-companies",
        help="Maximum companies to refresh in one run.",
    ),
    quotes: bool = typer.Option(
        True,
        "--quotes/--no-quotes",
        help="Refresh market quote snapshots when public data is available.",
    ),
    announcements: bool = typer.Option(
        True,
        "--announcements/--no-announcements",
        help="Refresh announcements via STOCK_ANNOUNCEMENT_API_URL when configured.",
    ),
    financials: bool = typer.Option(
        True,
        "--financials/--no-financials",
        help="Refresh financial summaries via STOCK_FINANCIALS_API_URL when configured.",
    ),
    rebuild_index: bool = typer.Option(
        True,
        "--rebuild-index/--no-rebuild-index",
        help="Rebuild BM25 RAG index after refresh.",
    ),
) -> None:
    """Refresh announcements, quotes and financial data into the Markdown RAG store."""

    from stock_agent.market.market_refresh import refresh_dynamic_knowledge
    from stock_agent.market.stock_universe import collect_company_universe

    sector_list = [item.strip() for item in sectors.split(",") if item.strip()]
    with console.status("[bold cyan]正在刷新公告/行情/财务数据知识库...[/bold cyan]", spinner="dots"):
        records = collect_company_universe(
            output,
            sectors=sector_list,
            include_remote=False,
        )
        items = refresh_dynamic_knowledge(
            output,
            records,
            max_companies=max_companies,
            include_quotes=quotes,
            include_announcements=announcements,
            include_financials=financials,
        )
        if rebuild_index:
            RagIndexer(default_knowledge_roots(output), default_index_path()).build()
    console.print(
        f"[green]动态知识库刷新完成：[/green] {output / 'dynamic'} "
        f"({len(items)} items, companies={min(len(records), max_companies)})"
    )
    if rebuild_index:
        console.print(f"[green]RAG 索引已重建：[/green] {default_index_path()}")


@app.command("add-knowledge")
def add_knowledge(
    json_file: Path = typer.Argument(..., help="JSON file describing one knowledge item."),
    output: Path = typer.Option(
        settings.knowledge_base_path,
        "--output",
        "-o",
        help="Directory containing stock universe Markdown knowledge files.",
    ),
    rebuild_index: bool = typer.Option(
        True,
        "--rebuild-index/--no-rebuild-index",
        help="Rebuild BM25 RAG index after adding the item.",
    ),
) -> None:
    """Add a manual announcement/data item with verification tags."""

    import json
    from stock_agent.market.market_refresh import write_manual_item

    payload = json.loads(json_file.read_text(encoding="utf-8"))
    path = write_manual_item(output, payload)
    if rebuild_index:
        RagIndexer(default_knowledge_roots(output), default_index_path()).build()
    console.print(f"[green]资料已加入知识库：[/green] {path}")
    if rebuild_index:
        console.print(f"[green]RAG 索引已重建：[/green] {default_index_path()}")


@app.command("build-a-share-tech")
def build_a_share_tech(
    output: Path = typer.Option(
        Path("knowledge_base/a_share_technology"),
        "--output",
        "-o",
        help="Directory to write one Markdown analysis document per A-share technology stock.",
    ),
    max_companies: int | None = typer.Option(
        None,
        "--max-companies",
        help="Limit companies for testing. Omit for all matched A-share technology stocks.",
    ),
    profiles: bool = typer.Option(
        True,
        "--profiles/--no-profiles",
        help="Fetch Eastmoney F10 company profile data for each stock.",
    ),
    announcements: bool = typer.Option(
        True,
        "--announcements/--no-announcements",
        help="Fetch recent announcement titles for each stock.",
    ),
    announcement_limit: int = typer.Option(
        5,
        "--announcement-limit",
        help="Recent announcements to include per stock.",
    ),
    workers: int = typer.Option(
        8,
        "--workers",
        help="Concurrent workers for profile and announcement fetches.",
    ),
    source: str = typer.Option(
        "auto",
        "--source",
        help="Candidate source: auto, eastmoney or cninfo.",
    ),
    rebuild_a_share_index: bool = typer.Option(
        True,
        "--rebuild-a-share-index/--no-rebuild-a-share-index",
        help="After generation, rebuild the A-share market RAG source and BM25 index.",
    ),
) -> None:
    """Build detailed per-stock Markdown documents for all A-share technology companies."""

    from stock_agent.market.a_share_tech import build_a_share_tech_knowledge_base
    from stock_agent.market.market_scope import prepare_market_rag_source

    console.print("[bold cyan]正在抓取 A股科技相关股票并生成逐股分析文档...[/bold cyan]")
    with console.status("[bold cyan]抓取行情列表、公司概况和公告线索...[/bold cyan]", spinner="dots"):
        result = build_a_share_tech_knowledge_base(
            output,
            max_companies=max_companies,
            include_profiles=profiles,
            include_announcements=announcements,
            announcement_limit=announcement_limit,
            workers=workers,
            source_mode=source,
        )

    console.print(
        f"[green]A股科技逐股文档已生成：[/green] {result.output_dir} "
        f"(A股候选={result.total_a_share_count}, 命中={result.matched_count}, "
        f"文档={result.document_count})"
    )
    if result.fallback_reason:
        console.print(
            f"[yellow]候选源已降级为 {result.source_mode}：{result.fallback_reason}[/yellow]"
        )
    if result.category_counts:
        category_text = "；".join(
            f"{name}={count}" for name, count in result.category_counts.items()
        )
        console.print(f"[dim]分类统计：{category_text}[/dim]")
    console.print(
        "[yellow]资料标签已写入文档：第三方行情/F10/公告聚合均需核验，回答时会提示 "
        "#needs_verification / #needs_refresh。[/yellow]"
    )

    if rebuild_a_share_index:
        source_dir = market_source_path("A股")
        files = prepare_market_rag_source(
            source_root=settings.knowledge_base_path,
            output_root=source_dir,
            market="A股",
            include_dynamic=True,
        )
        payload = RagIndexer(source_dir, market_index_path("A股")).build()
        console.print(
            f"[green]A股 RAG 源和 BM25 索引已重建：[/green] {source_dir} "
            f"({len(files)} files, {payload['chunk_count']} chunks)"
        )
        console.print(
            "[dim]写入向量库可继续运行：./stock index-market A股 "
            "--qdrant-collection stock_agent_a_share --embedding-provider local --vector-store qdrant[/dim]"
        )


@app.command("refresh-a-share-tech")
def refresh_a_share_tech(
    output: Path = typer.Option(
        Path("knowledge_base/a_share_technology"),
        "--output",
        "-o",
        help="Directory containing A-share technology per-stock documents.",
    ),
    max_companies: int = typer.Option(
        200,
        "--max-companies",
        help="Maximum companies to refresh in one batch. Use --all for the full manifest.",
    ),
    all_companies: bool = typer.Option(
        False,
        "--all/--batch",
        help="Refresh all A-share technology companies from manifest instead of one batch.",
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        help="Batch offset in manifest order.",
    ),
    quotes: bool = typer.Option(
        True,
        "--quotes/--no-quotes",
        help="Refresh A-share market quote snapshots from Eastmoney when available.",
    ),
    announcements: bool = typer.Option(
        True,
        "--announcements/--no-announcements",
        help="Refresh recent announcement titles from Eastmoney announcement aggregation.",
    ),
    financials: bool = typer.Option(
        True,
        "--financials/--no-financials",
        help="Refresh financial summary via STOCK_FINANCIALS_API_URL when configured.",
    ),
    announcement_limit: int = typer.Option(
        5,
        "--announcement-limit",
        help="Recent announcements to include per stock.",
    ),
    workers: int = typer.Option(
        8,
        "--workers",
        help="Concurrent workers for refresh requests.",
    ),
    rebuild_a_share_index: bool = typer.Option(
        True,
        "--rebuild-a-share-index/--no-rebuild-a-share-index",
        help="After refresh, rebuild the A-share market RAG source and BM25 index.",
    ),
) -> None:
    """Refresh dynamic quote, announcement and financial documents for A-share tech stocks."""

    from stock_agent.market.a_share_refresh import refresh_a_share_tech_dynamic
    from stock_agent.market.market_scope import prepare_market_rag_source

    refresh_limit = None if all_companies else max_companies
    console.print("[bold cyan]正在刷新 A股科技逐股动态资料...[/bold cyan]")
    with console.status("[bold cyan]拉取行情、公告和财务数据并写入动态 RAG...[/bold cyan]", spinner="dots"):
        result = refresh_a_share_tech_dynamic(
            output,
            max_companies=refresh_limit,
            offset=offset,
            include_quotes=quotes,
            include_announcements=announcements,
            include_financials=financials,
            announcement_limit=announcement_limit,
            workers=workers,
        )

    console.print(
        f"[green]A股科技动态资料已刷新：[/green] {result.dynamic_dir} "
        f"(清单={result.total_manifest_count}, 本次={result.refreshed_count}, "
        f"行情={result.quote_count}, 公告={result.announcement_count}, 财务={result.financial_count})"
    )
    console.print(
        "[yellow]动态资料已带 #needs_verification / #needs_refresh；回答时必须提示核验公告原文、最新行情和财报。[/yellow]"
    )

    if rebuild_a_share_index:
        source_dir = market_source_path("A股")
        files = prepare_market_rag_source(
            source_root=settings.knowledge_base_path,
            output_root=source_dir,
            market="A股",
            include_dynamic=True,
        )
        payload = RagIndexer(source_dir, market_index_path("A股")).build()
        console.print(
            f"[green]A股 RAG 源和 BM25 索引已重建：[/green] {source_dir} "
            f"({len(files)} files, {payload['chunk_count']} chunks)"
        )
        console.print(
            "[dim]如需同步向量库，继续运行：./stock index-market A股 "
            "--qdrant-collection stock_agent_a_share --embedding-provider local --vector-store qdrant[/dim]"
        )


@app.command("daily-picks")
def daily_picks(
    output: Path = typer.Option(
        Path("knowledge_base/a_share_technology"),
        "--output",
        "-o",
        help="Directory containing A-share technology per-stock documents.",
    ),
    portfolio: Path | None = typer.Option(
        None,
        "--portfolio",
        help="Optional CSV/JSON portfolio. Columns: code,name,shares,cost_price,notes.",
    ),
    max_candidates: int = typer.Option(
        800,
        "--max-candidates",
        help="Maximum companies to scan before ranking. Use --all for the full manifest.",
    ),
    all_companies: bool = typer.Option(
        False,
        "--all/--sample",
        help="Scan all A-share technology companies from manifest.",
    ),
    top_k: int = typer.Option(30, "--top-k", help="Number of daily candidates to output."),
    workers: int = typer.Option(8, "--workers", help="Concurrent quote refresh workers."),
    timeout: float = typer.Option(8.0, "--timeout", help="Quote request timeout in seconds."),
    categories: str = typer.Option(
        "",
        "--categories",
        help="Comma-separated category filters, e.g. 半导体,机器人,无人机.",
    ),
    rebuild_a_share_index: bool = typer.Option(
        True,
        "--rebuild-a-share-index/--no-rebuild-a-share-index",
        help="After generating daily picks, rebuild the A-share market RAG source and BM25 index.",
    ),
    sync_vector_store: bool = typer.Option(
        True,
        "--sync-vector-store/--no-sync-vector-store",
        help="Also rebuild the A-share vector store after generating daily picks.",
    ),
    vector_store_provider: str = typer.Option(
        "qdrant",
        "--vector-store",
        help="Vector store backend for sync: qdrant or json.",
    ),
    qdrant_collection: str = typer.Option(
        "stock_agent_a_share",
        "--qdrant-collection",
        help="Qdrant collection name for A-share RAG.",
    ),
    qdrant_url: str = typer.Option(
        settings.qdrant_url,
        "--qdrant-url",
        help="Qdrant service URL.",
    ),
    embedding_provider: str = typer.Option(
        "local",
        "--embedding-provider",
        help="Embedding provider for vector sync: local, service or openai.",
    ),
    embedding_model: str = typer.Option(
        settings.embedding_model,
        "--embedding-model",
        help="Embedding model for vector retrieval.",
    ),
    embedding_device: str | None = typer.Option(
        None,
        "--embedding-device",
        help="Optional local embedding device, e.g. cpu, mps, cuda.",
    ),
) -> None:
    """Generate daily A-share quant picks, portfolio diagnostics and refreshed RAG artifacts."""

    from stock_agent.market.daily_picks import build_daily_picks
    from stock_agent.market.market_scope import prepare_market_rag_source

    category_list = [item.strip() for item in categories.split(",") if item.strip()]
    candidate_limit = None if all_companies else max_candidates
    console.print("[bold cyan]正在生成每日量化推荐观察池...[/bold cyan]")
    with console.status("[bold cyan]拉取行情、计算评分并分析持仓...[/bold cyan]", spinner="dots"):
        result = build_daily_picks(
            output,
            portfolio_path=portfolio,
            max_candidates=candidate_limit,
            top_k=top_k,
            workers=workers,
            timeout=timeout,
            categories=category_list,
        )

    console.print(
        f"[green]每日候选报告已生成：[/green] {result.report_path} "
        f"(清单={result.universe_count}, 扫描={result.scanned_count}, "
        f"候选={result.picked_count}, 持仓={result.portfolio_count})"
    )
    console.print(f"[green]latest 已更新：[/green] {result.latest_path}")
    console.print(
        "[yellow]报告已带 #quant_candidate / #needs_verification / #needs_refresh；"
        "回答推荐时会提醒核验公告、财报和最新行情。[/yellow]"
    )

    source_dir = market_source_path("A股")
    if sync_vector_store:
        try:
            index_market(
                "A股",
                source_root=settings.knowledge_base_path,
                output_source=source_dir,
                vector_store_provider=vector_store_provider,
                qdrant_collection=qdrant_collection,
                qdrant_url=qdrant_url,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
                include_dynamic=True,
            )
        except typer.BadParameter:
            raise
        except Exception as exc:
            console.print(
                "[yellow]向量库同步失败，已保留每日报告；将回退重建 A股 BM25 索引。"
                f"原因：{exc.__class__.__name__}: {exc}[/yellow]"
            )
            if rebuild_a_share_index:
                files = prepare_market_rag_source(
                    source_root=settings.knowledge_base_path,
                    output_root=source_dir,
                    market="A股",
                    include_dynamic=True,
                )
                payload = RagIndexer(source_dir, market_index_path("A股")).build()
                console.print(
                    f"[green]A股 RAG 源和 BM25 索引已重建：[/green] {source_dir} "
                    f"({len(files)} files, {payload['chunk_count']} chunks)"
                )
            console.print(
                "[dim]确认 Qdrant/Embedding 可用后可重试：./stock index-market A股 "
                "--qdrant-collection stock_agent_a_share --embedding-provider local "
                "--vector-store qdrant[/dim]"
            )
    elif rebuild_a_share_index:
        files = prepare_market_rag_source(
            source_root=settings.knowledge_base_path,
            output_root=source_dir,
            market="A股",
            include_dynamic=True,
        )
        payload = RagIndexer(source_dir, market_index_path("A股")).build()
        console.print(
            f"[green]A股 RAG 源和 BM25 索引已重建：[/green] {source_dir} "
            f"({len(files)} files, {payload['chunk_count']} chunks)"
        )


@app.command("index-market")
def index_market(
    market: str = typer.Argument("A股", help="Market name, e.g. A股, 美股, 港股."),
    source_root: Path = typer.Option(
        settings.knowledge_base_path,
        "--source-root",
        help="Root stock universe knowledge directory.",
    ),
    output_source: Path | None = typer.Option(
        None,
        "--output-source",
        help="Market-specific source directory to create.",
    ),
    vector_store_provider: str = typer.Option(
        "qdrant",
        "--vector-store",
        help="Vector store backend: qdrant or json.",
    ),
    qdrant_collection: str | None = typer.Option(
        None,
        "--qdrant-collection",
        help="Qdrant collection name. Defaults to stock_agent_<market>.",
    ),
    qdrant_url: str = typer.Option(
        settings.qdrant_url,
        "--qdrant-url",
        help="Qdrant service URL.",
    ),
    embedding_provider: str = typer.Option(
        "local",
        "--embedding-provider",
        help="Embedding provider: local, service or openai.",
    ),
    embedding_model: str = typer.Option(
        settings.embedding_model,
        "--embedding-model",
        help="Embedding model for vector retrieval.",
    ),
    embedding_device: str | None = typer.Option(
        None,
        "--embedding-device",
        help="Optional local embedding device, e.g. cpu, mps, cuda.",
    ),
    include_dynamic: bool = typer.Option(
        True,
        "--include-dynamic/--no-include-dynamic",
        help="Include dynamic refresh status or data files.",
    ),
) -> None:
    """Build a market-specific RAG index and persist it into a vector database."""

    from stock_agent.market.market_scope import prepare_market_rag_source

    source_dir = output_source or market_source_path(market)
    files = prepare_market_rag_source(
        source_root=source_root,
        output_root=source_dir,
        market=market,
        include_dynamic=include_dynamic,
    )

    if embedding_provider not in {"local", "service", "openai"}:
        raise typer.BadParameter("--embedding-provider 仅支持 local、service 或 openai。")
    if vector_store_provider not in {"qdrant", "json"}:
        raise typer.BadParameter("--vector-store 仅支持 qdrant 或 json。")

    if embedding_provider == "service":
        embedding_client = create_embedding_client(
            EmbeddingConfig(
                provider="service",
                model=embedding_model,
                batch_size=64,
                service_url=os.getenv("EMBEDDING_SERVICE_URL") or settings.embedding_service_url,
            )
        )
    elif embedding_provider == "openai":
        codex_model_config = load_codex_model_config(Path.cwd())
        if not codex_model_config.api_key:
            raise typer.BadParameter("OpenAI embedding 需要 GATEWAY_API_KEY 或 OPENAI_API_KEY。")
        embedding_client = create_embedding_client(
            EmbeddingConfig(
                provider="openai",
                model=embedding_model,
                base_url=codex_model_config.base_url,
                api_key=codex_model_config.api_key,
            )
        )
    else:
        embedding_client = create_embedding_client(
            EmbeddingConfig(
                provider="local",
                model=embedding_model,
                batch_size=64,
                device=embedding_device,
                local_files_only=True,
            )
        )

    collection = qdrant_collection or f"stock_agent_{market}".replace("股", "_share")
    vector_store = create_vector_store(
        VectorStoreConfig(
            provider=vector_store_provider,
            path=market_vector_path(market),
            collection_name=collection,
            url=qdrant_url,
            recreate_collection=True,
        )
    )
    payload = RagIndexer(
        source_dir,
        market_index_path(market),
        vector_path=market_vector_path(market),
        embedding_client=embedding_client,
        vector_store=vector_store,
    ).build()
    metadata = {
        **vector_store.metadata,
        "market": market,
        "source_dir": str(source_dir),
        "source_files": [str(path) for path in files],
        "index_path": str(market_index_path(market)),
        "embedding_provider": embedding_client.config.provider,
        "embedding_model": embedding_client.config.model,
        "chunk_count": payload["chunk_count"],
    }
    save_vector_store_metadata(metadata)
    source_file_count = len(files)
    concise_metadata = {
        "vector_store": metadata["vector_store"],
        "collection_name": metadata.get("collection_name"),
        "market": market,
        "source_dir": str(source_dir),
        "source_file_count": source_file_count,
        "index_path": str(market_index_path(market)),
        "embedding_provider": embedding_client.config.provider,
        "embedding_model": embedding_client.config.model,
        "chunk_count": payload["chunk_count"],
    }
    console.print(
        f"[green]{market} RAG 已写入向量库：[/green] {concise_metadata} "
        f"({payload['chunk_count']} chunks)"
    )


@app.command("embedding-service")
def embedding_service(
    host: str = typer.Option("127.0.0.1", "--host", help="Embedding service host."),
    port: int = typer.Option(18210, "--port", help="Embedding service port."),
    model: str = typer.Option(
        settings.embedding_model,
        "--model",
        help="Local SentenceTransformers embedding model.",
    ),
    batch_size: int = typer.Option(64, "--batch-size", help="Embedding batch size."),
    device: str | None = typer.Option(None, "--device", help="Optional device: cpu, mps, cuda."),
    embedding_download: bool = typer.Option(
        False,
        "--embedding-download/--no-embedding-download",
        help="Allow model download when service starts.",
    ),
) -> None:
    """Run a local HTTP EmbeddingService."""

    import uvicorn
    from stock_agent.services.embedding_service import EmbeddingServiceSettings, create_app

    service_app = create_app(
        EmbeddingServiceSettings(
            model=model,
            batch_size=batch_size,
            device=device,
            local_files_only=not embedding_download,
        )
    )
    console.print(
        f"[green]EmbeddingService 启动中：[/green] http://{host}:{port} "
        f"({model})"
    )
    uvicorn.run(service_app, host=host, port=port)


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", "--host", help="API host."),
    port: int = typer.Option(18220, "--port", help="API port."),
) -> None:
    """Run the local HTTP API for desktop clients."""

    import uvicorn
    from stock_agent.api import create_app

    load_dotenv()
    console.print(f"[green]Stock Agent API 启动中：[/green] http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def doctor() -> None:
    """Check local production dependencies and RAG artifacts."""

    from stock_agent.interfaces.doctor import run_doctor

    results = run_doctor(
        index_path=default_index_path(),
        vector_store_metadata_path=default_vector_store_metadata_path(),
        embedding_service_url=settings.embedding_service_url,
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.qdrant_collection,
    )
    failed = False
    for result in results:
        status = "[green]OK[/green]" if result.ok else "[red]FAIL[/red]"
        console.print(f"{status} {result.name}: {result.message}")
        failed = failed or not result.ok
    if failed:
        raise typer.Exit(code=1)


@app.command("eval-rag")
def eval_rag(
    cases: Path = typer.Option(
        Path("backend/tests/fixtures/rag_eval_cases.json"),
        "--cases",
        help="Path to RAG evaluation cases JSON.",
    ),
    top_k: int = typer.Option(4, "--top-k", help="Number of retrieved chunks to evaluate."),
) -> None:
    """Run a lightweight RAG retrieval regression evaluation."""

    from stock_agent.rag.rag_eval import load_eval_cases, run_rag_eval

    embedding_client = load_embedding_client_for_existing_vectors(default_vector_path())
    vector_store = load_vector_store_for_run(default_vector_path())
    kb = load_knowledge_base(None, embedding_client=embedding_client, vector_store=vector_store)
    eval_cases = load_eval_cases(cases)
    results = run_rag_eval(kb, eval_cases, top_k=top_k)
    failed = False
    for result in results:
        status = "[green]PASS[/green]" if result.ok else "[red]FAIL[/red]"
        console.print(f"{status} {result.query}")
        if result.missing_sources:
            console.print(f"  missing sources: {', '.join(result.missing_sources)}")
        if result.missing_terms:
            console.print(f"  missing terms: {', '.join(result.missing_terms)}")
        failed = failed or not result.ok
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

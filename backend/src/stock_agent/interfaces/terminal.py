from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stock_agent.rag.knowledge_base import MarkdownKnowledgeBase


class TerminalCommandKind(str, Enum):
    ANSWER = "answer"
    HELP = "help"
    KB_SEARCH = "kb_search"
    WEB_SEARCH = "web_search"
    QUIT = "quit"
    TRANSCRIPT = "transcript"


@dataclass(frozen=True)
class TerminalCommand:
    kind: TerminalCommandKind
    payload: str = ""


def parse_terminal_command(text: str) -> TerminalCommand:
    stripped = text.strip()
    if not stripped:
        return TerminalCommand(TerminalCommandKind.ANSWER, stripped)
    if stripped in {"/q", "/quit", "退出", "结束"}:
        return TerminalCommand(TerminalCommandKind.QUIT)
    if stripped in {"/help", "帮助"}:
        return TerminalCommand(TerminalCommandKind.HELP)
    if stripped in {"/transcript", "记录"}:
        return TerminalCommand(TerminalCommandKind.TRANSCRIPT)
    if stripped.startswith("/kb "):
        return TerminalCommand(TerminalCommandKind.KB_SEARCH, stripped.removeprefix("/kb ").strip())
    if stripped.startswith("/search "):
        return TerminalCommand(TerminalCommandKind.KB_SEARCH, stripped.removeprefix("/search ").strip())
    if stripped.startswith("搜索 "):
        return TerminalCommand(TerminalCommandKind.KB_SEARCH, stripped.removeprefix("搜索 ").strip())
    if stripped.startswith("/web "):
        return TerminalCommand(TerminalCommandKind.WEB_SEARCH, stripped.removeprefix("/web ").strip())
    if stripped.startswith("联网搜索 "):
        return TerminalCommand(TerminalCommandKind.WEB_SEARCH, stripped.removeprefix("联网搜索 ").strip())
    return TerminalCommand(TerminalCommandKind.ANSWER, stripped)


def help_text() -> str:
    return """可用命令：
- 直接输入：描述股票、行业、投资周期、风险偏好或澄清问题
- `/kb 关键词`、`/search 关键词` 或 `搜索 关键词`：手动检索本地股票知识库
- `/web 关键词` 或 `联网搜索 关键词`：手动联网搜索，用于调试联网上下文
- `/transcript` 或 `记录`：查看当前研究记录
- `/help` 或 `帮助`：查看帮助
- `/quit`、`/q`、`退出`：结束当前分析"""


def render_search_results(kb: MarkdownKnowledgeBase | None, query: str, top_k: int = 3) -> str:
    if kb is None:
        return "未加载知识库，无法检索。"
    if not query:
        return "请输入要搜索的关键词，例如：`/search RAG 检索优化`。"
    chunks = kb.search(query, top_k=top_k)
    if not chunks:
        return f"没有找到与“{query}”相关的知识库片段。"

    sections: list[str] = [f"知识库搜索：{query}"]
    for index, chunk in enumerate(chunks, start=1):
        excerpt = chunk.content.replace("\n", " ").strip()
        if len(excerpt) > 260:
            excerpt = excerpt[:260].rstrip() + "..."
        sections.append(
            f"{index}. {chunk.heading}\n"
            f"   来源：{chunk.source}\n"
            f"   摘要：{excerpt}"
        )
    return "\n\n".join(sections)


def render_web_search_results(search_context: str, query: str) -> str:
    if not query:
        return "请输入要联网搜索的关键词，例如：`/web RAG 最新优化`。"
    return f"联网搜索：{query}\n\n{search_context}"

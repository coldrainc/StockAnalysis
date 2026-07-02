"""Stock Agent package.

The implementation is organized into subpackages:
- agent: loop, LLM harness, prompts, state, guardrails
- core: config, settings, model-provider config, conversation storage
- rag: embeddings, vector store, RAG index, knowledge base
- market: stock universe, market refresh, A-share builders, daily picks
- services: FastAPI, embedding service, web search
- interfaces: CLI, terminal UI, diagnostics
"""

from importlib import import_module
import sys

__all__ = ["__version__"]

__version__ = "0.1.0"

_COMPAT_MODULES = {
    "agent_loop": "stock_agent.agent.agent_loop",
    "guardrails": "stock_agent.agent.guardrails",
    "harness": "stock_agent.agent.harness",
    "harness_result": "stock_agent.agent.harness_result",
    "multi_agent": "stock_agent.agent.multi_agent",
    "prompt_orchestrator": "stock_agent.agent.prompt_orchestrator",
    "state": "stock_agent.agent.state",
    "codex_config": "stock_agent.core.codex_config",
    "config": "stock_agent.core.config",
    "conversation_store": "stock_agent.core.conversation_store",
    "settings": "stock_agent.core.settings",
    "embedding": "stock_agent.rag.embedding",
    "knowledge_base": "stock_agent.rag.knowledge_base",
    "knowledge_types": "stock_agent.rag.knowledge_types",
    "rag_eval": "stock_agent.rag.rag_eval",
    "rag_index": "stock_agent.rag.rag_index",
    "vector_store": "stock_agent.rag.vector_store",
    "a_share_refresh": "stock_agent.market.a_share_refresh",
    "a_share_tech": "stock_agent.market.a_share_tech",
    "daily_picks": "stock_agent.market.daily_picks",
    "market_refresh": "stock_agent.market.market_refresh",
    "market_scope": "stock_agent.market.market_scope",
    "stock_universe": "stock_agent.market.stock_universe",
    "embedding_service": "stock_agent.services.embedding_service",
    "web_search": "stock_agent.services.web_search",
    "doctor": "stock_agent.interfaces.doctor",
    "terminal": "stock_agent.interfaces.terminal",
}

for legacy_name, target in _COMPAT_MODULES.items():
    sys.modules.setdefault(f"{__name__}.{legacy_name}", import_module(target))

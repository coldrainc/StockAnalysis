# Stock Agent Backend

Python 后端子工程，负责股票分析 Agent 的核心能力：

- AgentLoop / AnalysisState
- LangChainStockHarness / ScriptedStockHarness
- HarnessGuardrails
- RAG 索引、检索、评测
- EmbeddingService
- Qdrant / JSON 向量库适配
- 本地 FastAPI API
- CLI 命令和 Python 测试

## Install

从仓库根目录安装：

```bash
.venv/bin/python -m pip install -e "backend[dev]"
```

也可以直接使用根目录封装：

```bash
make install
```

## Test

```bash
cd backend
../.venv/bin/python -m pytest
```

或从根目录运行：

```bash
make test
```

## Layout

```text
backend/
  pyproject.toml
  examples/
    stock_config.json
  src/stock_agent/
    agent/          # AgentLoop、LLM Harness、Prompt 编排、状态和护栏
    core/           # 配置、settings、模型 provider 配置、会话存储
    rag/            # Embedding、知识库、RAG index、向量库、RAG 评测
    market/         # 股票 universe、行情刷新、A股科技文档、每日候选池
    services/       # FastAPI、本地 embedding service、web search client
    interfaces/     # Typer CLI、终端交互、doctor 检查
    api.py          # 兼容入口，转发到 services/api.py
    cli.py          # 兼容入口，转发到 interfaces/cli.py
  tests/
```

新增后端能力时优先放入对应子包；只有需要兼容外部入口时，才在 `stock_agent/api.py` 或
`stock_agent/cli.py` 保留薄转发层。

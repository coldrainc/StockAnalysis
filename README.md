# Stock Agent

股票量化分析、持仓诊断与推荐观察池 Agent。

- `AgentLoop`：显式控制股票研究阶段：信息收集、基本面、催化剂、风险、推荐结论。
- `LangChainStockHarness`：封装模型调用、提示词、RAG 上下文和联网搜索上下文。
- `MarkdownKnowledgeBase` / `RagIndexer`：Markdown 公司库、BM25、可选 embedding hybrid RAG。
- `EmbeddingService`：本地 SentenceTransformers embedding 服务，默认 `BAAI/bge-small-zh-v1.5`。
- `Qdrant` / JSON vector store：支持生产化向量存储，也可离线使用本地 JSON。
- `collect-universe`：生成科技、材料、贵金属、能源、锂电池、银行、具身智能、无人机、机器人、硬件相关上市公司和关联公司的 RAG 知识库，并按 A 股、美股、港股分类。
- `build-a-share-tech`：抓取 A 股科技相关股票，按“每只股票一篇完整分析文档”生成 RAG 底稿。
- `refresh-kb`：定时刷新公告、行情和财务数据，写入带可信度标签的动态知识库。
- `daily-picks`：每日生成 A 股量化推荐观察池、持仓诊断报告，并刷新 A 股 RAG/向量库。

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "backend[dev]"
cp .env.example .env
```

模型配置：

```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini

# DeepSeek（OpenAI 兼容接口）
STOCK_MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

也可以用通用环境变量覆盖：

```bash
STOCK_MODEL=deepseek-v4-flash
STOCK_MODEL_BASE_URL=https://api.deepseek.com
STOCK_MODEL_API_KEY_ENV=DEEPSEEK_API_KEY
```

生成公司知识库并构建索引：

```bash
./stock collect-universe
./stock build-a-share-tech
./stock refresh-kb
./stock daily-picks --no-sync-vector-store
./stock index
```

离线 demo：

```bash
./stock demo
```

交互式股票分析：

```bash
./stock
```

Hybrid RAG：

```bash
docker compose up -d
./stock embedding-service
./stock index --embeddings --embedding-provider service --vector-store qdrant
./stock
```

只把 A 股数据写入向量数据库：

```bash
./stock collect-universe
./stock build-a-share-tech
./stock index-market A股 \
  --qdrant-collection stock_agent_a_share \
  --embedding-provider local \
  --vector-store qdrant
```

这会生成 A 股专用 RAG 源 `knowledge_base/A股_rag`、逐股分析文档源 `knowledge_base/a_share_technology/companies`、索引 `.stock_agent/A股_rag_index.json`，并写入 Qdrant collection `stock_agent_a_share`。

每日量化候选池和持仓分析：

```bash
./stock daily-picks \
  --portfolio portfolio.csv \
  --max-candidates 800 \
  --top-k 30
```

`daily-picks` 会生成 `knowledge_base/a_share_technology/daily/YYYY-MM-DD_daily_picks.md` 和 `latest_daily_picks.md`，然后默认重建 `knowledge_base/A股_rag`、`.stock_agent/A股_rag_index.json` 并尝试同步 Qdrant 向量库。若本机没有启动 Qdrant 或 embedding 模型不可用，命令会保留每日报告并回退到 BM25 索引；确认服务可用后可手动执行：

```bash
./stock index-market A股 \
  --qdrant-collection stock_agent_a_share \
  --embedding-provider local \
  --vector-store qdrant
```

持仓 CSV 支持英文或中文列名：

```csv
code,name,shares,cost_price,notes
300750,宁德时代,100,180,核心持仓
```

也可以使用中文列名：

```csv
股票代码,股票名称,持仓数量,成本价,备注
300750,宁德时代,100,180,核心持仓
```

桌面端：

```bash
./stock desktop
```

`./stock` 默认仍是终端交互模式；需要 GUI 时使用 `./stock desktop` 或 `./stock gui`。桌面端会自动检查 StockAgent 专用本地 API `http://127.0.0.1:18220`，未启动时会在后台拉起 API，日志写入 `.stock_agent/api.log`。如需临时改端口，可设置 `STOCK_AGENT_API_PORT` 或 `STOCK_AGENT_API_URL`。

## Portable Package

项目可以用 Git 已提交内容打包，安装依赖和运行状态不会进入压缩包：

```bash
git archive --format=tar.gz --prefix=StockAgent/ HEAD -o StockAgent-portable.tar.gz
```

在另一台机器解压后重新安装依赖：

```bash
tar -xzf StockAgent-portable.tar.gz
cd StockAgent
cp .env.example .env
make install
npm --prefix apps/desktop install
./stock desktop
```

压缩包不包含 `.venv/`、`node_modules/`、`.stock_agent/`、`.env` 等本地依赖、缓存、运行状态和密钥文件；知识库中已提交的 Markdown RAG 文档会随包一起带走。

## Data Flow

```text
collect-universe -> knowledge_base/stock_universe/*.md
collect-universe -> knowledge_base/stock_universe/markets/A股.md
collect-universe -> knowledge_base/stock_universe/markets/美股.md
collect-universe -> knowledge_base/stock_universe/markets/港股.md
refresh-kb -> knowledge_base/stock_universe/dynamic/*.md
build-a-share-tech -> knowledge_base/a_share_technology/companies/*.md
Markdown -> chunk -> BM25 index -> .stock_agent/rag_index.json
Markdown -> embedding -> .stock_agent/rag_vectors.json or Qdrant
query -> BM25 + vector recall -> MMR -> prompt context -> stock research answer
```

## A股科技逐股分析库

全量生成 A 股科技相关股票的逐股文档：

```bash
./stock build-a-share-tech --source cninfo --workers 12 --no-announcements
```

为了快速试跑，可以限制数量：

```bash
./stock build-a-share-tech --source cninfo --max-companies 20 --workers 4 --announcement-limit 2
```

`--source auto` 会优先尝试东方财富行情列表，失败时降级到巨潮资讯全 A 股列表；`--source cninfo` 更适合稳定全量跑批。全量跑批时建议先 `--no-announcements` 生成公司画像，再用动态刷新链路补公告和行情，避免公告接口限流影响主库生成。

输出结构：

- `knowledge_base/a_share_technology/overview.md`：A 股科技股票总览和分类统计。
- `knowledge_base/a_share_technology/companies/*.md`：每只股票一篇完整分析文档，包含公司画像、科技相关性、行情估值、基本面框架、公告线索、风险清单、跟踪指标。
- `knowledge_base/a_share_technology/categories/*.md`：按半导体、AI 算力、机器人、无人机、硬件、新能源科技等方向分类。
- `knowledge_base/a_share_technology/manifest.json`：生成清单、分类、命中原因和文档路径。

生成后运行 `./stock index-market A股 ...` 会自动把逐股分析文档纳入 A 股 RAG 源并写入向量库。

动态刷新 A 股科技逐股资料：

```bash
./stock refresh-a-share-tech --all --workers 8 --announcement-limit 3
./stock index-market A股 \
  --qdrant-collection stock_agent_a_share \
  --embedding-provider local \
  --vector-store qdrant
```

为了避免公告接口限流，也可以按批次刷新：

```bash
./stock refresh-a-share-tech --max-companies 300 --offset 0 --workers 6
./stock refresh-a-share-tech --max-companies 300 --offset 300 --workers 6
```

动态刷新输出在 `knowledge_base/a_share_technology/dynamic`，会随 `knowledge_base/A股_rag` 一起进入 BM25/RAG 源；同步到 Qdrant 时继续执行 `index-market A股`。

## Daily Quant Pipeline

建议每天盘后或盘前执行一次完整流水线：

```bash
./stock refresh-a-share-tech --max-companies 800 --workers 8 --announcement-limit 3
./stock daily-picks --portfolio portfolio.csv --max-candidates 800 --top-k 30
```

需要全量覆盖时：

```bash
./stock refresh-a-share-tech --all --workers 8 --announcement-limit 3
./stock daily-picks --all --portfolio portfolio.csv --top-k 50
```

量化观察池使用规则评分，不承诺收益。它会综合主题强度、涨跌幅、成交额、换手率、PE/PB 粗约束和风险惩罚，输出“强关注 / 观察 / 轻仓跟踪 / 暂不优先”等级。所有候选都会带 `#quant_candidate`、`#third_party_dataset`、`#needs_verification`、`#needs_refresh` 标签，回答推荐时会提醒核验公告原文、财报、最新行情和你的仓位约束。

## Dynamic Refresh

动态刷新命令会尽力拉取行情快照，并可通过环境变量接入公告和财务数据 API：

```bash
STOCK_ANNOUNCEMENT_API_URL=https://your-api/announcements
STOCK_FINANCIALS_API_URL=https://your-api/financials
./stock refresh-kb --max-companies 50
```

定时刷新可以用 cron，例如每 30 分钟刷新一次：

```cron
*/30 * * * * cd /path/to/StockAgent && ./stock refresh-kb --max-companies 50 >> .stock_agent/refresh.log 2>&1
0 */2 * * * cd /path/to/StockAgent && ./stock refresh-a-share-tech --max-companies 300 --offset 0 >> .stock_agent/a_share_refresh.log 2>&1
30 16 * * 1-5 cd /path/to/StockAgent && ./stock daily-picks --portfolio portfolio.csv --max-candidates 800 --top-k 30 >> .stock_agent/daily_picks.log 2>&1
```

资料标签会进入 RAG：

- `#official_or_exchange`：官方、交易所或你配置的权威接口。
- `#third_party_dataset`：第三方行情或数据源，只作为线索。
- `#user_supplied`：手动加入资料。
- `#needs_refresh`：需要定时刷新。
- `#needs_verification`：真实性或口径存疑，回答时必须提醒。

## Market Classification

知识库同时按行业和交易市场分类：

- 行业文件：`knowledge_base/stock_universe/科技.md`、`机器人.md`、`硬件.md` 等。
- 市场文件：`knowledge_base/stock_universe/markets/A股.md`、`美股.md`、`港股.md`。

回答时如果同一公司有 A/H/ADR 等多地代码，应明确说明使用的是哪个市场和代码。

## Universe CSV

可以用 CSV 扩展公司库：

```csv
sector,name,ticker,market,aliases,reason
科技,示例公司,000001.SZ,CN,别名1|别名2,关联逻辑
```

导入：

```bash
./stock collect-universe --extra-csv your_companies.csv
```

手动加入资料：

```json
{
  "title": "某公司公告摘要",
  "category": "公告",
  "company": "示例公司",
  "ticker": "000001.SZ",
  "market": "CN",
  "content": "公告摘要正文",
  "source": "https://example.com/announcement",
  "reliability": "user_supplied",
  "tags": ["#user_supplied", "#needs_verification"]
}
```

```bash
./stock add-knowledge item.json
```

免责声明：本项目输出仅供研究辅助，不构成投资建议或收益承诺；带 `#needs_verification` 或 `#needs_refresh` 的资料必须核验后再使用。

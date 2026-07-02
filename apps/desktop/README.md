# Stock Agent Desktop

Electron 桌面端子工程，负责量化工作台、应用内报告页、知识库状态页、持仓页、对话 UI 和本地 API 调用。

## Install

从仓库根目录安装桌面端依赖：

```bash
npm --prefix apps/desktop install
```

依赖会安装在：

```text
apps/desktop/node_modules
```

## Run

桌面端依赖本地 API：

```bash
./stock desktop
```

也可以使用 Makefile：

```bash
make desktop
```

`./stock desktop` 会自动安装缺失的 Electron 依赖、检查本地 API，并在 API 未启动时拉起 StockAgent 专用地址 `http://127.0.0.1:18220`。终端模式仍然是默认的 `./stock`。如需临时改端口，可设置 `STOCK_AGENT_API_PORT` 或 `STOCK_AGENT_API_URL`。

桌面端页面：

- 工作台：展示每日候选 Top、扫描数量、RAG chunks，并支持一键刷新候选池。
- 每日报告：在应用内渲染 `latest_daily_picks.md`，可查看完整 Markdown 和候选详情。
- 知识库：展示 Qdrant collection、embedding、RAG 源和资料核验标签。
- 持仓：选择 CSV/JSON 后预览持仓，并结合日报里的持仓分析进行诊断。

## Layout

```text
apps/desktop/
  package.json
  package-lock.json
  vite.config.ts
  tsconfig.json
  dist/renderer/        # Vite 构建产物，Electron 生产模式加载这里
  src/
    main.js
    preload.js
    renderer/
      index.html
      styles.css
      src/
        App.tsx
        main.tsx
        api/
        components/
        pages/
```

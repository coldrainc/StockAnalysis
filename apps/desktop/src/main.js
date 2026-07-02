const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs/promises");
const path = require("path");

const API_BASE_URL = process.env.STOCK_AGENT_API_URL || "http://127.0.0.1:18220";
const PROJECT_ROOT = path.resolve(__dirname, "../../..");
const DAILY_REPORT_PATH = path.join(
  PROJECT_ROOT,
  "knowledge_base",
  "a_share_technology",
  "daily",
  "latest_daily_picks.md"
);
const VECTOR_METADATA_PATH = path.join(PROJECT_ROOT, ".stock_agent", "vector_store.json");
const A_SHARE_INDEX_PATH = path.join(PROJECT_ROOT, ".stock_agent", "A股_rag_index.json");
const DEFAULT_PORTFOLIO_PATH = path.join(PROJECT_ROOT, "holdings", "current_positions.csv");

function createWindow() {
  const win = new BrowserWindow({
    width: 1320,
    height: 880,
    minWidth: 1080,
    minHeight: 700,
    title: "Stock Agent",
    backgroundColor: "#f5f7fa",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  if (process.env.STOCK_AGENT_DESKTOP_DEV_URL) {
    win.loadURL(process.env.STOCK_AGENT_DESKTOP_DEV_URL);
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "renderer", "index.html"));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("api:health", async () => {
  return requestJson("/health");
});

ipcMain.handle("api:create-session", async (_event, payload) => {
  return requestJson("/sessions", {
    method: "POST",
    body: JSON.stringify(payload || {})
  });
});

ipcMain.handle("api:send-message", async (_event, payload) => {
  return requestJson(`/sessions/${payload.sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ message: payload.message })
  });
});

ipcMain.handle("desktop:workspace", async () => {
  return readWorkspaceSnapshot();
});

ipcMain.handle("desktop:choose-portfolio", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择持仓文件",
    properties: ["openFile"],
    filters: [
      { name: "Portfolio", extensions: ["csv", "json"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return readPortfolioPreview(result.filePaths[0]);
});

ipcMain.handle("desktop:run-daily-picks", async (_event, payload) => {
  return runDailyPicks(payload || {});
});

async function requestJson(route, options = {}) {
  const response = await fetch(`${API_BASE_URL}${route}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

async function readWorkspaceSnapshot() {
  const [report, vectorMetadata, indexMetadata, defaultPortfolio] = await Promise.all([
    readDailyReport(),
    readJsonIfExists(VECTOR_METADATA_PATH),
    readJsonIfExists(A_SHARE_INDEX_PATH),
    readPortfolioPreviewIfExists(DEFAULT_PORTFOLIO_PATH)
  ]);
  return {
    projectRoot: PROJECT_ROOT,
    defaultPortfolio,
    dailyReport: report,
    vectorStore: {
      provider: vectorMetadata?.vector_store || "unknown",
      collection: vectorMetadata?.collection_name || "-",
      market: vectorMetadata?.market || "-",
      sourceDir: vectorMetadata?.source_dir || "-",
      embeddingProvider: vectorMetadata?.embedding_provider || "-",
      embeddingModel: vectorMetadata?.embedding_model || "-",
      chunkCount: vectorMetadata?.chunk_count || indexMetadata?.chunk_count || 0
    }
  };
}

async function readDailyReport() {
  try {
    const [content, stat] = await Promise.all([
      fs.readFile(DAILY_REPORT_PATH, "utf-8"),
      fs.stat(DAILY_REPORT_PATH)
    ]);
    return {
      exists: true,
      path: DAILY_REPORT_PATH,
      modifiedAt: stat.mtime.toISOString(),
      content,
      ...parseDailyReport(content)
    };
  } catch (_error) {
    return {
      exists: false,
      path: DAILY_REPORT_PATH,
      content: "",
      generatedAt: "",
      scannedCount: 0,
      topK: 0,
      tags: [],
      picks: []
    };
  }
}

async function readJsonIfExists(filePath) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf-8"));
  } catch (_error) {
    return null;
  }
}

function parseDailyReport(content) {
  const generatedAt = extractListValue(content, "生成时间");
  const scannedCount = parseInt(extractListValue(content, "本次扫描数量") || "0", 10) || 0;
  const topK = parseInt(extractListValue(content, "输出 Top K") || "0", 10) || 0;
  const tagLine = extractListValue(content, "资料标签");
  const tags = tagLine ? tagLine.split(/\s+/).filter(Boolean) : [];
  const picks = [];
  const sectionPattern = /###\s+\d+\.\s+(.+?)（(.+?)）\n([\s\S]*?)(?=\n###\s+\d+\.|\n##\s+|$)/g;
  let match;
  while ((match = sectionPattern.exec(content))) {
    const block = match[3];
    picks.push({
      name: match[1].trim(),
      code: match[2].trim(),
      score: extractBullet(block, "评分"),
      rating: extractBullet(block, "等级"),
      price: extractBullet(block, "最新价"),
      pctChange: extractBullet(block, "涨跌幅"),
      amount: extractBullet(block, "成交额"),
      turnover: extractBullet(block, "换手率"),
      logic: extractBullet(block, "推荐逻辑"),
      risk: extractBullet(block, "风险提示")
    });
  }
  const portfolioSection = extractSection(content, "持仓分析");
  return { generatedAt, scannedCount, topK, tags, picks, portfolioSection };
}

function extractListValue(content, label) {
  const pattern = new RegExp(`^- ${escapeRegExp(label)}：(.+)$`, "m");
  const match = content.match(pattern);
  return match ? match[1].trim() : "";
}

function extractBullet(content, label) {
  return extractListValue(content, label);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function extractSection(content, heading) {
  const pattern = new RegExp(`## ${escapeRegExp(heading)}\\n([\\s\\S]*?)(?=\\n## |$)`);
  const match = content.match(pattern);
  return match ? match[1].trim() : "";
}

async function readPortfolioPreview(filePath) {
  const content = await fs.readFile(filePath, "utf-8");
  const ext = path.extname(filePath).toLowerCase();
  const rows = ext === ".json" ? parsePortfolioJson(content) : parsePortfolioCsv(content);
  return {
    path: filePath,
    rows: rows.slice(0, 80),
    totalRows: rows.length
  };
}

async function readPortfolioPreviewIfExists(filePath) {
  try {
    return await readPortfolioPreview(filePath);
  } catch (_error) {
    return null;
  }
}

function parsePortfolioJson(content) {
  const payload = JSON.parse(content);
  const rows = Array.isArray(payload) ? payload : payload.positions || [];
  return rows.filter((row) => row && typeof row === "object").map(normalizePortfolioRow);
}

function parsePortfolioCsv(content) {
  const lines = content.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length <= 1) return [];
  const headers = splitCsvLine(lines[0]).map((item) => item.trim());
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || "";
    });
    return normalizePortfolioRow(row);
  });
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"' && line[index + 1] === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function normalizePortfolioRow(row) {
  return {
    code: firstValue(row, ["code", "ticker", "股票代码"]),
    name: firstValue(row, ["name", "公司", "股票名称"]),
    shares: firstValue(row, ["shares", "持仓数量"]),
    costPrice: firstValue(row, ["cost_price", "cost", "成本价"]),
    marketValue: firstValue(row, ["market_value", "value", "持仓市值"]),
    notes: firstValue(row, ["notes", "备注"])
  };
}

function firstValue(row, keys) {
  for (const key of keys) {
    if (row[key] !== undefined && row[key] !== null && String(row[key]).trim()) {
      return String(row[key]).trim();
    }
  }
  return "";
}

function runDailyPicks(payload) {
  const maxCandidates = Number(payload.maxCandidates || 800);
  const topK = Number(payload.topK || 30);
  const syncVectorStore = payload.syncVectorStore !== false;
  const portfolio = String(payload.portfolio || "").trim();
  const args = [
    "daily-picks",
    "--max-candidates",
    String(maxCandidates),
    "--top-k",
    String(topK)
  ];
  if (portfolio) {
    args.push("--portfolio", portfolio);
  }
  if (!syncVectorStore) {
    args.push("--no-sync-vector-store");
  }

  return new Promise((resolve, reject) => {
    const child = spawn(path.join(PROJECT_ROOT, "stock"), args, {
      cwd: PROJECT_ROOT,
      env: process.env
    });
    let output = "";
    let errorOutput = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("daily-picks 执行超时"));
    }, 15 * 60 * 1000);

    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      errorOutput += chunk.toString();
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", async (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(errorOutput || output || `daily-picks exited with ${code}`));
        return;
      }
      resolve({
        output,
        workspace: await readWorkspaceSnapshot()
      });
    });
  });
}

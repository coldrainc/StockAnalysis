const messages = document.getElementById("messages");
const composer = document.getElementById("composer");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newSessionBtn = document.getElementById("newSessionBtn");
const offlineToggle = document.getElementById("offlineToggle");
const webSearchToggle = document.getElementById("webSearchToggle");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const embeddingText = document.getElementById("embeddingText");
const qdrantText = document.getElementById("qdrantText");
const vectorText = document.getElementById("vectorText");
const dailyText = document.getElementById("dailyText");
const sessionText = document.getElementById("sessionText");
const modePill = document.getElementById("modePill");
const emptyStartBtn = document.getElementById("emptyStartBtn");
const choosePortfolioBtn = document.getElementById("choosePortfolioBtn");
const portfolioPath = document.getElementById("portfolioPath");
const syncVectorToggle = document.getElementById("syncVectorToggle");
const runDailyBtn = document.getElementById("runDailyBtn");
const dailyGeneratedAt = document.getElementById("dailyGeneratedAt");
const scannedCount = document.getElementById("scannedCount");
const topKCount = document.getElementById("topKCount");
const chunkCount = document.getElementById("chunkCount");
const picksList = document.getElementById("picksList");
const openDailyBtn = document.getElementById("openDailyBtn");
const pageTitle = document.getElementById("pageTitle");
const reportPickList = document.getElementById("reportPickList");
const reportMeta = document.getElementById("reportMeta");
const reportDetail = document.getElementById("reportDetail");
const reportMarkdown = document.getElementById("reportMarkdown");
const askReportBtn = document.getElementById("askReportBtn");
const kbProvider = document.getElementById("kbProvider");
const kbCollection = document.getElementById("kbCollection");
const kbMarket = document.getElementById("kbMarket");
const kbEmbedding = document.getElementById("kbEmbedding");
const kbChunks = document.getElementById("kbChunks");
const kbSourceDir = document.getElementById("kbSourceDir");
const tagList = document.getElementById("tagList");
const portfolioAnalyzeBtn = document.getElementById("portfolioAnalyzeBtn");
const portfolioSummary = document.getElementById("portfolioSummary");
const portfolioTable = document.getElementById("portfolioTable");
const portfolioReport = document.getElementById("portfolioReport");

let sessionId = null;
let busy = false;
let selectedPortfolio = "";
let selectedPortfolioPreview = null;
let workspaceSnapshot = null;
let activePage = "dashboard";

checkHealth();
loadWorkspace();

newSessionBtn.addEventListener("click", async () => {
  await createSession();
});

emptyStartBtn.addEventListener("click", async () => {
  await createSession();
});

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendMessage();
});

choosePortfolioBtn.addEventListener("click", async () => {
  const selected = await window.stockAgent.choosePortfolio();
  if (!selected) return;
  selectedPortfolioPreview = selected;
  selectedPortfolio = selected.path;
  portfolioPath.textContent = compactPath(selected.path);
  portfolioPath.title = selected.path;
  renderPortfolio(selected);
  showPage("portfolio");
});

runDailyBtn.addEventListener("click", async () => {
  await refreshDailyPicks();
});

openDailyBtn.addEventListener("click", async () => {
  showPage("report");
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", async () => {
    await sendPrompt(button.dataset.prompt);
  });
});

document.querySelectorAll(".nav-btn").forEach((button) => {
  button.addEventListener("click", () => showPage(button.dataset.page));
});

askReportBtn.addEventListener("click", async () => {
  await sendPrompt("请解读当前桌面端展示的每日报告，输出今日强关注 Top、候选分层、资料核验提醒、触发条件和失效条件。");
});

portfolioAnalyzeBtn.addEventListener("click", async () => {
  await sendPrompt("请基于我刚选择的持仓文件和最新 daily-picks 做持仓诊断，重点分析浮盈浮亏、仓位集中、主题暴露、候选池匹配度和加减仓前置条件。");
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

async function checkHealth() {
  try {
    const health = await window.stockAgent.health();
    if (health.app !== "stock-agent" || health.service !== "stock-agent-api") {
      throw new Error("18220 端口不是 Stock Agent API，请停止冲突服务后重新运行 ./stock desktop");
    }
    statusDot.className = "status-dot ok";
    statusText.textContent = "已连接";
    embeddingText.textContent = health.embedding_service_url || "-";
    qdrantText.textContent = health.qdrant_url || "-";
  } catch (error) {
    statusDot.className = "status-dot fail";
    statusText.textContent = "API 未启动";
    embeddingText.textContent = "请先运行 ./stock api";
    qdrantText.textContent = error.message;
  }
}

async function loadWorkspace() {
  try {
    workspaceSnapshot = await window.stockAgent.workspace();
    renderWorkspace(workspaceSnapshot);
  } catch (error) {
    dailyText.textContent = `读取失败：${error.message}`;
    renderEmptyPicks("无法读取最新日报。");
  }
}

function renderWorkspace(snapshot) {
  const report = snapshot.dailyReport || {};
  const vector = snapshot.vectorStore || {};
  vectorText.textContent = `${vector.provider || "-"} / ${vector.collection || "-"}`;
  dailyText.textContent = report.exists ? formatDateTime(report.modifiedAt) : "未生成";
  dailyGeneratedAt.textContent = report.generatedAt || "-";
  scannedCount.textContent = formatInteger(report.scannedCount);
  topKCount.textContent = formatInteger(report.topK);
  chunkCount.textContent = formatInteger(vector.chunkCount);
  renderPicks(report.picks || []);
  renderReport(report);
  renderKnowledge(snapshot);
  renderPortfolioReport(report.portfolioSection || "");
}

function renderPicks(picks) {
  picksList.innerHTML = "";
  if (!picks.length) {
    renderEmptyPicks("还没有每日候选。");
    return;
  }
  picks.slice(0, 6).forEach((pick, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "pick-card";
    card.title = `${pick.name} ${pick.code}`;
    card.addEventListener("click", async () => {
      await sendPrompt(
        `请详细分析 ${pick.name}（${pick.code}）：解释它进入今日候选池的评分逻辑、上涨触发条件、失效条件、主要风险、资料核验点，并说明是否适合加入观察仓。`
      );
    });

    const header = document.createElement("div");
    header.className = "pick-head";
    header.innerHTML = `<span>${index + 1}. ${escapeHtml(pick.name)}</span><strong>${escapeHtml(pick.score || "-")}</strong>`;

    const meta = document.createElement("div");
    meta.className = "pick-meta";
    meta.textContent = `${pick.code} · ${pick.rating || "-"} · ${pick.pctChange || "-"}`;

    const quote = document.createElement("div");
    quote.className = "pick-quote";
    quote.innerHTML = `<span>${escapeHtml(pick.price || "-")}</span><span>${escapeHtml(pick.amount || "-")}</span>`;

    const risk = document.createElement("div");
    risk.className = "pick-risk";
    risk.textContent = pick.risk || "需核验公告和行情";

    card.appendChild(header);
    card.appendChild(meta);
    card.appendChild(quote);
    card.appendChild(risk);
    picksList.appendChild(card);
  });
}

function renderReport(report) {
  const picks = report.picks || [];
  reportMeta.textContent = report.exists
    ? `${report.generatedAt || "-"} · 扫描 ${formatInteger(report.scannedCount)} · Top ${formatInteger(report.topK)}`
    : "未生成每日报告";
  renderReportPickList(picks);
  renderReportDetail(picks[0]);
  reportMarkdown.innerHTML = report.content
    ? renderMarkdown(report.content)
    : `<div class="empty-picks">还没有可展示的每日报告。</div>`;
}

function renderReportPickList(picks) {
  reportPickList.innerHTML = "";
  if (!picks.length) {
    reportPickList.innerHTML = `<div class="empty-picks">暂无候选。</div>`;
    return;
  }
  picks.forEach((pick, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "report-pick";
    button.innerHTML = `
      <span>${index + 1}. ${escapeHtml(pick.name)} <small>${escapeHtml(pick.code)}</small></span>
      <strong>${escapeHtml(pick.score || "-")}</strong>
    `;
    button.addEventListener("click", () => renderReportDetail(pick));
    reportPickList.appendChild(button);
  });
}

function renderReportDetail(pick) {
  if (!pick) {
    reportDetail.innerHTML = `<div class="empty-picks">请选择候选股票。</div>`;
    return;
  }
  reportDetail.innerHTML = `
    <div class="detail-card">
      <div class="detail-head">
        <div>
          <h3>${escapeHtml(pick.name)}（${escapeHtml(pick.code)}）</h3>
          <p>${escapeHtml(pick.rating || "-")} · ${escapeHtml(pick.pctChange || "-")} · ${escapeHtml(pick.turnover || "-")}</p>
        </div>
        <strong>${escapeHtml(pick.score || "-")}</strong>
      </div>
      <dl class="detail-list compact-list">
        <div><dt>最新价</dt><dd>${escapeHtml(pick.price || "-")}</dd></div>
        <div><dt>成交额</dt><dd>${escapeHtml(pick.amount || "-")}</dd></div>
        <div><dt>推荐逻辑</dt><dd>${escapeHtml(pick.logic || "-")}</dd></div>
        <div><dt>风险提示</dt><dd>${escapeHtml(pick.risk || "需核验公告和行情")}</dd></div>
      </dl>
    </div>
  `;
}

function renderKnowledge(snapshot) {
  const vector = snapshot.vectorStore || {};
  const report = snapshot.dailyReport || {};
  kbProvider.textContent = vector.provider || "-";
  kbCollection.textContent = vector.collection || "-";
  kbMarket.textContent = vector.market || "-";
  kbEmbedding.textContent = `${vector.embeddingProvider || "-"} / ${vector.embeddingModel || "-"}`;
  kbChunks.textContent = formatInteger(vector.chunkCount);
  kbSourceDir.textContent = vector.sourceDir || "-";
  const tags = report.tags?.length
    ? report.tags
    : ["#quant_candidate", "#third_party_dataset", "#needs_verification", "#needs_refresh"];
  tagList.innerHTML = tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
}

function renderPortfolio(preview) {
  if (!preview) {
    portfolioSummary.textContent = "未选择持仓文件。";
    portfolioTable.innerHTML = `<div class="empty-picks">选择 CSV/JSON 后在这里预览。</div>`;
    return;
  }
  portfolioSummary.textContent = `${compactPath(preview.path)} · ${preview.totalRows} 条持仓记录`;
  if (!preview.rows.length) {
    portfolioTable.innerHTML = `<div class="empty-picks">文件中没有可识别的持仓行。</div>`;
    return;
  }
  const rows = preview.rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.code || "-")}</td>
          <td>${escapeHtml(row.name || "-")}</td>
          <td>${escapeHtml(row.shares || "-")}</td>
          <td>${escapeHtml(row.costPrice || "-")}</td>
          <td>${escapeHtml(row.notes || "-")}</td>
        </tr>
      `
    )
    .join("");
  portfolioTable.innerHTML = `
    <table>
      <thead>
        <tr><th>代码</th><th>名称</th><th>数量</th><th>成本价</th><th>备注</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderPortfolioReport(markdown) {
  portfolioReport.innerHTML = markdown
    ? renderMarkdown(markdown)
    : `<div class="empty-picks">日报中还没有持仓分析。选择持仓文件并刷新候选池后会出现在这里。</div>`;
}

function renderEmptyPicks(text) {
  picksList.innerHTML = `<div class="empty-picks">${escapeHtml(text)}</div>`;
}

async function createSession() {
  if (busy) return;
  setBusy(true, "正在创建会话...");
  clearMessages();
  try {
    const response = await window.stockAgent.createSession({
      offline: offlineToggle.checked,
      web_search: webSearchToggle.checked
    });
    sessionId = response.session_id;
    sessionText.textContent = `会话 ${sessionId.slice(0, 8)}`;
    modePill.textContent = offlineToggle.checked ? "离线模式" : "模型模式";
    addMessage("agent", response.message);
    renderGuardrails(response.guardrails);
  } catch (error) {
    addMessage("system", `创建会话失败：${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || busy) return;
  if (!(await ensureSession())) return;

  input.value = "";
  input.style.height = "auto";
  await postMessage(text);
}

async function sendPrompt(text) {
  if (!text || busy) return;
  if (!(await ensureSession())) return;
  await postMessage(text);
}

async function ensureSession() {
  if (sessionId) return true;
  await createSession();
  return Boolean(sessionId);
}

async function postMessage(text) {
  addMessage("user", text);
  setBusy(true, "量化分析师正在研究...");

  try {
    const response = await window.stockAgent.sendMessage({
      sessionId,
      message: text
    });
    addMessage("agent", response.message);
    renderGuardrails(response.guardrails);
    if (response.completed) {
      modePill.textContent = "已完成";
    }
  } catch (error) {
    addMessage("system", `发送失败：${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function refreshDailyPicks() {
  if (busy) return;
  setBusy(true, "正在刷新每日候选池...");
  runDailyBtn.disabled = true;
  try {
    const response = await window.stockAgent.runDailyPicks({
      portfolio: selectedPortfolio,
      maxCandidates: 800,
      topK: 30,
      syncVectorStore: syncVectorToggle.checked
    });
    workspaceSnapshot = response.workspace;
    renderWorkspace(workspaceSnapshot);
    addMessage(
      "system",
      `每日候选池已刷新。${syncVectorToggle.checked ? "A股 RAG/向量库已同步。" : "已跳过向量库同步。"}`
    );
  } catch (error) {
    addMessage("system", `刷新失败：${error.message}`);
  } finally {
    runDailyBtn.disabled = false;
    setBusy(false);
  }
}

function showPage(page) {
  activePage = page || "dashboard";
  document.querySelectorAll(".page").forEach((item) => {
    item.classList.toggle("active", item.id === `page-${activePage}`);
  });
  document.querySelectorAll(".nav-btn").forEach((item) => {
    item.classList.toggle("active", item.dataset.page === activePage);
  });
  const titles = {
    dashboard: "股票量化分析与推荐",
    report: "每日推荐报告",
    knowledge: "RAG 知识库状态",
    portfolio: "持仓分析"
  };
  pageTitle.textContent = titles[activePage] || titles.dashboard;
}

function addMessage(role, text) {
  removeEmptyState();
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : role === "system" ? "!" : "AI";

  const stack = document.createElement("div");
  stack.className = "bubble-stack";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = `${roleLabel(role)} · ${currentTime()}`;

  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = text;

  stack.appendChild(meta);
  stack.appendChild(item);
  row.appendChild(avatar);
  row.appendChild(stack);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
}

function renderGuardrails(guardrails) {
  if (!guardrails || guardrails.length === 0) return;
  addMessage("system", `Harness 护栏：${guardrails.join("；")}`);
}

function setBusy(nextBusy, label = "") {
  busy = nextBusy;
  sendBtn.disabled = nextBusy;
  newSessionBtn.disabled = nextBusy;
  emptyStartBtn.disabled = nextBusy;
  choosePortfolioBtn.disabled = nextBusy;
  runDailyBtn.disabled = nextBusy;
  if (nextBusy) {
    removeTyping();
    const typing = document.createElement("div");
    typing.className = "typing";
    typing.dataset.typing = "true";
    typing.textContent = label;
    messages.appendChild(typing);
    messages.scrollTop = messages.scrollHeight;
  } else {
    removeTyping();
  }
}

function removeTyping() {
  const typing = messages.querySelector("[data-typing='true']");
  if (typing) typing.remove();
}

function clearMessages() {
  messages.innerHTML = "";
}

function removeEmptyState() {
  const empty = messages.querySelector(".empty-state");
  if (empty) empty.remove();
}

function roleLabel(role) {
  if (role === "user") return "用户";
  if (role === "system") return "系统";
  return "分析师";
}

function currentTime() {
  return new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit"
  });
}

function formatDateTime(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function formatInteger(value) {
  const number = Number(value || 0);
  return number ? number.toLocaleString("zh-CN") : "-";
}

function compactPath(value) {
  if (!value) return "";
  const parts = value.split("/");
  if (parts.length <= 3) return value;
  return `${parts.at(-2)}/${parts.at(-1)}`;
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const html = [];
  let inList = false;
  const closeList = () => {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  };
  for (const line of lines) {
    if (!line.trim()) {
      closeList();
      continue;
    }
    if (line.startsWith("# ")) {
      closeList();
      html.push(`<h1>${inlineMarkdown(line.slice(2))}</h1>`);
    } else if (line.startsWith("## ")) {
      closeList();
      html.push(`<h2>${inlineMarkdown(line.slice(3))}</h2>`);
    } else if (line.startsWith("### ")) {
      closeList();
      html.push(`<h3>${inlineMarkdown(line.slice(4))}</h3>`);
    } else if (line.startsWith("- ")) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${inlineMarkdown(line.slice(2))}</li>`);
    } else {
      closeList();
      html.push(`<p>${inlineMarkdown(line)}</p>`);
    }
  }
  closeList();
  return html.join("");
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/(#(?:[a-zA-Z_]+))/g, "<span class=\"tag-inline\">$1</span>");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

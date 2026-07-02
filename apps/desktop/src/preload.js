const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("stockAgent", {
  health: () => ipcRenderer.invoke("api:health"),
  createSession: (payload) => ipcRenderer.invoke("api:create-session", payload),
  sendMessage: (payload) => ipcRenderer.invoke("api:send-message", payload),
  workspace: () => ipcRenderer.invoke("desktop:workspace"),
  choosePortfolio: () => ipcRenderer.invoke("desktop:choose-portfolio"),
  runDailyPicks: (payload) => ipcRenderer.invoke("desktop:run-daily-picks", payload)
});

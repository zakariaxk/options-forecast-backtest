const API = "/api/v1";

const NAV = [
  { id: "overview", label: "Overview" },
  { id: "predict", label: "Predict" },
  { id: "explore", label: "Explore" },
  { id: "backtest", label: "Backtest" },
];

const state = {
  models: [],
  predictionRuns: [],
  backtestRuns: [],
  lastPrediction: null,
  lastBacktest: null,
};

function el(id) {
  return document.getElementById(id);
}

function setJson(target, obj) {
  target.textContent = JSON.stringify(obj ?? {}, null, 2);
}

async function apiFetch(path, options) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const contentType = res.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) payload = await res.json();
  else payload = await res.text();
  if (!res.ok) {
    const detail = payload?.detail ?? payload;
    const message =
      typeof detail === "string" ? detail : detail?.message ?? JSON.stringify(detail ?? payload);
    throw new Error(message);
  }
  return payload;
}

function activePage() {
  const hash = (location.hash || "#overview").replace("#", "");
  return NAV.some((x) => x.id === hash) ? hash : "overview";
}

function renderNav() {
  const nav = el("nav");
  const page = activePage();
  nav.innerHTML = NAV.map(
    (item) => `<a href="#${item.id}" class="${item.id === page ? "active" : ""}">${item.label}</a>`,
  ).join("");
}

function showPage() {
  const page = activePage();
  for (const item of NAV) {
    const section = el(`page-${item.id}`);
    if (!section) continue;
    section.classList.toggle("hidden", item.id !== page);
  }
  renderNav();
}

async function refreshHealth() {
  const pill = el("healthPill");
  pill.className = "pill";
  pill.textContent = "Checking API…";
  try {
    const data = await apiFetch("/health");
    pill.classList.add("pill--ok");
    pill.textContent = `API OK · ${new Date(data.time).toLocaleString()}`;
  } catch (err) {
    pill.classList.add("pill--bad");
    pill.textContent = `API ERROR · ${err.message}`;
  }
}

function renderModels() {
  const wrap = el("modelsList");
  if (!state.models.length) {
    wrap.innerHTML = `<div class="item">No models found</div>`;
    return;
  }
  wrap.innerHTML = state.models
    .map(
      (m) => `
        <div class="item">
          <div><strong>${m.name}</strong></div>
          <div class="meta">${m.runs?.length ? `runs: ${m.runs.join(", ")}` : "no runs"}</div>
        </div>
      `,
    )
    .join("");
}

function renderRuns() {
  const predWrap = el("predictionRuns");
  const btWrap = el("backtestRuns");

  predWrap.innerHTML =
    state.predictionRuns.length === 0
      ? `<div class="item">No prediction runs</div>`
      : state.predictionRuns
          .slice(-30)
          .reverse()
          .map((r) => `<div class="item"><strong>${r.symbol}</strong> · ${r.model_name} · ${r.prediction_run_id}</div>`)
          .join("");

  btWrap.innerHTML =
    state.backtestRuns.length === 0
      ? `<div class="item">No backtest runs</div>`
      : state.backtestRuns
          .slice(-30)
          .reverse()
          .map((r) => `<div class="item"><strong>${r.symbol}</strong> · ${r.bt_id}</div>`)
          .join("");
}

function renderModelSelects() {
  const modelSelect = el("modelSelect");
  const exploreModel = el("exploreModel");
  const names = state.models.map((m) => m.name);
  const options = names.map((n) => `<option value="${n}">${n}</option>`).join("");
  modelSelect.innerHTML = options || `<option value="xgb_reg">xgb_reg</option>`;
  exploreModel.innerHTML = options || `<option value="xgb_reg">xgb_reg</option>`;
}

function renderExploreSelectors() {
  const symbols = [...new Set(state.predictionRuns.map((r) => r.symbol))].sort();
  const exploreSymbol = el("exploreSymbol");
  exploreSymbol.innerHTML = symbols.map((s) => `<option value="${s}">${s}</option>`).join("");
  if (!exploreSymbol.value && symbols.length) exploreSymbol.value = symbols[0];
  updateExploreRuns();
}

function updateExploreRuns() {
  const symbol = el("exploreSymbol").value;
  const model = el("exploreModel").value;
  const runs = state.predictionRuns
    .filter((r) => r.symbol === symbol && r.model_name === model)
    .map((r) => r.prediction_run_id)
    .sort();
  const exploreRun = el("exploreRun");
  exploreRun.innerHTML = runs.map((r) => `<option value="${r}">${r}</option>`).join("");
  if (!exploreRun.value && runs.length) exploreRun.value = runs[runs.length - 1];
}

function renderBacktestPredictionSelect() {
  const select = el("btPredRun");
  const items = state.predictionRuns.slice().reverse();
  select.innerHTML = items
    .map(
      (r) =>
        `<option value="${r.symbol}||${r.model_name}||${r.prediction_run_id}">${r.symbol} · ${r.model_name} · ${r.prediction_run_id}</option>`,
    )
    .join("");
}

function renderTable(container, rows) {
  if (!rows || rows.length === 0) {
    container.innerHTML = `<div class="muted" style="padding: 12px;">No rows</div>`;
    return;
  }
  const cols = Object.keys(rows[0]);
  const header = cols.map((c) => `<th>${c}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`)
    .join("");
  container.innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

async function refreshData() {
  state.models = await apiFetch("/models");
  state.predictionRuns = await apiFetch("/predictions/runs");
  state.backtestRuns = await apiFetch("/backtests/runs");
  renderModels();
  renderRuns();
  renderModelSelects();
  renderExploreSelectors();
  renderBacktestPredictionSelect();
}

function bindEvents() {
  window.addEventListener("hashchange", showPage);

  el("predictRefresh").addEventListener("click", async () => {
    await refreshData();
  });

  el("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    const payload = {
      symbol: String(form.get("symbol") || "").trim(),
      feature_version: String(form.get("feature_version") || "v1").trim(),
      model_name: String(form.get("model_name") || "").trim() || null,
      model_run_id: String(form.get("model_run_id") || "").trim() || null,
      prediction_run_id: String(form.get("prediction_run_id") || "").trim() || null,
      as_of_date: String(form.get("as_of_date") || "").trim() || null,
    };
    try {
      const data = await apiFetch("/predictions", { method: "POST", body: JSON.stringify(payload) });
      state.lastPrediction = data;
      setJson(el("predictResult"), data);
      await refreshData();
    } catch (err) {
      setJson(el("predictResult"), { error: err.message });
    }
  });

  el("openPrediction").addEventListener("click", () => {
    if (!state.lastPrediction) return;
    location.hash = "#explore";
    setTimeout(() => {
      el("exploreSymbol").value = state.lastPrediction?.symbol ?? el("exploreSymbol").value;
      el("exploreModel").value = state.lastPrediction?.model_name ?? el("exploreModel").value;
      updateExploreRuns();
      el("exploreRun").value = state.lastPrediction?.prediction_run_id ?? el("exploreRun").value;
    }, 0);
  });

  el("exploreSymbol").addEventListener("change", updateExploreRuns);
  el("exploreModel").addEventListener("change", updateExploreRuns);

  el("exploreLoad").addEventListener("click", async () => {
    const symbol = el("exploreSymbol").value;
    const model = el("exploreModel").value;
    const run = el("exploreRun").value;
    const limit = parseInt(el("exploreLimit").value || "200", 10);
    const wrap = el("exploreTableWrap");
    wrap.innerHTML = `<div class="muted" style="padding: 12px;">Loading…</div>`;
    try {
      const data = await apiFetch(
        `/predictions/${encodeURIComponent(symbol)}/${encodeURIComponent(run)}/data?model_name=${encodeURIComponent(model)}&limit=${encodeURIComponent(
          String(limit),
        )}`,
      );
      renderTable(wrap, data.rows);
    } catch (err) {
      wrap.innerHTML = `<div class="muted" style="padding: 12px;">Error: ${err.message}</div>`;
    }
  });

  el("btRefresh").addEventListener("click", async () => {
    await refreshData();
  });

  el("backtestForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    const predSel = String(el("btPredRun").value || "");
    const [predSymbol, predModel, predRun] = predSel.split("||");
    try {
      const predMeta = await apiFetch(
        `/predictions/${encodeURIComponent(predSymbol)}/${encodeURIComponent(predRun)}?model_name=${encodeURIComponent(predModel)}`,
      );
      const predictions_uri = predMeta.predictions_uri;
      const config = {
        name: `ui_bt_${new Date().toISOString().slice(0, 10)}`,
        symbol: String(form.get("symbol") || "").trim(),
        strategy: String(form.get("strategy") || "straddle").trim(),
        start_date: String(form.get("start_date") || "").trim(),
        end_date: String(form.get("end_date") || "").trim(),
        data: { predictions_uri },
      };
      const data = await apiFetch("/backtests", { method: "POST", body: JSON.stringify({ config }) });
      state.lastBacktest = { ...data, symbol: config.symbol };
      setJson(el("backtestResult"), state.lastBacktest);
      await refreshData();
    } catch (err) {
      setJson(el("backtestResult"), { error: err.message });
    }
  });

  el("loadBacktest").addEventListener("click", async () => {
    if (!state.lastBacktest?.bt_id || !state.lastBacktest?.symbol) return;
    try {
      const data = await apiFetch(
        `/backtests/${encodeURIComponent(state.lastBacktest.symbol)}/${encodeURIComponent(state.lastBacktest.bt_id)}/data`,
      );
      setJson(el("btMetrics"), data.metrics);
      renderTable(el("btTrades"), data.trades || []);
    } catch (err) {
      setJson(el("btMetrics"), { error: err.message });
      el("btTrades").innerHTML = `<div class="muted" style="padding: 12px;">Error: ${err.message}</div>`;
    }
  });
}

async function main() {
  renderNav();
  showPage();
  bindEvents();
  await refreshHealth();
  try {
    await refreshData();
  } catch (err) {
    // show errors in overview lists
    el("modelsList").innerHTML = `<div class="item">Error: ${err.message}</div>`;
  }
}

main();


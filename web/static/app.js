/* ═══════════════════════════════════════════════════════════
   Stockpulse — app.js
   Tabs: Backtest · Forecast · Options Chain
   ═══════════════════════════════════════════════════════════ */

const API = "/api/v1";

// ── Helpers ─────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const show = el => el.classList.remove("hidden");
const hide = el => el.classList.add("hidden");

function fmt(n, d = 2) {
  if (n == null || isNaN(n)) return "\u2014";
  return Number(n).toFixed(d);
}
function pct(n) {
  if (n == null || isNaN(n)) return "\u2014";
  return (Number(n) * 100).toFixed(2) + "%";
}
function money(n) {
  if (n == null || isNaN(n)) return "\u2014";
  return "$" + Number(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

async function apiFetch(path, opts) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const payload = await res.json();
  if (!res.ok) {
    const msg = payload?.message || payload?.detail?.message || JSON.stringify(payload);
    throw new Error(msg);
  }
  return payload;
}

// ── Health check ────────────────────────────────────────────

async function checkHealth() {
  const pill = $("healthPill");
  try {
    await apiFetch("/health");
    pill.className = "pill pill--ok";
    pill.textContent = "connected";
  } catch {
    pill.className = "pill pill--bad";
    pill.textContent = "offline";
  }
}

// ── Tabs ────────────────────────────────────────────────────

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");

      document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
      const panel = $("panel-" + tab.dataset.tab);
      if (panel) panel.classList.add("active");
    });
  });
}

// ════════════════════════════════════════════════════════════
//  BACKTEST
// ════════════════════════════════════════════════════════════

const BT_METRICS = [
  { key: "total_return", label: "Return", format: pct },
  { key: "cagr",         label: "CAGR",   format: pct },
  { key: "sharpe",       label: "Sharpe", format: fmt },
  { key: "sortino",      label: "Sortino", format: fmt },
  { key: "max_drawdown", label: "Max DD", format: pct },
  { key: "volatility",   label: "Vol",    format: pct },
  { key: "calmar",       label: "Calmar", format: fmt },
];

function renderBtMetrics(summary) {
  $("btMetrics").innerHTML = BT_METRICS.map(({ key, label, format }) => {
    const v = summary[key];
    const cls = v > 0 ? "up" : v < 0 ? "down" : "";
    return `<div class="metric-chip"><div class="m-label">${label}</div><div class="m-value ${cls}">${format(v)}</div></div>`;
  }).join("");
}

let btChart = null;

function renderEquityCurve(curve) {
  const canvas = $("equityChart");
  if (btChart) btChart.destroy();

  btChart = new Chart(canvas, {
    type: "line",
    data: {
      labels: curve.map(p => p.date.slice(0, 10)),
      datasets: [{
        label: "NAV",
        data: curve.map(p => p.nav),
        borderColor: "#e2a84b",
        backgroundColor: "rgba(226, 168, 75, 0.06)",
        fill: true,
        tension: 0.15,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: chartOpts(v => money(v)),
  });
}

function renderBtTrades(trades) {
  const el = $("btTrades");
  if (!trades.length) { el.innerHTML = `<p style="color:var(--text3);font-size:0.85rem">No trades executed.</p>`; return; }
  const cols = Object.keys(trades[0]);
  el.innerHTML = `<div class="table-scroll"><table>
    <thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
    <tbody>${trades.map(t => `<tr>${cols.map(c => `<td>${t[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>`;
}

function updateParamVisibility() {
  const strategy = $("bt-strategy").value;
  const sma = $("smaParams");
  const rsi = $("rsiParams");
  if (sma) strategy === "sma_crossover" ? show(sma) : hide(sma);
  if (rsi) strategy === "rsi_mean_reversion" ? show(rsi) : hide(rsi);
}

function getBtParams(form) {
  const s = form.get("strategy");
  if (s === "sma_crossover") return { fast_period: parseInt(form.get("fast_period")) || 20, slow_period: parseInt(form.get("slow_period")) || 50 };
  if (s === "rsi_mean_reversion") return { rsi_period: parseInt(form.get("rsi_period")) || 14, oversold: parseFloat(form.get("oversold")) || 30, overbought: parseFloat(form.get("overbought")) || 70 };
  return {};
}

async function handleBacktest(e) {
  e.preventDefault();
  const form = new FormData(e.currentTarget);
  const btn = $("btSubmitBtn");
  const errBox = $("btError");

  hide(errBox); hide($("btResults"));
  btn.disabled = true; btn.textContent = "Running\u2026";

  try {
    const data = await apiFetch("/backtests/", {
      method: "POST",
      body: JSON.stringify({
        symbol: form.get("symbol").trim().toUpperCase(),
        strategy: form.get("strategy"),
        start_date: form.get("start_date"),
        end_date: form.get("end_date"),
        initial_cash: parseFloat(form.get("initial_cash")) || 100000,
        params: getBtParams(form),
      }),
    });
    renderBtMetrics(data.summary);
    renderEquityCurve(data.equity_curve);
    renderBtTrades(data.trades);
    show($("btResults"));
  } catch (err) {
    errBox.textContent = err.message;
    show(errBox);
  } finally {
    btn.disabled = false; btn.textContent = "Run backtest";
  }
}

// ════════════════════════════════════════════════════════════
//  FORECAST
// ════════════════════════════════════════════════════════════

let fcChart = null;

function renderFcMetrics(data) {
  const chips = [
    { label: "Method", value: data.method },
    { label: "Last Close", value: money(data.last_close) },
    { label: "Horizon", value: data.horizon_days + " days" },
    { label: "MAE", value: "$" + fmt(data.diagnostics.mae) },
    { label: "RMSE", value: "$" + fmt(data.diagnostics.rmse) },
  ];
  $("fcMetrics").innerHTML = chips.map(c =>
    `<div class="metric-chip"><div class="m-label">${c.label}</div><div class="m-value">${c.value}</div></div>`
  ).join("");
}

function renderForecastChart(data) {
  const canvas = $("forecastChart");
  if (fcChart) fcChart.destroy();

  const histLabels = data.historical_tail.map(p => p.date);
  const histPrices = data.historical_tail.map(p => p.price);

  const fcLabels = data.forecast.map(p => p.date);
  const fcPrices = data.forecast.map(p => p.price);
  const fcUpper  = data.forecast.map(p => p.upper);
  const fcLower  = data.forecast.map(p => p.lower);

  const allLabels = [...histLabels, ...fcLabels];
  const histData  = [...histPrices, ...new Array(fcLabels.length).fill(null)];
  const predData  = [...new Array(histLabels.length - 1).fill(null), histPrices[histPrices.length - 1], ...fcPrices];
  const upperData = [...new Array(histLabels.length - 1).fill(null), histPrices[histPrices.length - 1], ...fcUpper];
  const lowerData = [...new Array(histLabels.length - 1).fill(null), histPrices[histPrices.length - 1], ...fcLower];

  fcChart = new Chart(canvas, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: [
        {
          label: "Historical",
          data: histData,
          borderColor: "#e8e4de",
          backgroundColor: "transparent",
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 1.5,
        },
        {
          label: "Forecast",
          data: predData,
          borderColor: "#e2a84b",
          backgroundColor: "transparent",
          borderDash: [6, 3],
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Upper 95%",
          data: upperData,
          borderColor: "rgba(109, 170, 236, 0.3)",
          backgroundColor: "rgba(109, 170, 236, 0.06)",
          fill: "+1",
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 1,
        },
        {
          label: "Lower 95%",
          data: lowerData,
          borderColor: "rgba(109, 170, 236, 0.3)",
          backgroundColor: "transparent",
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 1,
        },
      ],
    },
    options: chartOpts(v => money(v)),
  });
}

async function handleForecast(e) {
  e.preventDefault();
  const form = new FormData(e.currentTarget);
  const btn = $("fcSubmitBtn");
  const errBox = $("fcError");

  hide(errBox); hide($("fcResults"));
  btn.disabled = true; btn.textContent = "Forecasting\u2026";

  try {
    const data = await apiFetch("/forecast/", {
      method: "POST",
      body: JSON.stringify({
        symbol: form.get("symbol").trim().toUpperCase(),
        method: form.get("method"),
        horizon_days: parseInt(form.get("horizon_days")) || 30,
        lookback_days: parseInt(form.get("lookback_days")) || 252,
      }),
    });
    renderFcMetrics(data);
    renderForecastChart(data);
    show($("fcResults"));
  } catch (err) {
    errBox.textContent = err.message;
    show(errBox);
  } finally {
    btn.disabled = false; btn.textContent = "Forecast";
  }
}

// ════════════════════════════════════════════════════════════
//  OPTIONS CHAIN
// ════════════════════════════════════════════════════════════

function renderOptMeta(data) {
  const chips = [
    { label: "Underlying", value: money(data.underlying_price) },
    { label: "Expiry", value: data.expiry },
    { label: "Calls", value: data.calls.length },
    { label: "Puts", value: data.puts.length },
  ];
  $("optMeta").innerHTML = chips.map(c =>
    `<div class="metric-chip"><div class="m-label">${c.label}</div><div class="m-value">${c.value}</div></div>`
  ).join("");
}

function renderOptTable(contracts, elId) {
  const el = $(elId);
  if (!contracts.length) { el.innerHTML = `<p style="color:var(--text3);font-size:0.85rem;padding:12px">No data</p>`; return; }

  const show_cols = ["strike", "last_price", "bid", "ask", "volume", "open_interest", "implied_vol", "itm"];
  const nice = { strike: "Strike", last_price: "Last", bid: "Bid", ask: "Ask", volume: "Vol", open_interest: "OI", implied_vol: "IV", itm: "ITM" };
  const cols = show_cols.filter(c => c in contracts[0]);

  el.innerHTML = `<table>
    <thead><tr>${cols.map(c => `<th>${nice[c] || c}</th>`).join("")}</tr></thead>
    <tbody>${contracts.map(row => {
      const isItm = row.itm;
      return `<tr>${cols.map(c => {
        let val = row[c];
        if (c === "implied_vol" && val != null) val = (val * 100).toFixed(1) + "%";
        if (c === "itm") val = val ? "Yes" : "";
        if (c === "strike" || c === "last_price" || c === "bid" || c === "ask") val = val != null ? val.toFixed(2) : "";
        const cls = (c === "itm" && isItm) ? ' class="itm"' : "";
        return `<td${cls}>${val ?? ""}</td>`;
      }).join("")}</tr>`;
    }).join("")}</tbody>
  </table>`;
}

function populateExpiryDropdown(expiries, current) {
  const sel = $("opt-expiry");
  sel.innerHTML = expiries.map(e =>
    `<option value="${e}" ${e === current ? "selected" : ""}>${e}</option>`
  ).join("");
}

async function handleOptions(e) {
  e.preventDefault();
  const form = new FormData(e.currentTarget);
  const btn = $("optSubmitBtn");
  const errBox = $("optError");

  hide(errBox); hide($("optResults"));
  btn.disabled = true; btn.textContent = "Loading\u2026";

  const expiry = form.get("expiry");

  try {
    const data = await apiFetch("/options/chain", {
      method: "POST",
      body: JSON.stringify({
        symbol: form.get("symbol").trim().toUpperCase(),
        expiry: expiry || null,
      }),
    });

    populateExpiryDropdown(data.available_expiries, data.expiry);
    renderOptMeta(data);
    renderOptTable(data.calls, "optCalls");
    renderOptTable(data.puts, "optPuts");
    show($("optResults"));
  } catch (err) {
    errBox.textContent = err.message;
    show(errBox);
  } finally {
    btn.disabled = false; btn.textContent = "Load chain";
  }
}

// ── Shared chart options ────────────────────────────────────

function chartOpts(yFmt) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#26262b",
        titleColor: "#e8e4de",
        bodyColor: "#b3afa6",
        borderColor: "#3a3a42",
        borderWidth: 1,
        padding: 10,
        cornerRadius: 6,
        displayColors: false,
        callbacks: {
          label: ctx => (ctx.dataset.label || "") + ": " + yFmt(ctx.parsed.y),
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#7d796f", maxTicksLimit: 10, font: { size: 10, family: "'Inter'" } },
        grid: { color: "rgba(46,46,53,0.5)" },
      },
      y: {
        ticks: { color: "#7d796f", font: { size: 10, family: "'Inter'" }, callback: v => yFmt(v) },
        grid: { color: "rgba(46,46,53,0.5)" },
      },
    },
  };
}

// ── Init ────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  initTabs();

  // Backtest
  $("backtestForm").addEventListener("submit", handleBacktest);
  $("bt-strategy").addEventListener("change", updateParamVisibility);
  updateParamVisibility();

  // Forecast
  $("forecastForm").addEventListener("submit", handleForecast);

  // Options
  $("optionsForm").addEventListener("submit", handleOptions);
});

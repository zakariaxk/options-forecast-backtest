/* Backtest Platform — UI (vanilla JS, no build step) */

const API = "/api/v1";

// ── Helpers ─────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function show(el)  { el.classList.remove("hidden"); }
function hide(el)  { el.classList.add("hidden"); }

function fmt(n, decimals = 2) {
  if (n == null) return "—";
  return Number(n).toFixed(decimals);
}

function pct(n) {
  if (n == null) return "—";
  return (Number(n) * 100).toFixed(2) + "%";
}

function money(n) {
  if (n == null) return "—";
  return "$" + Number(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── API Call ────────────────────────────────────────────────

async function apiFetch(path, options) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await res.json();
  if (!res.ok) {
    const msg = payload?.message || payload?.detail?.message || JSON.stringify(payload);
    throw new Error(msg);
  }
  return payload;
}

// ── Health Check ────────────────────────────────────────────

async function checkHealth() {
  const pill = $("healthPill");
  try {
    const data = await apiFetch("/health");
    pill.className = "pill pill--ok";
    pill.textContent = "API OK";
  } catch {
    pill.className = "pill pill--bad";
    pill.textContent = "API Offline";
  }
}

// ── Render: Summary Metrics ─────────────────────────────────

const METRIC_CONFIG = [
  { key: "total_return", label: "Total Return", format: pct },
  { key: "cagr",         label: "CAGR",         format: pct },
  { key: "sharpe",       label: "Sharpe",       format: fmt },
  { key: "sortino",      label: "Sortino",      format: fmt },
  { key: "max_drawdown", label: "Max Drawdown", format: pct },
  { key: "volatility",   label: "Volatility",   format: pct },
  { key: "calmar",       label: "Calmar",       format: fmt },
];

function renderSummary(summary) {
  const grid = $("summaryGrid");
  grid.innerHTML = METRIC_CONFIG.map(({ key, label, format }) => {
    const val = summary[key];
    const formatted = format(val);
    const cls = val > 0 ? "positive" : val < 0 ? "negative" : "";
    return `
      <div class="metric-card">
        <div class="label">${label}</div>
        <div class="value ${cls}">${formatted}</div>
      </div>`;
  }).join("");
}

// ── Render: Equity Curve ────────────────────────────────────

let chartInstance = null;

function renderEquityCurve(equityCurve) {
  const canvas = $("equityChart");
  if (chartInstance) chartInstance.destroy();

  const labels = equityCurve.map(p => p.date.slice(0, 10));
  const data = equityCurve.map(p => p.nav);

  chartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "NAV ($)",
        data,
        borderColor: "#4f8ff7",
        backgroundColor: "rgba(79, 143, 247, 0.08)",
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => money(ctx.parsed.y),
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b8fa3", maxTicksLimit: 12, font: { size: 11 } },
          grid: { color: "rgba(42,45,58,0.5)" },
        },
        y: {
          ticks: {
            color: "#8b8fa3",
            font: { size: 11 },
            callback: v => "$" + v.toLocaleString(),
          },
          grid: { color: "rgba(42,45,58,0.5)" },
        },
      },
    },
  });
}

// ── Render: Trades Table ────────────────────────────────────

function renderTrades(trades) {
  const wrap = $("tradesTable");
  if (!trades.length) {
    wrap.innerHTML = `<p class="muted">No trades</p>`;
    return;
  }
  const cols = Object.keys(trades[0]);
  wrap.innerHTML = `
    <table>
      <thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
      <tbody>${trades.map(t =>
        `<tr>${cols.map(c => `<td>${t[c] ?? ""}</td>`).join("")}</tr>`
      ).join("")}</tbody>
    </table>`;
}

// ── Strategy Param Visibility ───────────────────────────────

function updateParamVisibility() {
  const strategy = $("strategySelect").value;
  const sma = $("smaParams");
  const rsi = $("rsiParams");
  if (sma) { strategy === "sma_crossover" ? show(sma) : hide(sma); }
  if (rsi) { strategy === "rsi_mean_reversion" ? show(rsi) : hide(rsi); }
}

function getStrategyParams(form) {
  const strategy = form.get("strategy");
  if (strategy === "sma_crossover") {
    return {
      fast_period: parseInt(form.get("fast_period")) || 20,
      slow_period: parseInt(form.get("slow_period")) || 50,
    };
  }
  if (strategy === "rsi_mean_reversion") {
    return {
      rsi_period: parseInt(form.get("rsi_period")) || 14,
      oversold: parseFloat(form.get("oversold")) || 30,
      overbought: parseFloat(form.get("overbought")) || 70,
    };
  }
  return {};
}

// ── Form Submit ─────────────────────────────────────────────

async function handleSubmit(e) {
  e.preventDefault();
  const form = new FormData(e.currentTarget);
  const btn = e.currentTarget.querySelector("button[type=submit]");
  const errorBox = $("errorBox");

  hide(errorBox);
  hide($("results"));
  btn.disabled = true;
  btn.textContent = "Running…";

  const payload = {
    symbol: form.get("symbol").trim().toUpperCase(),
    strategy: form.get("strategy"),
    start_date: form.get("start_date"),
    end_date: form.get("end_date"),
    initial_cash: parseFloat(form.get("initial_cash")) || 100000,
    params: getStrategyParams(form),
  };

  try {
    const data = await apiFetch("/backtests/", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    renderSummary(data.summary);
    renderEquityCurve(data.equity_curve);
    renderTrades(data.trades);
    show($("results"));
  } catch (err) {
    errorBox.textContent = err.message;
    show(errorBox);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Backtest";
  }
}

// ── Init ────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  $("backtestForm").addEventListener("submit", handleSubmit);
  $("strategySelect").addEventListener("change", updateParamVisibility);
  updateParamVisibility();
});

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from api.core.settings import get_settings
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from api.core.settings import get_settings
from dashboard.components import plots, tables
from dashboard.api_client import APIClient

st.set_page_config(page_title="Options Forecast Dashboard", layout="wide")
settings = get_settings()


def list_symbols(base: Path) -> list[str]:
    if not base.exists():
        return []
    return sorted([item.name for item in base.iterdir() if item.is_dir()])


def list_runs(base: Path) -> dict[str, list[str]]:
    runs: dict[str, list[str]] = {}
    for symbol_dir in base.iterdir() if base.exists() else []:
        if symbol_dir.is_dir():
            runs[symbol_dir.name] = sorted([child.name for child in symbol_dir.iterdir() if child.is_dir()])
    return runs


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _sidebar_config() -> dict:
    st.sidebar.header("Dashboard")
    base_url_default = st.session_state.get("api_base_url", "http://127.0.0.1:8000/api/v1")
    api_base_url = st.sidebar.text_input("API Base URL", value=base_url_default)
    st.session_state["api_base_url"] = api_base_url
    use_api = st.sidebar.toggle("Use API (recommended)", value=True)
    use_local = st.sidebar.toggle("Use local artifacts", value=True)
    if st.sidebar.button("Refresh cached data"):
        st.cache_data.clear()
    return {"api_base_url": api_base_url, "use_api": use_api, "use_local": use_local}


@st.cache_data(ttl=10)
def _api_health(api_base_url: str) -> bool:
    return APIClient(api_base_url).health_check()


@st.cache_data(ttl=30)
def _api_models(api_base_url: str) -> list[dict]:
    return APIClient(api_base_url).list_models()


@st.cache_data(ttl=30)
def _api_prediction_runs(api_base_url: str) -> list[dict]:
    return APIClient(api_base_url).list_prediction_runs()


@st.cache_data(ttl=30)
def _api_backtest_runs(api_base_url: str) -> list[dict]:
    return APIClient(api_base_url).list_backtest_runs()


def _status_bar(cfg: dict) -> None:
    left, right = st.columns([2, 1])
    with left:
        st.title("Options Forecast & Backtest")
        st.caption("Streamlit dashboard over stored artifacts + FastAPI endpoints.")
    with right:
        if cfg["use_api"]:
            ok = _api_health(cfg["api_base_url"])
            st.success("API online") if ok else st.error("API offline")
        else:
            st.info("API disabled")


def overview_page(cfg: dict):
    _status_bar(cfg)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Models")
        if cfg["use_api"]:
            models = _api_models(cfg["api_base_url"])
            st.metric("Discovered", len(models))
            if models:
                st.dataframe(pd.DataFrame(models), use_container_width=True)
        if cfg["use_local"]:
            models_dir = Path(settings.models_uri)
            local_models = list_symbols(models_dir)
            st.caption(f"Local: `{models_dir}`")
            st.write(local_models or "No local models")

    with col2:
        st.subheader("Prediction Runs")
        if cfg["use_api"]:
            runs = _api_prediction_runs(cfg["api_base_url"])
            st.metric("Discovered", len(runs))
            if runs:
                st.dataframe(pd.DataFrame(runs).tail(25), use_container_width=True)
        if cfg["use_local"]:
            predictions_dir = Path(settings.predictions_uri)
            local_runs = list_runs(predictions_dir)
            st.caption(f"Local: `{predictions_dir}`")
            st.write(local_runs or "No local prediction runs")

    with col3:
        st.subheader("Backtests")
        if cfg["use_api"]:
            runs = _api_backtest_runs(cfg["api_base_url"])
            st.metric("Discovered", len(runs))
            if runs:
                st.dataframe(pd.DataFrame(runs).tail(25), use_container_width=True)
        if cfg["use_local"]:
            backtests_dir = Path(settings.backtests_uri)
            local_symbols = list_symbols(backtests_dir)
            st.caption(f"Local: `{backtests_dir}`")
            st.write(local_symbols or "No local backtests")


def predictions_page(cfg: dict):
    _status_bar(cfg)
    st.header("Predictions")

    api_client = APIClient(cfg["api_base_url"])

    tab_browse, tab_create, tab_local = st.tabs(["Browse", "Create", "Local Files"])

    with tab_browse:
        if not cfg["use_api"]:
            st.info("Enable `Use API` in the sidebar to browse via API.")
        else:
            runs = _api_prediction_runs(cfg["api_base_url"])
            if not runs:
                st.warning("No prediction runs found. Generate one in the Create tab.")
            else:
                df_runs = pd.DataFrame(runs)
                symbols = sorted(df_runs["symbol"].unique().tolist())
                symbol = st.selectbox("Symbol", symbols)
                models = sorted(df_runs[df_runs["symbol"] == symbol]["model_name"].unique().tolist())
                model = st.selectbox("Model", models)
                run_ids = sorted(
                    df_runs[(df_runs["symbol"] == symbol) & (df_runs["model_name"] == model)]["prediction_run_id"]
                    .unique()
                    .tolist()
                )
                prediction_run_id = st.selectbox("Prediction Run", run_ids, index=len(run_ids) - 1)
                limit = st.slider("Rows", min_value=50, max_value=5000, value=200, step=50)
                option_filter = st.text_input("Filter option_key contains (optional)", value="")

                if st.button("Load", type="primary"):
                    with st.spinner("Loading predictions…"):
                        payload = api_client.get_prediction_data(
                            symbol=symbol,
                            model_name=model,
                            prediction_run_id=prediction_run_id,
                            limit=limit,
                        )
                    rows = payload.get("rows", [])
                    df = pd.DataFrame(rows)
                    if option_filter and not df.empty and "option_key" in df.columns:
                        df = df[df["option_key"].astype(str).str.contains(option_filter, case=False, na=False)]
                    st.subheader(f"Rows: {len(df)} (loaded {len(rows)})")
                    st.dataframe(df, use_container_width=True, height=520)
                    if not df.empty and "score" in df.columns:
                        st.plotly_chart(
                            px.histogram(df, x="score", nbins=40, title="Score Distribution"),
                            use_container_width=True,
                        )

    with tab_create:
        if not cfg["use_api"]:
            st.info("Enable `Use API` in the sidebar to create predictions via API.")
        else:
            models = _api_models(cfg["api_base_url"])
            model_names = [m["name"] for m in models] if models else ["xgb_reg"]

            st.subheader("Generate predictions")
            with st.form("predict_form", clear_on_submit=False):
                p_symbol = st.text_input("Symbol", value="AAPL")
                p_feature_version = st.text_input("Feature Version", value="v1")
                p_model = st.selectbox("Model Name", model_names)
                p_model_run_id = st.text_input("Model Run ID (optional; defaults to latest)", value="")
                p_prediction_run_id = st.text_input("Prediction Run ID (optional)", value="")
                p_as_of_date = st.text_input("As-of Date (optional, YYYY-MM-DD)", value="")

                submitted = st.form_submit_button("Generate", type="primary")
                if submitted:
                    payload: dict = {
                        "symbol": p_symbol.strip().upper(),
                        "model_name": p_model,
                        "feature_version": p_feature_version.strip(),
                    }
                    if p_model_run_id.strip():
                        payload["model_run_id"] = p_model_run_id.strip()
                    if p_prediction_run_id.strip():
                        payload["prediction_run_id"] = p_prediction_run_id.strip()
                    if p_as_of_date.strip():
                        payload["as_of_date"] = p_as_of_date.strip()
                    with st.spinner("Calling API…"):
                        try:
                            res = api_client.create_prediction(payload)
                            st.success(
                                f"Done: {res.get('symbol')} · {res.get('model_name')} · {res.get('prediction_run_id')}"
                            )
                            st.json(res)
                            st.cache_data.clear()
                        except Exception as exc:
                            st.error(str(exc))

    with tab_local:
        if not cfg["use_local"]:
            st.info("Enable `Use local artifacts` in the sidebar to browse from disk.")
        else:
            base = Path(settings.predictions_uri)
            symbols = list_symbols(base)
            symbol = st.selectbox("Symbol", symbols) if symbols else None
            if not symbol:
                st.info("No predictions available on disk.")
                return
            models = list_symbols(base / symbol)
            model = st.selectbox("Model", models)
            runs = list_symbols(base / symbol / model)
            run = st.selectbox("Run", runs)
            if run:
                path = base / symbol / model / run / "predictions.parquet"
                if path.exists():
                    df = load_parquet(path)
                    st.subheader(f"Rows: {len(df)}")
                    st.dataframe(df.head(2000), use_container_width=True, height=520)
                else:
                    st.error(f"File not found: {path}")


def backtest_page(cfg: dict):
    _status_bar(cfg)
    st.header("Backtests")

    api_client = APIClient(cfg["api_base_url"])

    tab_browse, tab_create, tab_local = st.tabs(["Browse", "Create", "Local Files"])

    with tab_browse:
        if not cfg["use_api"]:
            st.info("Enable `Use API` in the sidebar to browse via API.")
        else:
            runs = _api_backtest_runs(cfg["api_base_url"])
            if not runs:
                st.warning("No backtests found. Create one in the Create tab.")
            else:
                df_runs = pd.DataFrame(runs)
                symbols = sorted(df_runs["symbol"].unique().tolist())
                symbol = st.selectbox("Symbol", symbols, key="bt_browse_symbol")
                bt_ids = sorted(df_runs[df_runs["symbol"] == symbol]["bt_id"].unique().tolist())
                bt_id = st.selectbox("Backtest Run", bt_ids, index=len(bt_ids) - 1, key="bt_browse_id")

                col_a, col_b, col_c = st.columns([1, 1, 1])
                with col_a:
                    limit_trades = st.slider("Trades (tail)", 0, 5000, 500, 50)
                with col_b:
                    limit_equity = st.slider("Equity points (tail)", 0, 20000, 2000, 250)
                with col_c:
                    load = st.button("Load", type="primary", key="bt_load")

                if load:
                    with st.spinner("Loading backtest…"):
                        payload = api_client.get_backtest_data(
                            symbol=symbol,
                            bt_id=bt_id,
                            limit_trades=limit_trades,
                            limit_equity=limit_equity,
                        )
                    metrics = payload.get("metrics") or {}
                    trades_df = pd.DataFrame(payload.get("trades") or [])
                    equity_df = pd.DataFrame(payload.get("equity") or [])

                    st.subheader("Metrics")
                    st.dataframe(tables.metrics_table(metrics), use_container_width=True, height=260)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(plots.equity_curve(equity_df), use_container_width=True)
                        st.plotly_chart(plots.daily_returns(equity_df), use_container_width=True)
                    with c2:
                        st.plotly_chart(plots.drawdown_curve(equity_df), use_container_width=True)
                        st.plotly_chart(plots.rolling_sharpe(equity_df, window=20), use_container_width=True)

                    st.subheader("Trades (tail)")
                    st.dataframe(tables.trades_table(trades_df), use_container_width=True, height=520)

    with tab_create:
        if not cfg["use_api"]:
            st.info("Enable `Use API` in the sidebar to create backtests via API.")
        else:
            pred_runs = _api_prediction_runs(cfg["api_base_url"])
            st.subheader("Run a new backtest")

            with st.form("bt_form", clear_on_submit=False):
                b_symbol = st.text_input("Symbol", value="AAPL")
                b_start = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
                b_end = st.date_input("End Date", value=pd.to_datetime("2023-03-31"))
                b_strategy = st.selectbox("Strategy", ["straddle", "credit_spread", "covered_call"])
                st.caption("Backtests require a `predictions.parquet` artifact. Select an existing prediction run:")

                if pred_runs:
                    labels = [
                        f'{r["symbol"]} · {r["model_name"]} · {r["prediction_run_id"]}' for r in pred_runs[::-1]
                    ]
                    selected = st.selectbox("Prediction Run", labels)
                    pick = pred_runs[::-1][labels.index(selected)]
                else:
                    st.warning("No prediction runs found via API. Generate predictions first.")
                    pick = None

                advanced = st.toggle("Advanced config", value=False)
                if advanced:
                    col_u1, col_u2, col_u3 = st.columns(3)
                    with col_u1:
                        dte_min = st.number_input("Universe: DTE min", min_value=0, value=7, step=1)
                        dte_max = st.number_input("Universe: DTE max", min_value=0, value=30, step=1)
                    with col_u2:
                        max_spread = st.number_input("Universe: max spread %", min_value=0.0, value=0.05, step=0.01)
                        min_oi = st.number_input("Universe: min OI", min_value=0, value=100, step=50)
                    with col_u3:
                        top_k = st.number_input("Signal: top K", min_value=1, value=10, step=1)
                        max_notional = st.number_input("Risk: max gross notional", min_value=1000.0, value=100000.0, step=5000.0)

                submitted = st.form_submit_button("Run Backtest", type="primary", disabled=(pick is None))
                if submitted and pick is not None:
                    try:
                        meta = api_client.get_prediction_meta(
                            symbol=pick["symbol"],
                            model_name=pick["model_name"],
                            prediction_run_id=pick["prediction_run_id"],
                        )
                        predictions_uri = meta.get("predictions_uri")
                        if not predictions_uri:
                            raise RuntimeError("API did not return predictions_uri for selected run")

                        config: dict = {
                            "name": f"ui_{b_strategy}_{b_symbol}_{b_start}_{b_end}",
                            "symbol": b_symbol.strip().upper(),
                            "start_date": str(b_start),
                            "end_date": str(b_end),
                            "strategy": b_strategy,
                            "data": {"predictions_uri": predictions_uri},
                        }
                        if advanced:
                            config["universe"] = {
                                "dte_min": int(dte_min),
                                "dte_max": int(dte_max),
                                "min_oi": int(min_oi),
                                "max_spread_pct": float(max_spread),
                            }
                            config["signal"] = {"select_top_k": int(top_k)}
                            config["risk"] = {
                                "max_gross_notional": float(max_notional),
                                "max_position_per_option": 10,
                                "stop_loss_pct": 0.3,
                                "take_profit_pct": 0.5,
                            }

                        with st.spinner("Submitting backtest…"):
                            res = api_client.submit_backtest({"config": config})
                        st.success(f"Done: {res.get('bt_id')} ({res.get('status')})")
                        st.json(res)
                        st.cache_data.clear()
                    except Exception as exc:
                        st.error(str(exc))

    with tab_local:
        if not cfg["use_local"]:
            st.info("Enable `Use local artifacts` in the sidebar to browse from disk.")
        else:
            base = Path(settings.backtests_uri)
            symbols = list_symbols(base)
            symbol = st.selectbox("Symbol", symbols, key="bt_local_sym") if symbols else None
            if not symbol:
                st.info("No backtests found on disk.")
                return
            runs = list_symbols(base / symbol)
            run = st.selectbox("Run", runs, key="bt_local_run")
            if run:
                run_dir = base / symbol / run
                trades = load_parquet(run_dir / "trades.parquet")
                equity = load_parquet(run_dir / "equity.parquet")
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists():
                    st.subheader("Metrics")
                    st.json(metrics_path.read_text())
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(plots.equity_curve(equity), use_container_width=True)
                    st.plotly_chart(plots.daily_returns(equity), use_container_width=True)
                with c2:
                    st.plotly_chart(plots.drawdown_curve(equity), use_container_width=True)
                    st.plotly_chart(plots.rolling_sharpe(equity, window=20), use_container_width=True)
                st.subheader("Trades")
                st.dataframe(tables.trades_table(trades), use_container_width=True, height=520)


def reports_page(cfg: dict):
    st.title("Reports")
    st.caption("Quick links to local artifact directories (parquet/JSON).")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predictions")
        st.code(str(Path(settings.predictions_uri).resolve()))
    with col2:
        st.subheader("Backtests")
        st.code(str(Path(settings.backtests_uri).resolve()))


pages = {
    "Overview": overview_page,
    "Predictions": predictions_page,
    "Backtests": backtest_page,
    "Reports": reports_page,
}


cfg = _sidebar_config()

tab_overview, tab_pipelines, tab_predictions, tab_backtests, tab_reports = st.tabs(
    ["Overview", "Pipelines", "Predictions", "Backtests", "Reports"]
)

with tab_overview:
    overview_page(cfg)
    if cfg["use_api"] and not _api_health(cfg["api_base_url"]):
        st.info(
            "To enable full functionality (create/browse runs), start the API: `source .venv/bin/activate && make api`"
        )
    models_dir = Path(settings.models_uri)
    predictions_dir = Path(settings.predictions_uri)
    backtests_dir = Path(settings.backtests_uri)
    local_model_count = len(list_symbols(models_dir))
    local_pred_count = sum(len(v) for v in list_runs(predictions_dir).values())
    local_bt_count = sum(len(list_symbols(backtests_dir / s)) for s in list_symbols(backtests_dir))
    if local_model_count == 0 and local_pred_count == 0 and local_bt_count == 0:
        st.subheader("Getting started")
        st.code(
            "\n".join(
                [
                    "source .venv/bin/activate",
                    "make ingest SYMBOL=AAPL START=2020-01-01 END=2020-12-31",
                    "make features SYMBOL=AAPL PARTITION=\"AAPL/2020-01-01_2020-12-31\" VERSION=v1",
                    "make train SYMBOL=AAPL MODEL=xgb_reg RUN=demo VERSION=v1",
                    "make predict SYMBOL=AAPL MODEL=xgb_reg RUN=demo VERSION=v1",
                    "make backtest SYMBOL=AAPL START=2020-01-01 END=2020-12-31 PREDICTIONS=data/predictions/AAPL/xgb_reg/demo/predictions.parquet",
                ]
            )
        )

with tab_pipelines:
    _status_bar(cfg)
    st.header("Pipelines (end-to-end)")
    if not cfg["use_api"]:
        st.info("Enable `Use API` in the sidebar to run pipelines from the dashboard.")
    else:
        api_client = APIClient(cfg["api_base_url"])
        with st.expander("1) Ingest (Yahoo Finance)", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                p_symbol = st.text_input("Symbol", value="AAPL", key="pipe_symbol")
            with col2:
                p_start = st.text_input("Start Date", value="2024-01-01", key="pipe_start")
            with col3:
                p_end = st.text_input("End Date", value="2024-03-01", key="pipe_end")
            col4, col5 = st.columns(2)
            with col4:
                p_min_oi = st.number_input("Min OI", min_value=0, value=100, step=50, key="pipe_min_oi")
            with col5:
                p_min_vol = st.number_input("Min Volume", min_value=0, value=0, step=5, key="pipe_min_vol")

            if st.button("Run Ingest", type="primary", key="pipe_ingest"):
                with st.spinner("Ingesting… (requires network access for yfinance)"):
                    try:
                        res = api_client.create_ingestion(
                            {
                                "symbol": p_symbol.strip().upper(),
                                "start_date": p_start.strip(),
                                "end_date": p_end.strip(),
                                "min_open_interest": int(p_min_oi),
                                "min_volume": int(p_min_vol),
                            }
                        )
                        st.session_state["last_ingest"] = res
                        st.success(f'Ingested: {res.get("partition")}')
                        st.json(res)
                        st.cache_data.clear()
                    except Exception as exc:
                        st.error(str(exc))

        with st.expander("2) Features", expanded=True):
            last_ingest = st.session_state.get("last_ingest") or {}
            default_partition = last_ingest.get("partition", "")
            col1, col2, col3 = st.columns(3)
            with col1:
                f_symbol = st.text_input("Symbol", value=st.session_state.get("pipe_symbol", "AAPL"), key="feat_symbol")
            with col2:
                f_partition = st.text_input("Raw Partition", value=default_partition, key="feat_partition")
            with col3:
                f_version = st.text_input("Feature Version", value="v1", key="feat_version")
            col4, col5 = st.columns(2)
            with col4:
                f_horizon = st.number_input("Horizon Days", min_value=1, value=5, step=1, key="feat_horizon")
            with col5:
                f_thresh = st.number_input("Classification Threshold", value=0.0, step=0.01, key="feat_thresh")
            if st.button("Run Features", type="primary", key="pipe_features"):
                with st.spinner("Building features…"):
                    try:
                        res = api_client.create_features(
                            {
                                "symbol": f_symbol.strip().upper(),
                                "raw_partition": f_partition.strip(),
                                "version": f_version.strip(),
                                "horizon_days": int(f_horizon),
                                "classification_threshold": float(f_thresh),
                            }
                        )
                        st.session_state["last_features"] = res
                        st.success(f'Features built: {res.get("features_uri")}')
                        st.json(res)
                        st.cache_data.clear()
                    except Exception as exc:
                        st.error(str(exc))

        with st.expander("3) Train", expanded=True):
            last_features = st.session_state.get("last_features") or {}
            col1, col2, col3 = st.columns(3)
            with col1:
                t_symbol = st.text_input("Symbol", value=st.session_state.get("feat_symbol", "AAPL"), key="train_symbol")
            with col2:
                t_feature_version = st.text_input(
                    "Feature Version",
                    value=st.session_state.get("feat_version", "v1"),
                    key="train_feature_version",
                )
            with col3:
                t_run = st.text_input("Model Run Name", value="demo", key="train_run")
            col4, col5 = st.columns(2)
            with col4:
                t_model = st.selectbox("Model", ["xgb_reg", "xgb_clf", "torch_lstm"], key="train_model")
            with col5:
                t_target = st.selectbox("Target", ["regression", "classification"], key="train_target")
            if st.button("Run Train", type="primary", key="pipe_train"):
                with st.spinner("Training…"):
                    try:
                        res = api_client.create_train(
                            {
                                "symbol": t_symbol.strip().upper(),
                                "feature_version": t_feature_version.strip(),
                                "run_name": t_run.strip(),
                                "model_name": t_model,
                                "target": t_target,
                            }
                        )
                        st.session_state["last_train"] = res
                        st.success(f'Trained: {res.get("model_name")} · {res.get("run_name")}')
                        st.json(res)
                        st.cache_data.clear()
                    except Exception as exc:
                        st.error(str(exc))

        with st.expander("4) Predict + Backtest (one-click)", expanded=True):
            st.caption("Uses the trained model run as the prediction model_run_id.")
            col1, col2 = st.columns(2)
            with col1:
                e_symbol = st.text_input("Symbol", value=st.session_state.get("train_symbol", "AAPL"), key="e_symbol")
                e_start = st.text_input("Backtest Start", value=st.session_state.get("pipe_start", "2024-01-01"), key="e_start")
                e_end = st.text_input("Backtest End", value=st.session_state.get("pipe_end", "2024-03-01"), key="e_end")
            with col2:
                e_model = st.text_input("Model Name", value=st.session_state.get("train_model", "xgb_reg"), key="e_model")
                e_model_run = st.text_input("Model Run ID", value=st.session_state.get("train_run", "demo"), key="e_model_run")
                e_strategy = st.selectbox("Strategy", ["straddle", "credit_spread", "covered_call"], key="e_strategy")

            if st.button("Run Predict + Backtest", type="primary", key="pipe_e2e"):
                try:
                    with st.spinner("Generating predictions…"):
                        pred = api_client.create_prediction(
                            {
                                "symbol": e_symbol.strip().upper(),
                                "model_name": e_model.strip(),
                                "model_run_id": e_model_run.strip(),
                                "feature_version": st.session_state.get("train_feature_version", "v1"),
                            }
                        )
                        st.session_state["last_prediction"] = pred
                        st.json(pred)

                    with st.spinner("Resolving predictions artifact path…"):
                        meta = api_client.get_prediction_meta(
                            symbol=e_symbol.strip().upper(),
                            model_name=e_model.strip(),
                            prediction_run_id=pred["prediction_run_id"],
                        )
                        predictions_uri = meta["predictions_uri"]

                    with st.spinner("Running backtest…"):
                        cfg_bt = {
                            "name": f"ui_bt_{e_strategy}_{e_symbol}_{e_start}_{e_end}",
                            "symbol": e_symbol.strip().upper(),
                            "start_date": e_start.strip(),
                            "end_date": e_end.strip(),
                            "strategy": e_strategy,
                            "data": {"predictions_uri": predictions_uri},
                            "universe": {
                                "dte_min": 5,
                                "dte_max": 60,
                                "moneyness_min": -0.2,
                                "moneyness_max": 0.2,
                                "min_oi": 0,
                                "max_spread_pct": 1.0,
                            },
                        }
                        bt = api_client.submit_backtest({"config": cfg_bt})
                        st.session_state["last_backtest"] = bt
                        st.success(f'Backtest done: {bt.get("bt_id")}')
                        st.json(bt)
                        st.cache_data.clear()
                        st.info("Now open the Predictions / Backtests tabs to browse the generated runs.")
                except Exception as exc:
                    st.error(str(exc))

with tab_predictions:
    predictions_page(cfg)

with tab_backtests:
    backtest_page(cfg)

with tab_reports:
    reports_page(cfg)

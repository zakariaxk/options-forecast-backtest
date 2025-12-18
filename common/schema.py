from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator

SEED_DEFAULT = 1337


class IngestConfig(BaseModel):
    symbol: str
    start_date: date
    end_date: date
    dest_uri: str = "data/raw"
    min_open_interest: int = 100
    min_volume: int = 10
    max_expiries: Optional[int] = None
    max_contracts_per_expiry: Optional[int] = None
    seed: int = SEED_DEFAULT

    @validator("end_date")
    def _validate_dates(cls, value: date, values):
        start = values.get("start_date")
        if start and value < start:
            raise ValueError("end_date must be on or after start_date")
        return value


class FeatureConfig(BaseModel):
    symbol: str
    version: str = "v1"
    raw_uri: str = "data/raw"
    raw_partition: str
    processed_uri: str = "data/processed"
    horizon_days: int = 5
    classification_threshold: float = 0.0


class TrainConfig(BaseModel):
    symbol: str
    feature_version: str
    run_name: str
    model_name: Literal["xgb_reg", "xgb_clf", "torch_lstm"] = "xgb_reg"
    target: Literal["regression", "classification"] = "regression"
    processed_uri: str = "data/processed"
    output_uri: str = "data/models"
    seed: int = SEED_DEFAULT
    seq_len: int = 32
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class PredictConfig(BaseModel):
    symbol: str
    model_name: str
    model_run_id: str
    prediction_run_id: str
    feature_version: str
    processed_uri: str = "data/processed"
    models_uri: str = "data/models"
    output_uri: str = "data/predictions"
    as_of_date: Optional[date] = None
    horizon: Optional[Literal["1d", "5d", "10d"]] = None


class RiskConfig(BaseModel):
    max_gross_notional: float
    max_position_per_option: int
    stop_loss_pct: float
    take_profit_pct: float


class ExecutionConfig(BaseModel):
    slippage_bps: int = 10
    commission_per_contract: float = 0.65
    price: Literal["mid", "bid", "ask"] = "mid"


class UniverseConfig(BaseModel):
    dte_min: int = 7
    dte_max: int = 30
    type: tuple[Literal["C", "P"], ...] = ("C", "P")
    moneyness_min: float = -0.05
    moneyness_max: float = 0.05
    min_oi: int = 100
    max_spread_pct: float = 0.05


class SignalConfig(BaseModel):
    select_top_k: int = 10
    side: Literal["buy", "sell"] = "buy"
    prediction_run: Optional[str] = None


class RebalanceConfig(BaseModel):
    frequency: Literal["daily", "weekly"] = "daily"
    exit_after_days: int = 5


class ReportConfig(BaseModel):
    save_trades: bool = True
    save_daily_equity: bool = True


class BacktestConfig(BaseModel):
    name: str
    symbol: str
    start_date: date
    end_date: date
    strategy: Optional[str] = None
    data: dict = Field(default_factory=dict)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(
        default_factory=lambda: RiskConfig(
            max_gross_notional=100000.0,
            max_position_per_option=10,
            stop_loss_pct=0.3,
            take_profit_pct=0.5,
        )
    )
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    reports: ReportConfig = Field(default_factory=ReportConfig)


class PredictRequest(BaseModel):
    symbol: str
    model_name: Optional[str] = None
    model_run_id: Optional[str] = None
    prediction_run_id: Optional[str] = None
    feature_version: str = "v1"
    as_of_date: Optional[date] = None
    horizon: Optional[Literal["1d", "5d", "10d"]] = None


class PredictResponse(BaseModel):
    prediction_run_id: str
    symbol: str
    model_name: str
    model_run_id: str
    n: int
    uri: str


class BacktestRequest(BaseModel):
    config_yaml: Optional[str] = None
    config: Optional[BacktestConfig] = None


class BacktestResponse(BaseModel):
    bt_id: str
    status: Literal["queued", "running", "done", "error"]


class IngestRequest(BaseModel):
    symbol: str
    start_date: date
    end_date: date
    min_open_interest: int = 100
    min_volume: int = 0
    max_expiries: Optional[int] = 6
    max_contracts_per_expiry: Optional[int] = 250


class IngestResponse(BaseModel):
    symbol: str
    partition: str
    equity_uri: str
    options_uri: str
    metadata_uri: str


class FeatureRequest(BaseModel):
    symbol: str
    raw_partition: str
    version: str = "v1"
    horizon_days: int = 5
    classification_threshold: float = 0.0


class FeatureResponse(BaseModel):
    symbol: str
    version: str
    features_uri: str
    schema_uri: str
    scaler_uri: str
    hash: str


class TrainRequest(BaseModel):
    symbol: str
    feature_version: str
    run_name: str
    model_name: Literal["xgb_reg", "xgb_clf", "torch_lstm"] = "xgb_reg"
    target: Literal["regression", "classification"] = "regression"
    seed: int = SEED_DEFAULT
    seq_len: int = 32
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class TrainResponse(BaseModel):
    symbol: str
    model_name: str
    run_name: str
    artifacts: dict

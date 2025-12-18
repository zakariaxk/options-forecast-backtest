from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


class APIClientError(RuntimeError):
    pass


class APIClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("OFB_API_BASE_URL") or "http://127.0.0.1:8000/api/v1").rstrip("/")

    def _url(self, path: str) -> str:
        path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._url(path), data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8") if exc.fp else ""
            message = body
            try:
                parsed = json.loads(body) if body else {}
                detail = parsed.get("detail")
                if isinstance(detail, dict) and "message" in detail:
                    message = detail["message"]
                elif isinstance(detail, str):
                    message = detail
                elif parsed:
                    message = json.dumps(parsed)
            except Exception:
                pass
            raise APIClientError(f"{exc.code} {method} {path}: {message}") from exc
        except urllib.error.URLError as exc:
            raise APIClientError(f"Network error calling {method} {path}: {exc.reason}") from exc

    def health_check(self) -> bool:
        try:
            self._request("GET", "/health")
            return True
        except APIClientError:
            return False

    def list_models(self) -> list[Dict[str, Any]]:
        return self._request("GET", "/models/") or []

    def list_prediction_runs(self) -> list[Dict[str, Any]]:
        return self._request("GET", "/predictions/runs") or []

    def get_prediction_meta(self, symbol: str, model_name: str, prediction_run_id: str) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/predictions/{symbol}/{prediction_run_id}?model_name={urllib.parse.quote(model_name)}",
        )

    def get_prediction_data(
        self,
        symbol: str,
        model_name: str,
        prediction_run_id: str,
        limit: int = 200,
        sort: str = "score",
        descending: bool = True,
    ) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/predictions/{symbol}/{prediction_run_id}/data?model_name={urllib.parse.quote(model_name)}&limit={limit}&sort={urllib.parse.quote(sort)}&descending={'true' if descending else 'false'}",
        )

    def create_prediction(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/predictions/", config)

    def list_backtest_runs(self) -> list[Dict[str, Any]]:
        return self._request("GET", "/backtests/runs") or []

    def submit_backtest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/backtests/", payload)

    def get_backtest_data(
        self,
        symbol: str,
        bt_id: str,
        limit_trades: int = 500,
        limit_equity: int = 2000,
    ) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/backtests/{symbol}/{bt_id}/data?limit_trades={limit_trades}&limit_equity={limit_equity}",
        )

    def create_ingestion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/pipelines/ingest", payload)

    def create_features(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/pipelines/features", payload)

    def create_train(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/pipelines/train", payload)

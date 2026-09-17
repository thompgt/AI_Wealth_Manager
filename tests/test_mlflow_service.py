"""Unit tests for MLflow experiment and metrics tracking service."""

import pytest
from services import mlflow_service
from config import settings


def test_mlflow_service_availability():
    """Verify is_available returns boolean without raising."""
    avail = mlflow_service.is_available()
    assert isinstance(avail, bool)


def test_mlflow_service_disabled(monkeypatch):
    """When MLFLOW_ENABLED is false, operations degrade to no-ops."""
    monkeypatch.setattr(settings, "MLFLOW_ENABLED", False)

    assert mlflow_service.is_available() is False
    assert mlflow_service.start_run("test-run-id") is None

    # None of these should raise
    mlflow_service.log_portfolio_metrics("test-run-id", {"sharpe_ratio": 1.25})
    mlflow_service.log_regime_inference("test-run-id", "Bull", 0.85)
    mlflow_service.log_llm_generation("test-run-id", "test_node", "gemini-test", 100, 50, 250.0)
    mlflow_service.end_run("test-run-id")


def test_mlflow_service_methods_fail_open(monkeypatch):
    """Verify service functions handle arbitrary exceptions without failing callers."""
    monkeypatch.setattr(settings, "MLFLOW_ENABLED", True)

    # Even with an empty or synthetic run ID, it should not raise
    mlflow_service.start_run("test-safe-run", tags={"env": "test"})
    mlflow_service.log_portfolio_metrics(
        "test-safe-run",
        {
            "sharpe_ratio": 1.5,
            "annual_return": 0.12,
            "annual_volatility": 0.15,
            "max_drawdown": -0.08,
            "portfolio_beta": 0.95,
            "diversification_score": 0.82,
        },
        weights={"AAPL": 0.4, "MSFT": 0.6},
    )
    mlflow_service.log_regime_inference(
        "test-safe-run",
        "Bull",
        0.88,
        signals={
            "ratio_signals": {"small_cap_vs_large_cap (IWM/SPY)": {"change_pct": 2.5}},
            "ticker_pct_change": {"SPY": 3.2, "^VIX": -5.1},
        },
    )
    mlflow_service.log_llm_generation(
        "test-safe-run",
        "stock_research",
        "gemini-2.5-flash",
        prompt_tokens=450,
        completion_tokens=120,
        latency_ms=850.0,
    )
    mlflow_service.end_run("test-safe-run", status="FINISHED")
